"""Rules engine: a balanced, hardware-aware axolotl config without any AI.

Used as (1) the fallback when no LLM is configured and (2) the baseline draft the
LLM refines, which keeps the AI anchored to numbers that actually fit the GPUs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import catalog
from .fetch import DatasetInfo, ModelInfo
from .hardware import Hardware
from .sampler import DatasetProfile

GiB = 1024**3


@dataclass
class Plan:
    method: str
    seq_len: int
    micro_batch: int
    grad_accum: int
    epochs: int
    lr: float
    sharding: str  # none | ddp | zero2 | zero3 | zero3_offload | fsdp2
    est_gb_per_gpu: float
    est_steps: int | None
    notes: list[str]

    def to_prompt(self) -> dict:
        return {
            "method": self.method,
            "sequence_len": self.seq_len,
            "micro_batch_size": self.micro_batch,
            "gradient_accumulation_steps": self.grad_accum,
            "num_epochs": self.epochs,
            "learning_rate": self.lr,
            "multi_gpu_strategy": self.sharding,
            "estimated_vram_gb_per_gpu": round(self.est_gb_per_gpu, 1),
            "estimated_optimizer_steps": self.est_steps,
            "notes": self.notes,
        }


def _mcfg(model: ModelInfo, *keys, default=0):
    return model._cfg(*keys, default=default) or default


def estimate_gb(
    model: ModelInfo, method: str, seq_len: int, micro_batch: int, shard: int = 1, cce: bool = True, optim_8bit: bool = False
) -> float:
    """Rough peak VRAM per GPU in GB (bf16, gradient checkpointing on)."""
    params = float(model.param_count or 7e9)
    hidden = int(_mcfg(model, "hidden_size", "d_model", default=4096))
    layers = int(_mcfg(model, "num_hidden_layers", "n_layer", default=32))
    vocab = int(_mcfg(model, "vocab_size", default=128000))
    if method == "full":
        static = params * (2 + 2 + (2 if optim_8bit else 8)) / shard
    elif method == "lora":
        static = params * 2 / shard + params * 0.005 * 16  # ~0.5% trainable at r=32
    else:  # qlora
        static = params * 0.6 / shard + params * 0.005 * 16
    tokens = seq_len * micro_batch
    activations = tokens * hidden * layers * 2 * 1.5 + tokens * hidden * 24
    logits = 0 if cce else tokens * vocab * 6
    return (static + activations + logits) * 1.1 / GiB + 1.5


def _round_seq(n: int) -> int:
    for size in (512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072):
        if n <= size:
            return size
    return 131072


def _goal_has(goal: str, *words: str) -> bool:
    g = goal.lower()
    return any(w in g for w in words)


def plan(settings, hw: Hardware, model: ModelInfo, prof: DatasetProfile) -> Plan:
    notes: list[str] = []
    n_gpu = max(hw.gpu_count, 1)
    vram = hw.min_vram_gb or 24.0
    budget = vram * 0.80  # headroom for fragmentation + eval spikes
    params_b = model.params_b or 7.0
    rows = prof.est_rows or 1000
    total_tokens_m = prof.lengths.get("est_total_tokens_millions") or rows * max(prof.lengths.get("mean", 512), 1) / 1e6
    cce_ok = model.model_type in catalog.CCE_MODELS

    # ── sequence length: cover p99 of real token lengths ───────────────────────
    max_pos = int(_mcfg(model, "max_position_embeddings", "max_seq_len", "seq_length", default=32768))
    want = _round_seq(max(int(prof.lengths.get("p99", 2048) * 1.05), 512))
    seq_cap = 32768 if not _goal_has(settings.goal, "long context", "128k", "64k") else 131072
    seq_len = min(want, max_pos, seq_cap)
    if prof.lengths.get("p99", 0) > seq_len:
        notes.append(f"p99 sample length {prof.lengths['p99']} > sequence_len {seq_len}: longest samples get truncated")

    # ── method ───────────────────────────────────────────────────────────────────────
    method = settings.method if settings.method in ("lora", "qlora", "full", "adapter") else "auto"
    if method == "adapter":  # setup screen: always a LoRA adapter (lora if bf16 weights fit, else qlora)
        fits = estimate_gb(model, "lora", seq_len, 1, shard=1, cce=cce_ok) < budget or (
            n_gpu > 1 and estimate_gb(model, "lora", seq_len, 1, shard=n_gpu, cce=cce_ok) < budget
        )
        method = "lora" if fits or model.quantization else "qlora"
        notes.append(f"adapter run: {method}")
    if method == "auto":
        if model.quantization:
            method = "lora"
            notes.append(f"base is pre-quantized ({model.quantization}) -> LoRA")
        else:
            shard = n_gpu if n_gpu > 1 else 1
            full_fits = estimate_gb(model, "full", seq_len, 1, shard=shard, cce=cce_ok) < budget
            big_data = total_tokens_m >= 50 or rows >= 50_000
            if full_fits and (params_b <= 1.0 or (big_data and params_b <= 14)):
                method = "full"
            elif estimate_gb(model, "lora", seq_len, 1, shard=1, cce=cce_ok) < budget:
                method = "lora"
            elif n_gpu > 1 and estimate_gb(model, "lora", seq_len, 1, shard=n_gpu, cce=cce_ok) < budget:
                method = "lora"
            else:
                method = "qlora"
        notes.append(f"method auto-picked: {method}")

    # ── multi-gpu strategy ─────────────────────────────────────────────────────────────
    sharding = "none"
    shard = 1
    if n_gpu > 1:
        single = estimate_gb(model, method, seq_len, 1, shard=1, cce=cce_ok)
        if single < budget:
            # plain DDP is fastest; ZeRO-2 only when a full fine-tune is tight on memory
            sharding = "zero2" if method == "full" and single > budget * 0.6 else "ddp"
        else:
            shard = n_gpu
            if method == "qlora":
                sharding = "fsdp2"
            elif estimate_gb(model, method, seq_len, 1, shard=n_gpu, cce=cce_ok) < budget:
                sharding = "zero3"
            else:
                sharding = "zero3_offload"
                notes.append("model does not fit even sharded: ZeRO-3 with CPU offload (slow but works)")

    # ── shrink sequence if even micro_batch 1 does not fit ─────────────────────────────
    while seq_len > 512 and estimate_gb(model, method, seq_len, 1, shard=shard, cce=cce_ok) > budget:
        seq_len //= 2
        notes.append(f"sequence_len reduced to {seq_len} to fit {vram:.0f} GB")

    # ── packing / micro batch ────────────────────────────────────────────────────────────
    packing = hw.min_cc >= 8.0 or not hw.gpus
    micro = 1
    if not packing:
        for mb in (2, 4, 8, 16):
            if estimate_gb(model, method, seq_len, mb, shard=shard, cce=cce_ok) < budget:
                micro = mb
    elif estimate_gb(model, method, seq_len, 2, shard=shard, cce=cce_ok) < budget * 0.7 and seq_len <= 4096:
        micro = 2

    # ── batch size: 16-64 (packed) sequences per optimizer step, keep >= ~150 steps ─────
    if packing:
        avg_fill = 0.9
    else:
        avg_fill = min(1.0, max(prof.lengths.get("mean", seq_len), 1) / seq_len)
    target_seqs = 32 if method == "full" else 16
    if seq_len <= 1024:
        target_seqs *= 2  # short rows carry few tokens each
    grad_accum = max(1, min(32, math.ceil(target_seqs / (micro * n_gpu))))

    epochs = 3 if rows < 2_000 else 2 if rows < 20_000 else 1
    if method == "full" and rows >= 2_000:
        epochs = min(epochs, 2)

    def steps_for(ga: int) -> int:
        per_step = seq_len * micro * n_gpu * ga * avg_fill
        return max(1, math.ceil(total_tokens_m * 1e6 * epochs / per_step))

    while grad_accum > 1 and steps_for(grad_accum) < 150:
        grad_accum = max(1, grad_accum // 2)
    est_steps = steps_for(grad_accum)

    # ── learning rate ───────────────────────────────────────────────────────────────────
    if method in ("lora", "qlora"):
        lr = 2e-4 if rows < 200_000 else 1e-4
    elif params_b >= 7:
        lr = 1e-5
    elif params_b >= 1:
        lr = 2e-5
    else:
        lr = 5e-5
    if prof.detected.get("rl"):
        lr = 5e-6 if method == "full" else 5e-5
        notes.append(f"preference data detected -> rl: {prof.detected['rl']} with a low learning rate")

    est = estimate_gb(model, method, seq_len, micro, shard=shard, cce=cce_ok)
    return Plan(method, seq_len, micro, grad_accum, epochs, lr, sharding, est, est_steps, notes)


def build_config(settings, hw: Hardware, model: ModelInfo, data: DatasetInfo, prof: DatasetProfile, run_name: str) -> tuple[dict, Plan]:
    p = plan(settings, hw, model, prof)
    feats = {f.key: f for f in catalog.features() if f.available}
    cfg: dict = {"base_model": model.base_model}
    if model.revision:
        cfg["revision_of_model"] = model.revision
    if model.config.get("auto_map"):
        cfg["trust_remote_code"] = True

    # ── data ───────────────────────────────────────────────────────────────────────────
    ds = dict(data.axolotl_entry())
    ds.update({k: v for k, v in prof.detected.get("axolotl", {}).items() if v is not None})
    if "type" not in ds:
        ds["type"] = "completion"
    cfg["datasets"] = [ds]
    if prof.detected.get("rl"):
        cfg["rl"] = prof.detected["rl"]
    if ds.get("type") == "chat_template" or str(ds.get("type", "")).startswith("chat_template"):
        cfg["chat_template"] = "tokenizer_default" if model.has_chat_template else "tokenizer_default_fallback_chatml"
    cfg["dataset_prepared_path"] = str(settings.runs_dir / run_name / "prepared")
    cfg["val_set_size"] = 0.02 if (prof.est_rows or 0) >= 5_000 else 0.0
    cfg["output_dir"] = str(settings.outputs_dir / run_name)
    cfg["dataset_num_proc"] = max(1, min(hw.cpu_count, 32))

    # ── model / adapter ─────────────────────────────────────────────────────────────────
    if p.method == "qlora":
        cfg["load_in_4bit"] = True
        cfg["adapter"] = "qlora"
    elif p.method == "lora":
        cfg["adapter"] = "lora"
    if p.method in ("lora", "qlora"):
        r = 16 if (prof.est_rows or 0) < 2_000 else 64 if (prof.est_rows or 0) > 200_000 else 32
        cfg.update({"lora_r": r, "lora_alpha": r * 2, "lora_dropout": 0.05 if (prof.est_rows or 0) < 50_000 else 0.0, "lora_target_linear": True})

    # ── sequence / packing / attention ─────────────────────────────────────────────────
    cfg["sequence_len"] = p.seq_len
    flash = hw.min_cc >= 8.0 or not hw.gpus
    cfg["attn_implementation"] = "flash_attention_2" if flash else "sdpa"
    if not prof.detected.get("rl"):
        cfg["sample_packing"] = bool(flash)
        if flash and cfg["val_set_size"]:
            cfg["eval_sample_packing"] = True
    cfg["pad_to_sequence_len"] = bool(flash) and not prof.detected.get("rl")

    # ── batch / schedule ───────────────────────────────────────────────────────────────
    cfg.update(
        {
            "micro_batch_size": p.micro_batch,
            "gradient_accumulation_steps": p.grad_accum,
            "num_epochs": p.epochs,
            "optimizer": "paged_adamw_8bit" if p.method == "qlora" else "adamw_torch_fused",
            "lr_scheduler": "cosine",
            "learning_rate": p.lr,
            "warmup_ratio": 0.1 if (p.est_steps or 0) < 300 else 0.05,
            "weight_decay": 0.0 if p.method != "full" else 0.01,
            "max_grad_norm": 1.0,
        }
    )

    # ── precision / memory ─────────────────────────────────────────────────────────────
    if hw.bf16 or not hw.gpus:
        cfg["bf16"] = "auto"
        cfg["tf32"] = True
    else:
        cfg["bf16"] = False
        cfg["fp16"] = True
    cfg["gradient_checkpointing"] = True
    cfg["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    # ── plugins ──────────────────────────────────────────────────────────────────────
    plugins: list[str] = []
    cce = "cut_cross_entropy" in feats and model.model_type in catalog.CCE_MODELS
    if cce:
        plugins.append(feats["cut_cross_entropy"].plugin_class)
        cfg["cut_cross_entropy"] = True
    if "liger" in feats:
        plugins.append(feats["liger"].plugin_class)
        cfg.update(
            {
                "liger_rope": True,
                "liger_rms_norm": True,
                "liger_glu_activation": True,
                "liger_layer_norm": True,
                "liger_fused_linear_cross_entropy": not cce,
            }
        )
    if model.is_moe and "kernels" in feats:
        plugins.append(feats["kernels"].plugin_class)
        cfg["use_kernels"] = True
        cfg["use_scattermoe"] = True
    if (
        "fla_sm120_shim" in feats
        and hw.gpus
        and hw.min_cc >= 12.0
        and model.model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next", "kimi_linear")
    ):
        plugins.append(feats["fla_sm120_shim"].plugin_class)
    if plugins:
        cfg["plugins"] = plugins

    # ── multi-gpu ───────────────────────────────────────────────────────────────────────
    ds_dir = catalog.DEEPSPEED_DIR
    if p.sharding == "zero2":
        cfg["deepspeed"] = str(ds_dir / "zero2.json")
    elif p.sharding == "zero3":
        cfg["deepspeed"] = str(ds_dir / ("zero3_bf16.json" if hw.bf16 else "zero3.json"))
    elif p.sharding == "zero3_offload":
        cfg["deepspeed"] = str(ds_dir / "zero3_bf16_cpuoffload_params.json")
    elif p.sharding == "fsdp2":
        arch = (model.architectures or ["Model"])[0]
        layer_cls = arch.replace("ForCausalLM", "DecoderLayer").replace("ForConditionalGeneration", "DecoderLayer")
        cfg["fsdp_version"] = 2
        cfg["fsdp_config"] = {
            "offload_params": False,
            "cpu_ram_efficient_loading": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": layer_cls,
            "state_dict_type": "FULL_STATE_DICT",
            "reshard_after_forward": True,
        }

    # ── saving / logging ────────────────────────────────────────────────────────────────
    cfg["saves_per_epoch"] = 2 if p.epochs == 1 else 1
    cfg["save_total_limit"] = 2
    if cfg["val_set_size"]:
        cfg["evals_per_epoch"] = 4 if p.epochs == 1 else 2
    cfg["logging_steps"] = 1
    cfg["auto_resume_from_checkpoints"] = True
    cfg["dataloader_num_workers"] = max(1, min(hw.cpu_count // max(hw.gpu_count, 1), 8))
    cfg["strict"] = False
    return cfg, p
