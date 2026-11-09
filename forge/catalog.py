"""Everything installed in the image that a config can switch on: plugins, multi-GPU
strategies, attention backends, optimizers. Fed to the AI so it only uses real things."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

BUILD_STATUS = Path("/opt/forge/plugins.json")
DEEPSPEED_DIR = Path(os.environ.get("FORGE_DEEPSPEED_DIR", "/opt/forge/deepspeed_configs"))

# model types cut-cross-entropy patches (axolotl 0.21.1 README)
CCE_MODELS = set(
    """afmoe apertus arcee beam cohere cohere2 cohere2_moe cohere2_vision cohere_compass cohere_compass_text
    deepseek_v2 deepseek_v3 deepseek_v4 exaone4 exaone4_5 exaone_moe falcon_h1 gemma gemma2 gemma3 gemma3_text
    gemma3n gemma3n_text gemma4 gemma4_text gemma4_unified gemma4_unified_text glm glm4 glm4_moe glm4_moe_lite
    glm46v glm4v glm4v_moe glm5_next glm_image glm_moe_dsa gpt_oss granite granitemoe granitemoehybrid
    granitemoeshared hunyuan_v1_dense hunyuan_v1_moe internvl kimi_linear lfm2 lfm2_moe lfm2_vl llama llama4
    llama4_text llava minimax minimax_m2 ministral ministral3 mistral mistral3 mistral4 mixtral mllama
    mllama_text_model muse_glimmer nemotron nemotron_h olmo olmo2 olmo3 olmoe paligemma phi phi3
    phi4_multimodal qwen2 qwen2_5_vl qwen2_moe qwen2_vl qwen3 qwen3_5 qwen3_5_text qwen3_5_moe
    qwen3_5_moe_text qwen3_moe qwen3_next qwen3_vl qwen3_vl_moe qwen4_exp qwen4_exp_text seed_oss smollm3
    smolvlm step3p5 step3p7 voxtral""".split()
)
DENSEMIXER_MODELS = {"olmoe", "qwen2_moe", "qwen3_moe"}


@dataclass
class Feature:
    key: str
    kind: str  # plugin | builtin | multi-gpu | tracker
    title: str
    when: str
    yaml: str
    plugin_class: str | None = None
    requires: list[str] = field(default_factory=list)
    notes: str = ""
    default_on: bool = False
    available: bool = True
    status: str = "ok"


FEATURES: list[Feature] = [
    Feature(
        "cut_cross_entropy", "plugin", "Cut Cross Entropy (Apple CCE)",
        "Almost always for supported model types: computes the loss without materialising logits, big VRAM saving "
        "especially with large vocabularies (Qwen, Gemma, Llama 3+). Do not combine with liger_fused_linear_cross_entropy.",
        "plugins:\n  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin\ncut_cross_entropy: true",
        "axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin", ["cut_cross_entropy.transformers"],
        notes="supported model_type list is checked by forge", default_on=True,
    ),
    Feature(
        "liger", "plugin", "Liger Kernel (Triton fused kernels)",
        "Almost always: fused RoPE / RMSNorm / SwiGLU / LayerNorm cut memory and speed things up, works with FSDP and "
        "DeepSpeed. When CCE is on keep liger_fused_linear_cross_entropy: false.",
        "plugins:\n  - axolotl.integrations.liger.LigerPlugin\nliger_rope: true\nliger_rms_norm: true\n"
        "liger_glu_activation: true\nliger_layer_norm: true\nliger_fused_linear_cross_entropy: false",
        "axolotl.integrations.liger.LigerPlugin", ["liger_kernel"], default_on=True,
    ),
    Feature(
        "deepspeed", "multi-gpu", "DeepSpeed ZeRO 1/2/3 (+CPU offload)",
        "Multi-GPU. zero1/zero2 for LoRA or models that fit per GPU; zero3 (optionally *_cpuoffload_*) to shard full "
        "fine-tunes that do not fit on one GPU. Mutually exclusive with FSDP.",
        f"deepspeed: {DEEPSPEED_DIR}/zero2.json   # files: " + ", ".join(
            ["zero1.json", "zero1_torch_compile.json", "zero2.json", "zero2_torch_compile.json", "zero3.json",
             "zero3_bf16.json", "zero3_bf16_cpuoffload_params.json", "zero3_bf16_cpuoffload_all.json"]
        ),
        requires=["deepspeed"],
    ),
    Feature(
        "fsdp2", "multi-gpu", "PyTorch FSDP2",
        "Multi-GPU sharding (recommended by axolotl), required for FSDP+QLoRA. Only fsdp_version: 2 exists. Mutually "
        "exclusive with deepspeed. transformer_layer_cls_to_wrap = the model's decoder layer class (e.g. Qwen3DecoderLayer).",
        "fsdp_version: 2\nfsdp_config:\n  offload_params: false\n  cpu_ram_efficient_loading: true\n"
        "  auto_wrap_policy: TRANSFORMER_BASED_WRAP\n  transformer_layer_cls_to_wrap: <DecoderLayerClass>\n"
        "  state_dict_type: FULL_STATE_DICT\n  reshard_after_forward: true",
    ),
    Feature(
        "context_parallel", "multi-gpu", "Context parallel (ringmaster: Ulysses / Ring / USP)",
        "Multi-GPU + very long sequences (>= 32k) that do not fit on one GPU even with checkpointing. Built in: no "
        "plugins entry. Needs a flash attention backend for Ring; works with FSDP2 and sample packing.",
        "context_parallel_size: 2   # or the block form:\n# context_parallel:\n#   size: 4\n#   backend: auto",
        requires=["ringmaster"],
    ),
    Feature(
        "expert_parallel", "multi-gpu", "Expert parallel (MoE token dispatch)",
        "Multi-GPU + MoE models only: shards experts across GPUs. backend torch works on any fabric, deep_ep needs NVLink "
        "and the optional DeepEP build. Built in: no plugins entry.",
        "expert_parallel_size: 2\nexpert_parallel_backend: torch   # or auto / deep_ep",
    ),
    Feature(
        "kernels", "plugin", "MoE kernels (ScatterMoE / SonicMoE)",
        "MoE models (qwen3_moe, gpt_oss, deepseek, glm4_moe...): faster, lower-VRAM expert GEMMs incl. LoRA on experts. "
        "use_scattermoe on any GPU; use_sonicmoe only on Hopper/Blackwell. Pick one.",
        "plugins:\n  - axolotl.integrations.kernels.KernelsPlugin\nuse_kernels: true\nuse_scattermoe: true",
        "axolotl.integrations.kernels.KernelsPlugin", ["kernels"],
    ),
    Feature(
        "densemixer", "plugin", "DenseMixer (MoE router gradients)",
        "Only MoE model types olmoe / qwen2_moe / qwen3_moe: better router training via dense gradient estimate.",
        "plugins:\n  - axolotl.integrations.densemixer.DenseMixerPlugin",
        "axolotl.integrations.densemixer.DenseMixerPlugin", ["densemixer"],
    ),
    Feature(
        "spectrum", "plugin", "Spectrum (train only high-SNR layers)",
        "Full fine-tunes where you want to freeze the low signal-to-noise half of the network to save memory/time. "
        "Needs a pre-scanned SNR file for the model (spectrum_model_name).",
        "plugins:\n  - axolotl.integrations.spectrum.SpectrumPlugin\nspectrum_top_fraction: 0.5\n"
        "spectrum_model_name: <hf model the scan exists for>",
        "axolotl.integrations.spectrum.SpectrumPlugin",
    ),
    Feature(
        "mixlora", "plugin", "MixLoRA (MoE of LoRA experts)",
        "Alternative adapter for dense SwiGLU models (gate/up/down_proj) when you want more adapter capacity than LoRA. "
        "lora_target_modules must not include gate/up/down_proj and lora_target_linear is not allowed; no merge-lora.",
        "plugins:\n  - axolotl.integrations.mixlora.MixLoraPlugin\nadapter: mixlora\nlora_target_modules: [q_proj, k_proj, v_proj, o_proj]\n"
        "mixlora_num_experts: 8\nmixlora_top_k: 2",
        "axolotl.integrations.mixlora.MixLoraPlugin",
    ),
    Feature(
        "grokfast", "plugin", "Grokfast (amplify slow gradients)",
        "Small datasets / algorithmic tasks where you want faster generalisation. Rarely needed for chat SFT.",
        "plugins:\n  - axolotl.integrations.grokfast.GrokfastPlugin\ngrokfast_alpha: 2.0\ngrokfast_lamb: 0.98",
        "axolotl.integrations.grokfast.GrokfastPlugin",
    ),
    Feature(
        "kd", "plugin", "Knowledge distillation",
        "Only when the dataset carries teacher logprobs (a logprobs field per sample).",
        "plugins:\n  - axolotl.integrations.kd.KDPlugin\nkd_trainer: true\nkd_ce_alpha: 0.1\nkd_alpha: 0.9\nkd_temperature: 1.0\n"
        "# dataset type: axolotl.integrations.kd.chat_template with logprobs_field",
        "axolotl.integrations.kd.KDPlugin", ["orjson"],
    ),
    Feature(
        "lm_eval", "plugin", "LM Eval Harness after training",
        "When the user asks for benchmark numbers after training. Adds time.",
        "plugins:\n  - axolotl.integrations.lm_eval.LMEvalPlugin\nlm_eval_tasks: [hellaswag, arc_easy]\nlm_eval_batch_size: 8",
        "axolotl.integrations.lm_eval.LMEvalPlugin", ["lm_eval"],
    ),
    Feature(
        "diffusion", "plugin", "Diffusion LM training (LLaDA-style)",
        "Only when the user explicitly wants diffusion language-model training.",
        "plugins:\n  - axolotl.integrations.diffusion.DiffusionPlugin\ndiffusion:\n  noise_schedule: linear",
        "axolotl.integrations.diffusion.DiffusionPlugin",
    ),
    Feature(
        "diffusion_decision", "plugin", "Typed decision training for native diffusion LMs",
        "Only for native diffusion models (e.g. Nemotron diffusion) when explicitly requested.",
        "plugins:\n  - axolotl.integrations.diffusion_decision.DiffusionDecisionPlugin\ndiffusion_decision: {}",
        "axolotl.integrations.diffusion_decision.DiffusionDecisionPlugin",
    ),
    Feature(
        "fla_sm120_shim", "plugin", "FLA sm_120 shim",
        "RTX 50xx / sm_120 GPUs + linear-attention models (Qwen3.5, Qwen3-Next) without context parallel: avoids a "
        "TileLang misaligned-address crash. Numerics unchanged.",
        "plugins:\n  - axolotl.integrations.fla_sm120_shim.FlaSm120ShimPlugin",
        "axolotl.integrations.fla_sm120_shim.FlaSm120ShimPlugin", ["fla"],
    ),
    Feature(
        "llm_compressor", "plugin", "LLMCompressor sparse fine-tuning",
        "Only for models already sparsified with llm-compressor (e.g. neuralmagic *-pruned*/-sparse checkpoints).",
        "plugins:\n  - axolotl.integrations.llm_compressor.LLMCompressorPlugin\nllmcompressor:\n  recipe: {...}\n  save_compressed: true",
        "axolotl.integrations.llm_compressor.LLMCompressorPlugin", ["llmcompressor"],
    ),
    Feature(
        "swanlab", "tracker", "SwanLab experiment tracking",
        "Only if the user asks for SwanLab. (wandb is disabled in this image.)",
        "plugins:\n  - axolotl.integrations.swanlab.SwanLabPlugin\nuse_swanlab: true\nswanlab_project: <name>",
        "axolotl.integrations.swanlab.SwanLabPlugin", ["swanlab"],
    ),
    Feature(
        "tensorboard", "tracker", "TensorBoard (local)",
        "Cheap local loss curves; served on :6006 when FORGE_TENSORBOARD=1.",
        "use_tensorboard: true", requires=["tensorboard"],
    ),
    Feature(
        "nemo_gym", "plugin", "NeMo Gym RL environments",
        "Not for SFT. GRPO with NeMo Gym servers; needs a separate Gym checkout + vLLM server. Never auto-enable.",
        "plugins:\n  - axolotl.integrations.nemo_gym.NemoGymPlugin",
        "axolotl.integrations.nemo_gym.NemoGymPlugin",
    ),
    Feature(
        "hatchery", "plugin", "Hatchery / Tinker remote training",
        "Sends training to a remote Tinker/Hatchery API instead of local GPUs. Never auto-enable.",
        "plugins:\n  - axolotl.integrations.hatchery.HatcheryPlugin\nhatchery:\n  backend: tinker",
        "axolotl.integrations.hatchery.HatcheryPlugin", ["tinker"],
    ),
]

NOT_INSTALLED = {
    "mora": "needs a forked PEFT with MoRA support, which conflicts with axolotl's pinned peft==0.21.2",
    "cloud launchers (modal / baseten / nebius / coreweave)": "launch jobs on other clouds; pointless inside the GPU box",
    "vllm / mamba-ssm / auto-gptq / fbgemm-gpu extras": "axolotl declares them as conflicting with its own pins",
}

ATTENTION = {
    "flash_attention_2": "default on Ampere+ (sm80+); supports sample packing; fetched from HF hub kernels",
    "flash_attention_3": "Hopper only",
    "flash_attention_4": "Hopper/Blackwell, only if flash-attn-4 was built in",
    "flex_attention": "torch flex attention",
    "sdpa": "fallback for older GPUs (Volta/Turing); no sample packing",
    "xformers": "memory-efficient attention, any CUDA GPU",
}

OPTIMIZERS = [
    "adamw_torch_fused", "adamw_torch", "adamw_8bit", "paged_adamw_8bit", "paged_adamw_32bit", "adafactor",
    "lion_8bit", "schedule_free_adamw", "optimi_adamw", "ao_adamw_8bit", "ao_adamw_4bit", "ao_adamw_fp8",
    "adopt_adamw", "came_pytorch", "muon", "muon_8bit", "dion", "galore_adamw", "apollo_adamw", "scao",
    "q_galore_adamw8bit", "lomo", "flash_adamw",
]


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _load_build_status() -> dict:
    if BUILD_STATUS.exists():
        try:
            return json.loads(BUILD_STATUS.read_text())
        except ValueError:
            return {}
    return {}


def _try_import(plugin_class: str) -> tuple[bool, str]:
    module, cls = plugin_class.rsplit(".", 1)
    try:
        getattr(importlib.import_module(module), cls)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"import failed: {type(exc).__name__}: {str(exc)[:160]}"


def features() -> list[Feature]:
    status = _load_build_status()
    out = []
    for f in FEATURES:
        f = Feature(**asdict(f))
        if f.key in status:
            f.available = status[f.key]["available"]
            f.status = status[f.key]["status"]
            # the image is built without a GPU driver: retry imports that failed at build time
            if not f.available and f.plugin_class and f.status.startswith("import failed"):
                f.available, f.status = _try_import(f.plugin_class)
        else:
            missing = [r for r in f.requires if not _module_available(r)]
            if f.plugin_class and not _module_available(f.plugin_class.rsplit(".", 1)[0]):
                missing.append(f.plugin_class.rsplit(".", 1)[0])
            f.available = not missing
            f.status = "ok" if not missing else "missing: " + ", ".join(missing)
        if f.key == "deepspeed" and not DEEPSPEED_DIR.exists():
            f.available = False
            f.status = f"configs missing at {DEEPSPEED_DIR}"
        out.append(f)
    return out


def deepspeed_configs() -> list[str]:
    if not DEEPSPEED_DIR.exists():
        return []
    return sorted(str(p) for p in DEEPSPEED_DIR.glob("*.json"))


def for_prompt() -> dict:
    feats = [f for f in features() if f.available]
    return {
        "plugins_and_features": [
            {"key": f.key, "kind": f.kind, "title": f.title, "when_to_use": f.when, "yaml": f.yaml, "notes": f.notes}
            for f in feats
        ],
        "deepspeed_config_files": deepspeed_configs(),
        "attention_backends": ATTENTION,
        "optimizers": OPTIMIZERS,
        "not_installed": NOT_INSTALLED,
    }


def build_check(write: Path | None = None) -> int:
    """Import every plugin class once at image build time and record what works."""
    import torch  # noqa: F401  (hard requirement)
    import axolotl  # noqa: F401

    results: dict[str, dict] = {}
    for f in FEATURES:
        missing = [r for r in f.requires if not _module_available(r)]
        status, available = "ok", not missing
        if missing:
            status = "missing: " + ", ".join(missing)
        elif f.plugin_class:
            available, status = _try_import(f.plugin_class)
        results[f.key] = {"available": available, "status": status}
        print(f"  {'✔' if available else '✖'} {f.key:<20} {status}")
    if write:
        write.parent.mkdir(parents=True, exist_ok=True)
        write.write_text(json.dumps(results, indent=2))
    return 0
