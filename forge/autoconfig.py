"""AI-written axolotl configs: model + hardware + 5 random dataset sectors + installed
plugins + a rules-engine baseline go to the LLM; the answer is clamped by guardrails,
validated by axolotl itself, repaired by the LLM if axolotl rejects it, and - when a
training run crashes - fixed again from the crash log."""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from . import catalog, heuristic, providers, ui
from .fetch import DatasetInfo, ModelInfo
from .hardware import Hardware
from .sampler import DatasetProfile

SYSTEM_PROMPT = """You are the config engineer inside axolotl-forge, a Docker image for fine-tuning LLMs with Axolotl {axolotl_version}.
You write one complete Axolotl YAML config that is *balanced*: it fits the exact hardware with roughly 10-15% VRAM headroom,
switches on the installed speed/memory features that suit this model, and picks hyper-parameters that suit the dataset size
and the user's goal.

Hard rules:
1. Reply with exactly one ```yaml fenced block holding the full config. Start the YAML with 3-8 comment lines (# ...) that
   explain the important choices in plain words. Nothing outside the block.
2. Copy every key in LOCKED exactly (base_model, output_dir, dataset_prepared_path and each dataset's path / ds_type /
   data_files / split). Do not add other datasets.
3. Only use plugins and features from INSTALLED, with their exact class paths and keys. Never write wandb_*, use_wandb,
   mlflow_*, comet_*, report_to, hub_model_id, hub_strategy or push_to_hub: forge handles uploading.
4. cut_cross_entropy and liger_fused_linear_cross_entropy are mutually exclusive. CCE only for supported model types.
5. Use attn_implementation with a canonical name (flash_attention_2, flash_attention_3, sdpa, flex_attention, xformers...).
   Not the legacy `flash_attention: true`. sample_packing needs a flash / flex / xformers backend.
6. One GPU: no deepspeed, fsdp, context_parallel, expert_parallel or tensor_parallel. Several GPUs: plain DDP (no
   deepspeed/fsdp keys) when the model fits per GPU, otherwise exactly one of deepspeed (a path from
   deepspeed_config_files) or fsdp_version: 2.
7. Pick the dataset `type` and field mappings from the real records in DATASET.random_sectors, respecting the roles and
   field names that actually appear. DATASET.detected is a heuristic guess you may override.
8. bf16 only when every GPU supports it, otherwise fp16. QLoRA means adapter: qlora + load_in_4bit: true.
9. Start from BASELINE (computed from a VRAM model of this exact machine). Change numbers only for a concrete reason and keep
   the result inside the memory budget. Respect METHOD_PREFERENCE.
10. The YAML must pass axolotl's pydantic validation. When unsure about a key, leave it out.
"""

FIX_PROMPT = """## TRAINING FAILED (attempt {attempt} of {max_attempts})
The CURRENT CONFIG passed axolotl's validator, but the run crashed. What it printed (key error lines first, then the
log from the first traceback):
```
{error}
```
Find the cause and reply with the full corrected YAML. Typical fixes: CUDA out of memory -> lower micro_batch_size, then
sequence_len, keep gradient_checkpointing on, switch lora -> qlora, or (multi-GPU) shard with deepspeed zero3 / fsdp;
NaN or exploding loss -> lower learning_rate, bf16 instead of fp16; kernel / plugin / import errors -> drop that plugin or
switch attn_implementation to sdpa (and turn sample_packing off); tokenizer / prompt-format / dataset errors -> fix the
dataset type and field mappings using the random sectors. If the error cannot be fixed from the config (network, disk
full, gated model, missing token), keep the config and say so in the comment lines.
"""

METHOD_TEXT = {
    "adapter": "adapter only: LoRA or QLoRA (the user uploads the LoRA adapter AND a merged model). Never a full fine-tune.",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "full": "full fine-tune",
    "auto": "auto: pick full / LoRA / QLoRA from the hardware and data",
}

BANNED_PREFIXES = ("wandb_", "mlflow_", "comet_")
BANNED_KEYS = {"use_wandb", "report_to", "hub_model_id", "hub_strategy", "push_to_hub", "push_dataset_to_hub"}
LOCKED_DS_KEYS = ("path", "ds_type", "data_files", "split")
ADAPTER_KEYS = ("adapter", "load_in_4bit", "lora_r", "lora_alpha", "lora_dropout", "lora_target_linear")
# config keys owned by each plugin: dropped together with the plugin
PLUGIN_KEYS = {
    "cut_cross_entropy": ("cut_cross_entropy",),
    "liger": ("liger_",),
    "kernels": ("use_kernels", "use_scattermoe", "use_sonicmoe", "sonicmoe_"),
    "spectrum": ("spectrum_",),
    "mixlora": ("mixlora_",),
    "grokfast": ("grokfast_",),
    "kd": ("kd_",),
    "lm_eval": ("lm_eval_",),
    "diffusion": ("=diffusion",),  # "=" = exact key (diffusion_decision is a different plugin)
    "diffusion_decision": ("diffusion_decision",),
    "llm_compressor": ("llmcompressor",),
    "swanlab": ("swanlab_", "use_swanlab"),
    "hatchery": ("hatchery",),
    "nemo_gym": ("nemo_gym",),
}


def _axolotl_version() -> str:
    try:
        from importlib import metadata

        return metadata.version("axolotl")
    except Exception:
        return "0.21.x"


def run_name_for(settings, model: ModelInfo, data: DatasetInfo) -> str:
    if settings.run_name:
        return re.sub(r"[^A-Za-z0-9_.\-]+", "-", settings.run_name)
    model_name = model.hf_repo.split("/")[-1] if model.hf_repo else Path(model.base_model).name
    data_name = data.hf_repo.split("/")[-1] if data.hf_repo else data.files[0].stem
    m = re.sub(r"[^A-Za-z0-9.\-]+", "-", model_name)
    d = re.sub(r"[^A-Za-z0-9.\-]+", "-", data_name)
    stamp = _dt.datetime.now().strftime("%m%d-%H%M")
    return f"{m[:32]}-{d[:24]}-{stamp}".strip("-").lower()


def extract_yaml(text: str) -> tuple[dict, list[str]]:
    blocks = re.findall(r"```(?:ya?ml)?\s*\n(.*?)```", text, flags=re.S)
    body = max(blocks, key=len) if blocks else text
    data = yaml.safe_load(body)
    if not isinstance(data, dict):
        raise ValueError("the answer did not contain a YAML mapping")
    comments = []
    for line in body.splitlines():
        if line.strip().startswith("#"):
            comments.append(line.strip().lstrip("#").strip())
        elif line.strip():
            break
    return data, comments


# ── guardrails ───────────────────────────────────────────────────────────────────

def _locked(settings, model: ModelInfo, data: DatasetInfo, run_name: str) -> dict:
    return {
        "base_model": model.base_model,
        "output_dir": str(settings.outputs_dir / run_name),
        "dataset_prepared_path": str(settings.runs_dir / run_name / "prepared"),
        "datasets": [data.axolotl_entry()],
    }


def apply_guardrails(
    cfg: dict, settings, hw: Hardware, model: ModelInfo, data: DatasetInfo, run_name: str, baseline: dict | None = None
) -> list[str]:
    changes: list[str] = []
    locked = _locked(settings, model, data, run_name)

    for key in ("base_model", "output_dir", "dataset_prepared_path"):
        if cfg.get(key) != locked[key]:
            if key in cfg:
                changes.append(f"{key} forced to {locked[key]}")
            cfg[key] = locked[key]

    # datasets: keep the first entry's prompt settings, force our paths
    entries = cfg.get("datasets") if isinstance(cfg.get("datasets"), list) else []
    first = dict(entries[0]) if entries and isinstance(entries[0], dict) else {}
    if len(entries) > 1:
        changes.append(f"dropped {len(entries) - 1} extra dataset entr{'y' if len(entries) == 2 else 'ies'}")
    for k in LOCKED_DS_KEYS:
        first.pop(k, None)
    merged = {**locked["datasets"][0], **first}
    if "type" not in merged and not cfg.get("rl"):
        merged["type"] = "completion"
        changes.append("dataset type missing -> completion")
    cfg["datasets"] = [merged]

    for key in list(cfg.keys()):
        if key in BANNED_KEYS or key.startswith(BANNED_PREFIXES):
            cfg.pop(key)
            changes.append(f"removed {key}")

    # adapter-only runs (setup screen) must produce a LoRA
    if settings.method in ("adapter", "lora", "qlora") and cfg.get("adapter") not in ("lora", "qlora") and baseline:
        for k in [k for k in cfg if k.startswith(("lora_", "fsdp")) or k in ADAPTER_KEYS]:
            cfg.pop(k)
        for k in ADAPTER_KEYS:
            if k in baseline:
                cfg[k] = baseline[k]
        changes.append(f"method must be an adapter -> {cfg.get('adapter')} from the baseline")

    # plugins must be installed
    feats = {f.key: f for f in catalog.features()}
    installed = {f.plugin_class for f in feats.values() if f.plugin_class and f.available}
    plugins = [p for p in (cfg.get("plugins") or []) if isinstance(p, str)]
    kept = [p for p in plugins if p in installed]
    for p in plugins:
        if p not in installed:
            changes.append(f"removed unknown/unavailable plugin {p}")
    cce_cls = feats["cut_cross_entropy"].plugin_class
    if cce_cls in kept and model.model_type not in catalog.CCE_MODELS:
        kept.remove(cce_cls)
        changes.append(f"CCE removed: model type {model.model_type} not supported")
    for key, feat in feats.items():  # a plugin that is not loaded must not leave its keys behind
        prefixes = PLUGIN_KEYS.get(key)
        if not prefixes or not feat.plugin_class or feat.plugin_class in kept:
            continue
        stale = [k for k in cfg if any(k == p[1:] if p.startswith("=") else k.startswith(p) for p in prefixes)]
        for k in stale:
            cfg.pop(k)
        if stale:
            changes.append(f"removed {', '.join(stale)} ({key} plugin not loaded)")
    if cfg.get("adapter") == "mixlora" and feats["mixlora"].plugin_class not in kept:
        cfg["adapter"] = "lora"
        changes.append("adapter mixlora -> lora (MixLoRA plugin not loaded)")
    if cce_cls in kept and cfg.get("liger_fused_linear_cross_entropy"):
        cfg["liger_fused_linear_cross_entropy"] = False
        changes.append("liger_fused_linear_cross_entropy off (CCE is on)")
    if kept:
        cfg["plugins"] = list(dict.fromkeys(kept))
    else:
        cfg.pop("plugins", None)

    # attention
    if cfg.pop("flash_attention", None):
        cfg.setdefault("attn_implementation", "flash_attention_2")
        changes.append("flash_attention: true -> attn_implementation: flash_attention_2")
    attn = str(cfg.get("attn_implementation") or "")
    if hw.gpus and hw.min_cc < 8.0 and "flash" in attn:
        cfg["attn_implementation"] = "sdpa"
        changes.append("GPU older than Ampere -> attn_implementation: sdpa")
    if hw.gpus and hw.min_cc < 9.0 and attn == "flash_attention_3":
        cfg["attn_implementation"] = "flash_attention_2"
        changes.append("flash_attention_3 needs Hopper -> flash_attention_2")
    if cfg.get("attn_implementation") in ("sdpa", "eager") and cfg.get("sample_packing"):
        cfg["sample_packing"] = False
        cfg.pop("eval_sample_packing", None)
        changes.append("sample_packing off (needs a flash/flex/xformers backend)")

    # precision
    if hw.gpus and not hw.bf16 and cfg.get("bf16") not in (False, None):
        cfg["bf16"] = False
        cfg["fp16"] = True
        cfg.pop("tf32", None)
        changes.append("GPU has no bf16 -> fp16")

    if cfg.get("adapter") == "qlora" and not cfg.get("load_in_4bit"):
        cfg["load_in_4bit"] = True
        changes.append("qlora -> load_in_4bit: true")

    # multi-gpu
    multi_keys = [k for k in cfg if k == "deepspeed" or k.startswith("fsdp") or k.startswith("context_parallel")
                  or k.startswith("expert_parallel") or k in ("tensor_parallel_size", "dp_shard_size")]
    if hw.gpu_count <= 1 and multi_keys:
        for k in multi_keys:
            cfg.pop(k)
        changes.append("single GPU -> removed " + ", ".join(multi_keys))
    if cfg.get("deepspeed"):
        name = Path(str(cfg["deepspeed"])).name
        target = catalog.DEEPSPEED_DIR / name
        if target.exists() or not catalog.DEEPSPEED_DIR.exists():
            if str(cfg["deepspeed"]) != str(target):
                cfg["deepspeed"] = str(target)
                changes.append(f"deepspeed -> {target}")
        else:
            cfg.pop("deepspeed")
            changes.append(f"removed deepspeed config {name} (not found)")
        if cfg.get("deepspeed") and any(k.startswith("fsdp") for k in cfg):
            for k in [k for k in cfg if k.startswith("fsdp")]:
                cfg.pop(k)
            changes.append("deepspeed and fsdp both set -> kept deepspeed")

    max_pos = model._cfg("max_position_embeddings", "max_seq_len", "seq_length")
    if isinstance(max_pos, int) and isinstance(cfg.get("sequence_len"), int) and cfg["sequence_len"] > max_pos:
        changes.append(f"sequence_len {cfg['sequence_len']} > model max {max_pos} (kept; rope scaling may be intended)")

    cfg.setdefault("auto_resume_from_checkpoints", True)
    if model.config.get("auto_map"):
        cfg.setdefault("trust_remote_code", True)
    return changes


# ── validation ───────────────────────────────────────────────────────────────────

def validate(path: Path) -> tuple[bool | None, str]:
    """Run axolotl's own config loader in a subprocess. None = axolotl not installed here."""
    try:
        import importlib.util

        if importlib.util.find_spec("axolotl") is None:
            return None, "axolotl not installed - validation skipped"
    except (ImportError, ValueError):
        return None, "axolotl not installed - validation skipped"
    code = "import sys\nfrom axolotl.cli.config import load_cfg\nload_cfg(sys.argv[1])\nprint('FORGE_CONFIG_OK')\n"
    env = dict(os.environ)
    env.setdefault("AXOLOTL_DO_NOT_TRACK", "1")
    try:
        proc = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=900, env=env)
    except subprocess.TimeoutExpired:
        return False, "axolotl config validation timed out"
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 and "FORGE_CONFIG_OK" in out:
        return True, ""
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return False, re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(lines[-60:]))


# ── the drafter ──────────────────────────────────────────────────────────────────

class Drafter:
    """Owns one run's config: first draft, user-requested revisions, and fixes after crashes."""

    def __init__(self, settings, hw: Hardware, model: ModelInfo, data: DatasetInfo, prof: DatasetProfile,
                 provider: providers.Provider | None, run_name: str):
        self.settings, self.hw, self.model, self.data, self.prof = settings, hw, model, data, prof
        self.provider = provider
        self.run_name = run_name
        self.baseline, self.plan = heuristic.build_config(settings, hw, model, data, prof, run_name)
        self.locked = _locked(settings, model, data, run_name)
        self.path = settings.configs_dir / f"{run_name}.yml"
        self.system = SYSTEM_PROMPT.format(axolotl_version=_axolotl_version())
        self.feedback = ""
        self.source = "forge rules engine"

    # prompt -------------------------------------------------------------------------
    def base_prompt(self) -> str:
        sections = {
            "USER_GOAL": self.settings.goal or "(none given) - a solid general fine-tune of this model on this dataset",
            "METHOD_PREFERENCE": METHOD_TEXT.get(self.settings.method, self.settings.method),
            "HARDWARE": self.hw.to_prompt(),
            "MODEL": self.model.to_prompt(),
            "DATASET": self.prof.to_prompt(),
            "INSTALLED": catalog.for_prompt(),
            "CCE_SUPPORTED_MODEL_TYPE": self.model.model_type in catalog.CCE_MODELS,
            "LOCKED": self.locked,
            "BASELINE_PLAN": self.plan.to_prompt(),
        }
        parts = [f"## {k}\n{json.dumps(v, indent=2, default=str, ensure_ascii=False)}" for k, v in sections.items()]
        parts.append("## BASELINE_CONFIG\n```yaml\n" + yaml.safe_dump(self.baseline, sort_keys=False, allow_unicode=True) + "```")
        if self.feedback:
            parts.append(f"## EXTRA INSTRUCTIONS FROM THE USER\n{self.feedback}")
        return "\n\n".join(parts)

    def _ask(self, prompt: str, status: str) -> tuple[dict, list[str]] | None:
        if self.provider is None:
            return None
        try:
            with ui.console.status(f"[hot]{self.provider.label()}[/hot] {status}"):
                answer = providers.complete_with_recovery(self.provider, self.system, prompt)
            return extract_yaml(answer)
        except (providers.LLMError, ValueError, yaml.YAMLError) as exc:
            ui.warn(f"{self.provider.label()} failed: {exc}")
            return None

    # files ----------------------------------------------------------------------------
    def _header(self, source: str, rationale: list[str], changes: list[str]) -> list[str]:
        hw, model, data, prof = self.hw, self.model, self.data, self.prof
        lines = [
            f"axolotl-forge config · written by {source} · {_dt.datetime.now():%Y-%m-%d %H:%M}",
            f"hardware: {hw.gpu_label} · {hw.cpu_count} CPU · {hw.ram_gb:.0f} GB RAM",
            f"model:    {model.base_model} ({model.model_type}, {model.params_b or '?'}B)",
            f"dataset:  {data.link} ({prof.fmt}, ~{prof.est_rows or '?'} rows, {prof.detected.get('format')})",
            "",
        ]
        lines += [r for r in rationale if r]
        if changes:
            lines += ["", "forge guardrails: " + "; ".join(changes)]
        return lines

    def _commit(self, cfg: dict, rationale: list[str], source: str) -> list[str]:
        changes = apply_guardrails(cfg, self.settings, self.hw, self.model, self.data, self.run_name, self.baseline)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=120)
        header = self._header(source, rationale, changes)
        self.path.write_text("\n".join(f"# {h}" if h else "#" for h in header) + "\n\n" + body, encoding="utf-8")
        self.source = source
        if changes:
            ui.info("guardrails adjusted: " + "; ".join(changes))
        return changes

    def _validate_and_repair(self, cfg: dict, prompt: str) -> bool | None:
        with ui.console.status("asking axolotl to validate the config…"):
            ok, error = validate(self.path)
        if ok is None:
            ui.warn(error)
            return None
        attempts = 0
        while ok is False and self.provider is not None and attempts < self.settings.max_repairs:
            attempts += 1
            ui.warn(f"axolotl rejected the config - asking {self.provider.label()} to fix it ({attempts}/{self.settings.max_repairs})")
            repair = (
                prompt + "\n\n## YOUR PREVIOUS CONFIG\n```yaml\n" + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
                + "```\n\n## AXOLOTL VALIDATION ERROR\n```\n" + error + "\n```\nFix the problem and reply with the full corrected YAML."
            )
            got = self._ask(repair, "is repairing the config…")
            if not got:
                break
            cfg.clear()
            cfg.update(got[0])
            self._commit(cfg, got[1], f"{self.provider.label()} (repaired)")
            with ui.console.status("validating again…"):
                ok, error = validate(self.path)
        if ok:
            ui.ok("axolotl accepted the config")
        else:
            ui.panel(error[-3000:], "axolotl says")
        return ok

    # public ---------------------------------------------------------------------------
    def draft(self, feedback: str | None = None) -> Path:
        if feedback:
            self.feedback = (self.feedback + "\n" + feedback).strip()
        prompt = self.base_prompt()
        got = self._ask(prompt, "is reading your data and writing the config…")
        if got:
            cfg, rationale = got
            self._commit(cfg, rationale, self.provider.label())  # type: ignore[union-attr]
        else:
            cfg = dict(self.baseline)
            self._commit(cfg, list(self.plan.notes), "forge rules engine")
        if self._validate_and_repair(cfg, prompt) is False:
            ui.warn("falling back to the rules engine baseline")
            self.use_baseline()
        return self.path

    def use_baseline(self) -> Path:
        cfg = dict(self.baseline)
        self._commit(cfg, list(self.plan.notes), "forge rules engine")
        ok, err = validate(self.path)
        if ok is False:
            ui.err("axolotl rejected the baseline config too:")
            ui.panel(err[-3000:], "axolotl says")
        return self.path

    def fix_training_error(self, error: str, attempt: int, max_attempts: int) -> bool:
        """Hand the crash log to the AI; True when a new validated config was written."""
        if self.provider is None:
            return False
        current = self.path.read_text(encoding="utf-8")
        prompt = (
            self.base_prompt()
            + "\n\n## CURRENT CONFIG\n```yaml\n" + current + "\n```\n\n"
            + FIX_PROMPT.format(attempt=attempt, max_attempts=max_attempts, error=error[-12000:])
        )
        got = self._ask(prompt, "is reading the crash log and fixing the config…")
        if not got:
            return False
        cfg, rationale = got
        self._commit(cfg, rationale, f"{self.provider.label()} (fix after failed attempt {attempt})")
        for line in rationale:
            ui.console.print(f"   [soft]# {line}[/soft]")
        return self._validate_and_repair(cfg, prompt) is not False


# ── the classic `forge config` / `forge run` flow ──────────────────────────────

def generate(settings, hw: Hardware, model: ModelInfo, data: DatasetInfo, prof: DatasetProfile,
             provider: providers.Provider | None, run_name: str) -> tuple[Path, Drafter]:
    drafter = Drafter(settings, hw, model, data, prof, provider, run_name)
    path = drafter.draft()
    while True:
        action = review(path)
        if action == "accept":
            return path, drafter
        if action == "quit":
            raise SystemExit(0)
        if action == "baseline":
            drafter.use_baseline()
        elif action.startswith("regen:"):
            drafter.draft(action[6:])


def review(path: Path) -> str:
    ui.show_yaml(path.read_text(encoding="utf-8"), title=str(path))
    if not ui.interactive():
        return "accept"
    choice = ui.choose(
        "Use this config?",
        [
            ("accept", "yes, train with it"),
            ("edit", f"edit it first ({os.environ.get('EDITOR', 'nano')})"),
            ("regen", "regenerate with extra instructions for the AI"),
            ("baseline", "use the rules-engine baseline instead"),
            ("quit", "quit"),
        ],
    )
    if choice == "edit":
        subprocess.call([os.environ.get("EDITOR", "nano"), str(path)])
        ok, error = validate(path)
        if ok is False:
            ui.panel(error[-3000:], "axolotl says")
            return review(path)
        return "accept"
    if choice == "regen":
        return "regen:" + ui.ask("What should change? (e.g. 'use 8k context', '3 epochs', 'full fine-tune')")
    return choice
