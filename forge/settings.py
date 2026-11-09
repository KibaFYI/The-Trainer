"""All knobs, readable from env vars (template-friendly) and overridable from the CLI."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    # tolerate "VALUE   # comment" pasted from env files (docker keeps it literally)
    value = re.sub(r"\s+#.*$", "", value).strip()
    return value if value else default


def _flag(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # what to train
    model: str | None = None              # FORGE_MODEL    HF id / HF url / http(s) / s3:// / gdrive / local path
    dataset: str | None = None            # FORGE_DATASET  same link types; jsonl / json / parquet / csv / arrow
    method: str = "auto"                  # FORGE_METHOD   auto | lora | qlora | full
    goal: str = ""                        # FORGE_GOAL     free text for the AI ("roleplay chat, 2 epochs, 8k ctx")
    config: str | None = None             # FORGE_CONFIG   use this axolotl yaml (path/url) instead of generating one
    run_name: str | None = None           # FORGE_RUN_NAME

    # AI config generator
    provider: str = "auto"                # FORGE_LLM_PROVIDER  auto | anthropic | openai | openrouter | custom | none
    llm_model: str | None = None          # FORGE_LLM_MODEL
    llm_base_url: str | None = None       # FORGE_LLM_BASE_URL  (custom provider, OpenAI- or Anthropic-compatible)
    llm_api_key: str | None = None        # FORGE_LLM_API_KEY   (custom provider)
    llm_api_style: str = "openai"         # FORGE_LLM_API_STYLE openai | anthropic   (custom provider)
    llm_effort: str = "high"              # FORGE_LLM_EFFORT    claude effort: low | medium | high | xhigh | max
    sample_count: int = 5                 # FORGE_SAMPLE_COUNT  random dataset "sectors" shown to the AI
    max_repairs: int = 2                  # FORGE_MAX_REPAIRS   AI retries when axolotl rejects the config

    # upload
    upload: bool = True                   # FORGE_UPLOAD
    hub_repo: str | None = None           # FORGE_HUB_REPO       full "namespace/name"
    hub_namespace: str | None = None      # FORGE_HUB_NAMESPACE  user or org; "ask" to pick from a list
    hub_name: str | None = None           # FORGE_HUB_NAME
    hub_private: bool = True              # FORGE_HUB_PRIVATE
    upload_what: str = "auto"             # FORGE_UPLOAD_WHAT    auto | adapter | merged | both | full
    merge: str = "auto"                   # FORGE_MERGE          auto | 1 | 0   (merge LoRA into base after training)

    # pipeline
    home: Path = field(default_factory=lambda: Path("/workspace/forge"))   # FORGE_HOME
    yes: bool = False                     # FORGE_YES  never ask, use defaults
    skip_model_download: bool = False     # FORGE_SKIP_MODEL_DOWNLOAD  only fetch config + tokenizer up front
    on_finish: str | None = None          # FORGE_ON_FINISH_CMD  e.g. "runpodctl stop pod $RUNPOD_POD_ID"
    hf_token: str | None = None           # HF_TOKEN
    discord_webhook: str | None = None    # FORGE_DISCORD_WEBHOOK  training done / uploaded / finished / failed
    train_attempts: int = 3               # FORGE_TRAIN_ATTEMPTS   crash -> AI fixes the config -> retry

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.model = _env("FORGE_MODEL")
        s.dataset = _env("FORGE_DATASET")
        s.method = (_env("FORGE_METHOD", "auto") or "auto").lower()
        s.goal = _env("FORGE_GOAL", "") or ""
        s.config = _env("FORGE_CONFIG")
        s.run_name = _env("FORGE_RUN_NAME")
        s.provider = (_env("FORGE_LLM_PROVIDER", "auto") or "auto").lower()
        s.llm_model = _env("FORGE_LLM_MODEL")
        s.llm_base_url = _env("FORGE_LLM_BASE_URL")
        s.llm_api_key = _env("FORGE_LLM_API_KEY")
        s.llm_api_style = (_env("FORGE_LLM_API_STYLE", "openai") or "openai").lower()
        s.llm_effort = (_env("FORGE_LLM_EFFORT", "high") or "high").lower()
        s.sample_count = int(_env("FORGE_SAMPLE_COUNT", "5") or 5)
        s.max_repairs = int(_env("FORGE_MAX_REPAIRS", "2") or 2)
        s.upload = _flag("FORGE_UPLOAD", True)
        s.hub_repo = _env("FORGE_HUB_REPO")
        s.hub_namespace = _env("FORGE_HUB_NAMESPACE")
        s.hub_name = _env("FORGE_HUB_NAME")
        s.hub_private = _flag("FORGE_HUB_PRIVATE", True)
        s.upload_what = (_env("FORGE_UPLOAD_WHAT", "auto") or "auto").lower()
        s.merge = (_env("FORGE_MERGE", "auto") or "auto").lower()
        s.home = Path(_env("FORGE_HOME", "/workspace/forge") or "/workspace/forge")
        s.yes = _flag("FORGE_YES", False)
        s.skip_model_download = _flag("FORGE_SKIP_MODEL_DOWNLOAD", False)
        s.on_finish = _env("FORGE_ON_FINISH_CMD")
        s.hf_token = _env("HF_TOKEN") or _env("HUGGING_FACE_HUB_TOKEN")
        s.discord_webhook = _env("FORGE_DISCORD_WEBHOOK")
        s.train_attempts = int(_env("FORGE_TRAIN_ATTEMPTS", "3") or 3)
        return s

    def apply_overrides(self, **overrides) -> "Settings":
        names = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if key in names and value is not None:
                setattr(self, key, Path(value) if key == "home" else value)
        return self

    # folders
    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def datasets_dir(self) -> Path:
        return self.home / "datasets"

    @property
    def configs_dir(self) -> Path:
        return self.home / "configs"

    @property
    def outputs_dir(self) -> Path:
        return self.home / "outputs"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.datasets_dir, self.configs_dir, self.outputs_dir, self.logs_dir, self.runs_dir):
            d.mkdir(parents=True, exist_ok=True)
