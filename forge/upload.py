"""Push finished runs to the Hugging Face Hub. The namespace (you or one of your orgs) and the
repo name are offered as pick-lists fetched from your account."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import ui

IGNORE = [
    "checkpoint-*", "checkpoint-*/**", "runs/**", "global_step*/**", "merged/**", ".adapter_shards/**",
    "*.pt", "*.pth", "debug.log", "*.lock", ".cache/**",
]
LARGE_FOLDER_GB = 20


@dataclass
class Identity:
    user: str
    orgs: list[str]
    token_role: str | None


def identity(token: str | None) -> Identity:
    from huggingface_hub import HfApi

    me = HfApi(token=token).whoami()
    orgs = []
    for org in me.get("orgs") or []:
        role = org.get("roleInOrg")
        if role in (None, "admin", "write", "contributor"):
            orgs.append(org.get("name"))
    role = ((me.get("auth") or {}).get("accessToken") or {}).get("role")
    return Identity(me["name"], [o for o in orgs if o], role)


def namespaces(token: str | None) -> list[tuple[str, str]]:
    ident = identity(token)
    return [(ident.user, f"{ident.user}  [dim](you)[/dim]")] + [(o, f"{o}  [dim](org)[/dim]") for o in ident.orgs]


def pick_namespace(settings, ident: Identity) -> str:
    if settings.hub_repo and "/" in settings.hub_repo:
        return settings.hub_repo.split("/", 1)[0]
    ns = settings.hub_namespace
    known = [ident.user, *ident.orgs]
    if ns and ns.lower() not in ("ask", "auto", "list"):
        if ns not in known:
            ui.warn(f"{ns} is not in your account's list ({', '.join(known)}); trying anyway")
        return ns
    options = [(ident.user, f"{ident.user}  [dim](you)[/dim]")] + [(o, f"{o}  [dim](org)[/dim]") for o in ident.orgs]
    if ui.interactive() and len(options) > 0:
        return ui.choose("Upload to which account / org?", options, allow_custom=True, custom_label="type another namespace")
    return ident.user


def pick_repo_name(settings, namespace: str, default_name: str) -> str:
    if settings.hub_repo and "/" in settings.hub_repo:
        return settings.hub_repo.split("/", 1)[1]
    if settings.hub_name:
        return settings.hub_name
    if not ui.interactive():
        return default_name
    existing: list[str] = []
    try:
        from huggingface_hub import HfApi

        existing = [m.id.split("/", 1)[1] for m in HfApi(token=settings.hf_token).list_models(author=namespace, limit=40)]
    except Exception:
        existing = []
    options = [("__new__", f"new repo: [hot]{default_name}[/hot]")] + [(n, f"push into existing: {n}") for n in existing[:25]]
    choice = ui.choose(f"Repo in {namespace}?", options, allow_custom=True, custom_label="type a new name")
    return default_name if choice == "__new__" else choice


def default_repo_name(base_model: str, dataset_link: str, method: str) -> str:
    base = base_model.rstrip("/").split("/")[-1]
    data = re.sub(r"\.(jsonl|json|parquet|csv|arrow)(\.gz|\.zst)?$", "", dataset_link.rstrip("/").split("/")[-1])
    name = f"{base}-{data}-{method}" if method in ("lora", "qlora") else f"{base}-{data}"
    return re.sub(r"[^A-Za-z0-9_.\-]+", "-", name).strip("-.")[:90]


# ── model card ───────────────────────────────────────────────────────────────────

def train_summary(output_dir: Path) -> dict:
    candidates = [output_dir / "trainer_state.json"] + sorted(
        output_dir.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(re.sub(r"\D", "", p.parent.name) or 0)
    )[::-1]
    for state_path in candidates:
        if state_path.exists():
            state = json.loads(state_path.read_text())
            hist = state.get("log_history", [])
            losses = [h["loss"] for h in hist if "loss" in h]
            evals = [h["eval_loss"] for h in hist if "eval_loss" in h]
            end = next((h for h in reversed(hist) if "train_runtime" in h), {})
            return {
                "steps": state.get("global_step"),
                "epochs": round(float(state.get("epoch") or 0), 2),
                "first_loss": round(losses[0], 4) if losses else None,
                "final_loss": round(losses[-1], 4) if losses else None,
                "final_eval_loss": round(evals[-1], 4) if evals else None,
                "runtime_h": round(end.get("train_runtime", 0) / 3600, 2) if end else None,
            }
    return {}


def model_card(cfg: dict, kind: str, dataset_link: str, dataset_repo: str | None, summary: dict, hw_label: str, repo_id: str, adapter_repo: str | None = None) -> str:
    base = cfg.get("base_model", "")
    clean_cfg = {k: v for k, v in cfg.items() if k not in ("output_dir", "dataset_prepared_path")}
    front = {
        "base_model": base,
        "library_name": "peft" if kind == "adapter" else "transformers",
        "tags": ["axolotl", "axolotl-forge", "generated_from_trainer"] + (["lora"] if kind == "adapter" else []),
    }
    if dataset_repo:
        front["datasets"] = [dataset_repo]
    rows = "\n".join(f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items() if v is not None)
    if kind == "adapter":
        usage = (
            "```python\nfrom peft import AutoPeftModelForCausalLM\nfrom transformers import AutoTokenizer\n\n"
            f'model = AutoPeftModelForCausalLM.from_pretrained("{repo_id}", torch_dtype="auto", device_map="auto")\n'
            f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n```'
        )
    else:
        usage = (
            "```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
            f'model = AutoModelForCausalLM.from_pretrained("{repo_id}", torch_dtype="auto", device_map="auto")\n'
            f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n```'
        )
    what = {
        "adapter": f"a LoRA adapter for [{base}](https://huggingface.co/{base})",
        "merged": f"[{base}](https://huggingface.co/{base}) with the fine-tuned LoRA merged in",
        "full": f"a full fine-tune of [{base}](https://huggingface.co/{base})",
    }[kind]
    extra = f"\nThe LoRA adapter alone lives in [{adapter_repo}](https://huggingface.co/{adapter_repo}).\n" if adapter_repo else ""
    return (
        "---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n\n"
        f"# {repo_id.split('/')[-1]}\n\n"
        f"This is {what}, trained with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) using axolotl-forge.\n"
        f"{extra}\n"
        f"**Dataset:** {dataset_link}  \n**Hardware:** {hw_label}\n\n"
        + (f"## Training results\n\n| | |\n|---|---|\n{rows}\n\n" if rows else "")
        + f"## Usage\n\n{usage}\n\n"
        "## Training config\n\n<details><summary>axolotl config</summary>\n\n```yaml\n"
        + yaml.safe_dump(clean_cfg, sort_keys=False, allow_unicode=True)
        + "```\n\n</details>\n"
    )


# ── upload ───────────────────────────────────────────────────────────────────────

def _folder_gb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1024**3


def push_folder(folder: Path, repo_id: str, private: bool, token: str | None, card: str, message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    (folder / "README.md").write_text(card, encoding="utf-8")
    size = _folder_gb(folder)
    ui.info(f"uploading {folder} ({size:.1f} GB) -> [hot]{repo_id}[/hot]")
    if size > LARGE_FOLDER_GB:
        api.upload_large_folder(repo_id=repo_id, folder_path=str(folder), repo_type="model", ignore_patterns=IGNORE)
    else:
        api.upload_folder(repo_id=repo_id, folder_path=str(folder), commit_message=message, ignore_patterns=IGNORE)
    url = f"https://huggingface.co/{repo_id}"
    ui.ok(f"uploaded: {url}")
    return url


def upload_run(settings, cfg_path: Path, dataset_link: str, dataset_repo: str | None, hw_label: str) -> list[str]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    output_dir = Path(cfg["output_dir"])
    if not output_dir.exists():
        raise RuntimeError(f"{output_dir} does not exist - did training finish?")
    token = settings.hf_token
    if not token:
        if ui.interactive():
            token = ui.ask("Hugging Face token with write access (HF_TOKEN)", password=True)
            settings.hf_token = token
        else:
            raise RuntimeError("HF_TOKEN is not set - cannot upload")
    ident = identity(token)
    if ident.token_role and ident.token_role not in ("write", "fineGrained", "admin"):
        ui.warn(f"your HF token role is '{ident.token_role}' - uploads need a write token")
    namespace = pick_namespace(settings, ident)
    adapter = cfg.get("adapter") in ("lora", "qlora")
    method = cfg.get("adapter") or "full"
    name = pick_repo_name(settings, namespace, default_repo_name(cfg.get("base_model", "model"), dataset_link, method))
    repo_id = f"{namespace}/{name}"
    merged_dir = output_dir / "merged"
    what = settings.upload_what
    if what == "auto":
        what = ("both" if merged_dir.exists() else "adapter") if adapter else "full"
    if what in ("merged", "both") and not merged_dir.exists():
        ui.warn("no merged model found - uploading the adapter only")
        what = "adapter"
    targets = {
        "full": (None, repo_id),
        "adapter": (repo_id, None),
        "merged": (None, repo_id),
        "both": (f"{repo_id}-lora", repo_id),  # merged model in the main repo, adapter in <repo>-lora
    }[what]
    return upload_targets(settings, cfg_path, targets[0], targets[1], dataset_link, dataset_repo, hw_label)


def upload_targets(
    settings, cfg_path: Path, lora_repo: str | None, merged_repo: str | None,
    dataset_link: str, dataset_repo: str | None, hw_label: str,
) -> list[str]:
    """Push the adapter and/or the merged (or full) model to explicit repos."""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    output_dir = Path(cfg["output_dir"])
    merged_dir = output_dir / "merged"
    adapter = cfg.get("adapter") in ("lora", "qlora", "mixlora")
    summary = train_summary(output_dir)
    token, private, msg = settings.hf_token, settings.hub_private, "upload from axolotl-forge"
    urls: list[str] = []
    if merged_repo:
        if not adapter:
            card = model_card(cfg, "full", dataset_link, dataset_repo, summary, hw_label, merged_repo)
            urls.append(push_folder(output_dir, merged_repo, private, token, card, msg))
        elif merged_dir.exists():
            card = model_card(cfg, "merged", dataset_link, dataset_repo, summary, hw_label, merged_repo,
                              lora_repo if lora_repo else None)
            urls.append(push_folder(merged_dir, merged_repo, private, token, card, msg))
        else:
            ui.warn("no merged model found - skipping the merged upload")
    if lora_repo and adapter:
        card = model_card(cfg, "adapter", dataset_link, dataset_repo, summary, hw_label, lora_repo)
        urls.append(push_folder(output_dir, lora_repo, private, token, card, msg))
    return urls
