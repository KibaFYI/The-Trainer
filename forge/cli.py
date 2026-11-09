"""forge — the axolotl-forge command line."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from . import __version__, ui
from .settings import Settings

EPILOG = """
examples
  forge                                         the setup screen (same as: forge setup)
  forge run --model Qwen/Qwen3-8B --dataset https://huggingface.co/datasets/mlabonne/FineTome-100k --yes
  forge config --model meta-llama/Llama-3.1-8B --dataset ./chats.jsonl --goal "roleplay, 8k context"
  forge train /workspace/forge/configs/my-run.yml
  forge upload /workspace/forge/configs/my-run.yml --namespace ask
  forge sample ./data.parquet -n 5
  forge orgs | forge models --provider openrouter | forge plugins | forge doctor
"""


def _common_run_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("what to train")
    g.add_argument("--model", "-m", help="model link: HF id, HF url, https://, s3://, gdrive, local path  [FORGE_MODEL]")
    g.add_argument("--dataset", "-d", help="dataset link (jsonl / json / parquet / csv / arrow, archives ok)  [FORGE_DATASET]")
    g.add_argument("--method", choices=["auto", "lora", "qlora", "full"], help="[FORGE_METHOD]")
    g.add_argument("--goal", "-g", help="free-text instructions for the AI  [FORGE_GOAL]")
    g.add_argument("--config", "-c", help="skip generation and use this axolotl yaml (path or url)  [FORGE_CONFIG]")
    g.add_argument("--run-name", help="[FORGE_RUN_NAME]")
    a = p.add_argument_group("AI config generator")
    a.add_argument("--provider", choices=["auto", "anthropic", "openai", "openrouter", "custom", "none"], help="[FORGE_LLM_PROVIDER]")
    a.add_argument("--llm-model", help="[FORGE_LLM_MODEL]")
    a.add_argument("--base-url", dest="llm_base_url", help="custom provider endpoint  [FORGE_LLM_BASE_URL]")
    a.add_argument("--api-style", dest="llm_api_style", choices=["openai", "anthropic"], help="custom provider style  [FORGE_LLM_API_STYLE]")
    a.add_argument("--effort", dest="llm_effort", choices=["low", "medium", "high", "xhigh", "max"], help="Claude effort  [FORGE_LLM_EFFORT]")
    a.add_argument("--samples", "-n", dest="sample_count", type=int, help="random dataset sectors shown to the AI  [FORGE_SAMPLE_COUNT]")
    u = p.add_argument_group("upload")
    u.add_argument("--repo", dest="hub_repo", help="namespace/name  [FORGE_HUB_REPO]")
    u.add_argument("--namespace", dest="hub_namespace", help="user or org, or 'ask' to pick from your list  [FORGE_HUB_NAMESPACE]")
    u.add_argument("--name", dest="hub_name", help="[FORGE_HUB_NAME]")
    u.add_argument("--public", action="store_true", help="make the repo public  [FORGE_HUB_PRIVATE=0]")
    u.add_argument("--no-upload", action="store_true", help="[FORGE_UPLOAD=0]")
    u.add_argument("--upload-what", choices=["auto", "adapter", "merged", "both", "full"], help="[FORGE_UPLOAD_WHAT]")
    u.add_argument("--no-merge", action="store_true", help="do not merge LoRA into the base  [FORGE_MERGE=0]")
    p.add_argument("--yes", "-y", action="store_true", help="never ask; use defaults  [FORGE_YES=1]")
    p.add_argument("--home", help="working folder  [FORGE_HOME]")


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    over = {k: getattr(args, k, None) for k in (
        "model", "dataset", "method", "goal", "config", "run_name", "provider", "llm_model", "llm_base_url",
        "llm_api_style", "llm_effort", "sample_count", "hub_repo", "hub_namespace", "hub_name", "upload_what", "home",
    )}
    s.apply_overrides(**over)
    if getattr(args, "public", False):
        s.hub_private = False
    if getattr(args, "no_upload", False):
        s.upload = False
    if getattr(args, "no_merge", False):
        s.merge = "0"
    if getattr(args, "yes", False):
        s.yes = True
    ui.set_yes(s.yes)
    return s


# ── commands ─────────────────────────────────────────────────────────────────────

def cmd_setup(args) -> int:
    import shutil

    wrap = (
        not args.here
        and not os.environ.get("TMUX")
        and os.environ.get("FORGE_SETUP_TMUX", "1") != "0"
        and shutil.which("tmux")
        and sys.stdin.isatty()
    )
    if wrap:  # run inside tmux so an ssh disconnect never kills training
        os.execvp("tmux", ["tmux", "new-session", "-A", "-s", "forge", "forge setup --here"])
    from . import setup_screen

    return setup_screen.run_setup(Settings.from_env())


def cmd_run(args) -> int:
    from . import pipeline

    return pipeline.run(_settings(args))


def cmd_config(args) -> int:
    from . import pipeline

    return pipeline.config_only(_settings(args))


def cmd_train(args) -> int:
    from . import pipeline

    s = _settings(args)
    return pipeline.train_existing(s, Path(args.config_file))


def cmd_upload(args) -> int:
    import yaml

    from . import hardware, upload

    s = _settings(args)
    target = Path(args.target)
    cfg_path = target if target.is_file() else None
    if cfg_path is None:
        cands = sorted(s.configs_dir.glob("*.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
        cfg_path = next((c for c in cands if Path(yaml.safe_load(c.read_text()).get("output_dir", "")) == target.resolve()), None)
        if cfg_path is None:
            ui.err("give the run's config yaml (forge keeps them in $FORGE_HOME/configs)")
            return 2
    cfg = yaml.safe_load(cfg_path.read_text())
    link = s.dataset or str((cfg.get("datasets") or [{}])[0].get("path", "dataset"))
    hw = hardware.detect(s.home)
    if s.hub_namespace is None and s.hub_repo is None and ui.interactive():
        s.hub_namespace = "ask"
    urls = upload.upload_run(s, cfg_path, link, None, hw.gpu_label)
    for url in urls:
        ui.ok(url)
    return 0


def cmd_sample(args) -> int:
    from . import fetch, pipeline, sampler

    s = _settings(args)
    s.ensure_dirs()
    info = fetch.fetch_dataset(args.dataset_link, s)
    model_dir = None
    if args.model:
        model_dir = fetch.fetch_model(args.model, s, light=True).local_dir
    prof = sampler.profile(info, args.n, model_dir=model_dir, seed=args.seed)
    pipeline.show_profile(prof)
    if args.json:
        import json

        print(json.dumps(prof.to_prompt(), indent=2, default=str, ensure_ascii=False))
    return 0


def cmd_fetch(args) -> int:
    from . import fetch

    s = _settings(args)
    s.ensure_dirs()
    if args.kind == "model":
        info = fetch.fetch_model(args.link, s, light=args.light)
        ui.kv_table("model", list(info.to_prompt().items()))
    else:
        info = fetch.fetch_dataset(args.link, s)
        ui.kv_table("dataset", [("files", len(info.files)), ("format", info.fmt), ("root", info.root), ("axolotl", info.axolotl_entry())])
    return 0


def cmd_orgs(args) -> int:
    from . import upload

    s = Settings.from_env()
    if not s.hf_token:
        ui.err("set HF_TOKEN first")
        return 2
    ident = upload.identity(s.hf_token)
    ui.table("where you can upload", ["namespace", "kind"], [(ident.user, "you")] + [(o, "org") for o in ident.orgs])
    if ident.token_role:
        ui.info(f"token role: {ident.token_role}")
    return 0


def cmd_models(args) -> int:
    from . import providers

    s = Settings.from_env()
    if args.provider:
        s.provider = args.provider
    try:
        provider = providers.build(s)
    except providers.LLMError as exc:
        ui.err(str(exc))
        return 2
    if provider is None:
        ui.err("no provider configured (set an API key or --provider)")
        return 2
    models = sorted(provider.list_models(), key=lambda m: m[1], reverse=True)
    ui.table(f"{provider.name} models (newest first)", ["model id"], [(m[0],) for m in models[: args.limit]])
    return 0


def cmd_plugins(args) -> int:
    from . import catalog

    if args.build_check:
        return catalog.build_check(Path(args.write) if args.write else None)
    rows = [(("✔ " if f.available else "✖ ") + f.key, f.kind, f.title, f.status) for f in catalog.features()]
    ui.table("installed plugins & features", ["feature", "kind", "what", "status"], rows)
    ui.table("not in this image", ["what", "why"], list(catalog.NOT_INSTALLED.items()))
    return 0


def _ver(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "-"


def cmd_doctor(args) -> int:
    from . import hardware, pipeline, providers

    s = Settings.from_env()
    hw = hardware.detect(s.home)
    pipeline.show_hardware(hw)
    pkgs = ["axolotl", "torch", "transformers", "peft", "trl", "accelerate", "datasets", "deepspeed", "liger-kernel",
            "cut-cross-entropy", "axolotl-ringmaster", "flash-linear-attention", "bitsandbytes", "kernels", "llmcompressor",
            "wandb", "anthropic", "openai", "huggingface-hub"]
    ui.table("packages", ["package", "version"], [(p, _ver(p)) for p in pkgs])
    ui.kv_table(
        "environment",
        [
            ("AI provider", providers.detect_provider_name(s)),
            ("HF token", "set" if s.hf_token else "missing (needed for upload / gated models)"),
            ("wandb", f"WANDB_MODE={os.environ.get('WANDB_MODE', '-')} (disabled in this image)"),
            ("FORGE_HOME", s.home),
            ("HF_HOME", os.environ.get("HF_HOME", "-")),
        ],
    )
    notes = Path("/opt/forge/build-notes.txt")
    if notes.exists():
        ui.panel(notes.read_text().strip(), "build notes")
    return 0


def cmd_hw(args) -> int:
    from . import hardware, pipeline

    pipeline.show_hardware(hardware.detect(Settings.from_env().home))
    return 0


def cmd_logs(args) -> int:
    log = Settings.from_env().logs_dir / "latest.log"
    if not log.exists():
        ui.err(f"no run log yet ({log})")
        return 1
    if args.follow:
        os.execvp("tail", ["tail", "-n", "200", "-F", str(log)])
    sys.stdout.write(log.read_text(errors="replace")[-20000:])
    return 0


def cmd_attach(args) -> int:
    if subprocess.call(["tmux", "has-session", "-t", "forge"], stderr=subprocess.DEVNULL) != 0:
        ui.err("no forge job running (start one with: forge run)")
        return 1
    os.execvp("tmux", ["tmux", "attach", "-t", "forge"])
    return 0


def cmd_banner(args) -> int:
    from . import hardware

    ui.banner(f"v{__version__}")
    hw = hardware.detect(Settings.from_env().home)
    ui.console.print(f"   [dim]gpus[/dim]  {hw.gpu_label}")
    ui.console.print("   [dim]start[/dim] [hot]forge[/hot] [dim](guided)  ·  forge --help  ·  forge doctor[/dim]\n")
    return 0


def cmd_purple(args) -> int:
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ui.err("usage: purple <command> [args...]")
        return 2
    return ui.run_purple(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forge",
        description="axolotl-forge: plug-and-play Axolotl fine-tuning with AI-written configs",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"axolotl-forge {__version__}")
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("setup", help="the setup screen (wolfgon): keys, links, Claude, uploads -> train -> back")
    sp.add_argument("--here", action="store_true", help="run in this terminal instead of the tmux session 'forge'")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("run", help="full pipeline: fetch -> AI config -> train -> merge -> upload")
    _common_run_args(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("config", help="fetch + sample + write a config, no training")
    _common_run_args(sp)
    sp.set_defaults(func=cmd_config)

    sp = sub.add_parser("train", help="train (+merge +upload) an existing config")
    sp.add_argument("config_file")
    _common_run_args(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("upload", help="upload a finished run (give its config yaml)")
    sp.add_argument("target", help="the run's config yaml (or its output_dir)")
    _common_run_args(sp)
    sp.set_defaults(func=cmd_upload)

    sp = sub.add_parser("sample", help="show N random sectors + profile of a dataset")
    sp.add_argument("dataset_link")
    sp.add_argument("-n", type=int, default=5)
    sp.add_argument("--model", help="model link to count tokens with its tokenizer")
    sp.add_argument("--seed", type=int)
    sp.add_argument("--json", action="store_true", help="also print what the AI would see")
    sp.add_argument("--home")
    sp.set_defaults(func=cmd_sample)

    sp = sub.add_parser("fetch", help="download a model or dataset")
    sp.add_argument("kind", choices=["model", "dataset"])
    sp.add_argument("link")
    sp.add_argument("--light", action="store_true", help="model: config + tokenizer only")
    sp.add_argument("--home")
    sp.set_defaults(func=cmd_fetch)

    sub.add_parser("orgs", help="list the HF accounts/orgs you can upload to").set_defaults(func=cmd_orgs)

    sp = sub.add_parser("models", help="list models of your AI provider")
    sp.add_argument("--provider", choices=["anthropic", "openai", "openrouter", "custom"])
    sp.add_argument("--limit", type=int, default=40)
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("plugins", help="installed axolotl plugins and their status")
    sp.add_argument("--build-check", action="store_true", help=argparse.SUPPRESS)
    sp.add_argument("--write", help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_plugins)

    sub.add_parser("doctor", help="check GPUs, packages, keys").set_defaults(func=cmd_doctor)
    sub.add_parser("hw", help="show detected hardware").set_defaults(func=cmd_hw)

    sp = sub.add_parser("logs", help="show the latest run log")
    sp.add_argument("-f", "--follow", action="store_true")
    sp.set_defaults(func=cmd_logs)

    sub.add_parser("attach", help="attach to the background training session").set_defaults(func=cmd_attach)

    sp = sub.add_parser("banner")
    sp.add_argument("--entry", action="store_true")
    sp.set_defaults(func=cmd_banner)

    sp = sub.add_parser("purple", help="run any command with purple output")
    sp.add_argument("cmd", nargs=argparse.REMAINDER)
    sp.set_defaults(func=cmd_purple)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args = parser.parse_args(["setup"] if ui.interactive() else ["run"])
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        ui.err("interrupted")
        return 130
    except ui.CmdExit:
        return 0
    except ui.NeedsInput as exc:
        ui.err(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("FORGE_DEBUG"):
            raise
        ui.err(f"{type(exc).__name__}: {exc}")
        return 1
