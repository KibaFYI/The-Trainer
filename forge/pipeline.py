"""The end-to-end run: links -> fetch -> 5 random sectors -> AI config -> preprocess -> train
(crash log -> AI fix -> retry) -> merge -> upload -> Discord -> finish hook."""

from __future__ import annotations

import datetime as _dt
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import autoconfig, fetch, hardware, notify, providers, sampler, ui, upload
from .settings import Settings


class TrainingFailed(RuntimeError):
    def __init__(self, stage: str, error: str, attempts: int):
        super().__init__(f"{stage} failed after {attempts} attempt(s)")
        self.stage, self.error, self.attempts = stage, error, attempts


class _Tee(io.TextIOBase):
    """Send rich console output to the terminal and the run log."""

    def __init__(self, stream, log_path: Path):
        self.stream = stream
        self.log = open(log_path, "a", encoding="utf-8")

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.log.write(s)
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()


def start_log(settings: Settings, tag: str = "run") -> Path:
    settings.ensure_dirs()
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = settings.logs_dir / f"{tag}-{stamp}.log"
    log_path.touch()
    latest = settings.logs_dir / "latest.log"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(log_path.name)
    except OSError:
        pass
    stop_log()
    ui.console.file = _Tee(sys.stdout, log_path)
    return log_path


def stop_log() -> None:
    """Stop copying console output into the run log."""
    if isinstance(ui.console.file, _Tee):
        ui.console.file.log.close()
        ui.console.file = ui.console.file.stream


class RunState:
    def __init__(self, settings: Settings, run_name: str):
        self.path = settings.runs_dir / run_name / "state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"run": run_name, "stages": {}}

    def mark(self, stage: str, status: str, **extra) -> None:
        self.data["stages"][stage] = {"status": status, "at": _dt.datetime.now().isoformat(timespec="seconds"), **extra}
        self.path.write_text(json.dumps(self.data, indent=2))


def show_hardware(hw: hardware.Hardware) -> None:
    rows = [("GPUs", hw.gpu_label)]
    if hw.gpus:
        rows.append(("arch", ", ".join(sorted({g.arch for g in hw.gpus}))))
        rows.append(("NVLink", "yes" if hw.nvlink else "no"))
        rows.append(("driver / CUDA", f"{hw.driver} / {hw.cuda}"))
    rows += [("CPU / RAM", f"{hw.cpu_count} cores / {hw.ram_gb:.0f} GB"), ("disk free", f"{hw.disk_free_gb:.0f} GB")]
    rows.append(("torch / axolotl", f"{hw.torch or '-'} / {hw.axolotl or '-'}"))
    ui.kv_table("this machine", rows)


def show_profile(prof: sampler.DatasetProfile) -> None:
    ui.kv_table(
        "dataset profile",
        [
            ("format", f"{prof.fmt} -> {prof.detected.get('format')}"),
            ("size", f"{prof.total_mb:,.1f} MB, ~{prof.est_rows:,} rows" if prof.est_rows else f"{prof.total_mb:,.1f} MB"),
            ("tokens / sample", f"p50 {prof.lengths['p50']} · p90 {prof.lengths['p90']} · p99 {prof.lengths['p99']} · max {prof.lengths['max']}  ({prof.token_counter})"),
            ("total tokens", f"~{prof.lengths.get('est_total_tokens_millions', '?')} M"),
            ("fields", ", ".join(f"{k} {v:.0f}%" for k, v in list(prof.fields.items())[:10])),
        ],
    )
    for i, s in enumerate(prof.samples, start=1):
        text = json.dumps(sampler._truncate(s.record, 400), indent=2, ensure_ascii=False, default=str)
        lines = text.splitlines()
        if len(lines) > 30:
            text = "\n".join(lines[:30]) + f"\n… (+{len(lines) - 30} lines)"
        ui.panel(text, f"sector {i}/{len(prof.samples)} · {s.where}")


# ── training ─────────────────────────────────────────────────────────────────────

def axolotl_cmd() -> list[str]:
    return shlex.split(os.environ.get("FORGE_AXOLOTL_BIN", "axolotl"))


def train_env(settings: Settings) -> dict:
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("AXOLOTL_DO_NOT_TRACK", "1")
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token
    return env


_ANSI = re.compile(r"\x1b\[[0-9;:?]*[A-Za-z]|\x1b\][^\x07]*\x07")


def log_tail(log_path: Path, start: int, max_chars: int = 12000) -> str:
    """The useful end of what a step printed: from the last traceback, else the last lines."""
    with open(log_path, "rb") as fh:
        fh.seek(start)
        raw = fh.read().decode("utf-8", errors="replace")
    text = _ANSI.sub("", raw)
    lines = [ln.split("\r")[-1] for ln in text.split("\n")]  # keep only the final state of progress bars
    text = "\n".join(ln for ln in lines if ln.strip())
    idx = text.rfind("Traceback (most recent call last)")
    if idx != -1:
        text = text[idx:]
    else:
        text = "\n".join(text.splitlines()[-150:])
    return text[-max_chars:]


def train_once(settings: Settings, cfg_path: Path, hw: hardware.Hardware, log_path: Path) -> tuple[int, str, str]:
    """preprocess + train. Returns (exit code, failed stage, error text)."""
    env = train_env(settings)
    for stage in ("preprocess", "train"):
        cmd = [*axolotl_cmd(), stage, str(cfg_path)]
        if stage == "train" and hw.gpu_count > 1:
            cmd += ["--launcher", "accelerate", "--", f"--num_processes={hw.gpu_count}"]
        ui.info(" ".join(cmd))
        start = log_path.stat().st_size
        rc = ui.run_purple(cmd, log_path, env=env)
        if rc != 0:
            return rc, stage, log_tail(log_path, start)
    return 0, "", ""


def train_with_fixes(
    settings: Settings,
    cfg_path: Path,
    hw: hardware.Hardware,
    log_path: Path,
    drafter: autoconfig.Drafter | None,
    notifier: notify.Discord,
) -> int:
    """Train; on a crash hand the log to the AI, take its fixed config and try again.
    Returns the number of attempts it took. Raises TrainingFailed after the last attempt."""
    max_attempts = max(1, settings.train_attempts)
    state = RunState(settings, cfg_path.stem)
    stage, error = "train", ""
    for attempt in range(1, max_attempts + 1):
        ui.step(f"training · attempt {attempt}/{max_attempts}")
        t0 = time.time()
        state.mark("train", "running", attempt=attempt)
        rc, stage, error = train_once(settings, cfg_path, hw, log_path)
        if rc == 0:
            state.mark("train", "done", attempt=attempt, hours=round((time.time() - t0) / 3600, 2))
            ui.ok(f"training finished in {(time.time() - t0) / 3600:.2f} h (attempt {attempt})")
            return attempt
        if rc == ui.USER_CANCELLED:
            state.mark("train", "cancelled", attempt=attempt)
            raise KeyboardInterrupt
        state.mark("train", "failed", attempt=attempt, rc=rc, failed_stage=stage)
        ui.err(f"{stage} failed (exit {rc}) on attempt {attempt}/{max_attempts}")
        ui.panel(error[-2500:] or "(no output)", f"{stage} error")
        if attempt == max_attempts:
            break
        if drafter is None:
            ui.warn("no AI configured to fix the config - stopping here")
            break
        if drafter.fix_training_error(error, attempt, max_attempts):
            ui.ok("new config ready - trying again")
        else:
            ui.warn("the AI could not produce a fix - retrying the same config")
    notifier.failed(stage, error, attempts=attempt)
    raise TrainingFailed(stage, error, attempt)


def merge_lora(settings: Settings, cfg_path: Path, log_path: Path) -> bool:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("adapter") not in ("lora", "qlora"):
        return False
    ui.step("merge LoRA into the base model")
    rc = ui.run_purple([*axolotl_cmd(), "merge-lora", str(cfg_path)], log_path, env=train_env(settings))
    if rc == ui.USER_CANCELLED:
        raise KeyboardInterrupt
    if rc == 0:
        ui.ok(f"merged model: {Path(cfg['output_dir']) / 'merged'}")
        return True
    ui.warn("merge failed - the adapter is still fine")
    return False


# ── classic flow (`forge run` / `forge train`) ───────────────────────────────────

def _resolve_config(settings: Settings, run_name: str) -> Path:
    src = settings.config or ""
    dest = settings.configs_dir / f"{run_name}.yml"
    if src.startswith(("http://", "https://")):
        got = fetch.download_url(src, settings.configs_dir / "_downloaded")
        shutil.copy(got, dest)
    else:
        shutil.copy(Path(src).expanduser(), dest)
    ui.ok(f"using your config: {dest}")
    return dest


def _choose_provider(settings: Settings) -> providers.Provider | None:
    try:
        if ui.interactive():
            return providers.interactive_pick(settings)
        provider = providers.build(settings)
        if provider is None:
            ui.warn("no AI key found (ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY / FORGE_LLM_BASE_URL) - using the rules engine")
        return provider
    except providers.LLMError as exc:
        ui.warn(f"{exc} - using the rules engine")
        return None


def _plan_upload(settings: Settings, cfg_path: Path, dataset_link: str) -> None:
    """Pick the HF namespace + repo now, so the end of training needs no input."""
    if not settings.upload:
        ui.info("upload disabled (FORGE_UPLOAD=0)")
        return
    if not settings.hf_token:
        if ui.interactive():
            token = ui.ask("Hugging Face write token for the upload (blank = skip upload)", "", password=True)
            if token:
                settings.hf_token = token
                os.environ["HF_TOKEN"] = token
        if not settings.hf_token:
            ui.warn("no HF_TOKEN - the model will stay on disk (upload later with: forge upload)")
            settings.upload = False
            return
    try:
        ident = upload.identity(settings.hf_token)
    except Exception as exc:  # noqa: BLE001
        ui.warn(f"could not read your HF account ({exc}); upload disabled")
        settings.upload = False
        return
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ns = upload.pick_namespace(settings, ident)
    method = cfg.get("adapter") or "full"
    name = upload.pick_repo_name(settings, ns, upload.default_repo_name(cfg.get("base_model", "model"), dataset_link, method))
    settings.hub_repo = f"{ns}/{name}"
    if ui.interactive():
        settings.hub_private = ui.confirm("Make the repo private?", settings.hub_private)
    ui.ok(f"will upload to [hot]{settings.hub_repo}[/hot] ({'private' if settings.hub_private else 'public'})")


def train_and_ship(settings: Settings, cfg_path: Path, dataset_link: str, dataset_repo: str | None,
                   hw: hardware.Hardware, log_path: Path, drafter: autoconfig.Drafter | None = None) -> int:
    notifier = notify.Discord(settings.discord_webhook, cfg_path.stem)
    try:
        attempts = train_with_fixes(settings, cfg_path, hw, log_path, drafter, notifier)
    except TrainingFailed as exc:
        ui.err(f"{exc} - full log: {log_path}")
        return 1
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    summary = upload.train_summary(Path(cfg["output_dir"]))
    notifier.training_done(summary, attempts)

    want_merge = settings.merge in ("1", "true", "yes") or (settings.merge == "auto" and settings.upload_what != "adapter")
    if want_merge:
        merge_lora(settings, cfg_path, log_path)

    urls: list[str] = []
    if settings.upload and settings.hub_repo:
        ui.step("upload")
        try:
            urls = upload.upload_run(settings, cfg_path, dataset_link, dataset_repo, hw.gpu_label)
            notifier.uploaded(urls)
        except Exception as exc:  # noqa: BLE001
            ui.err(f"upload failed: {exc}")
            notifier.failed("upload", str(exc))
            ui.info(f"retry with: forge upload {cfg_path}")
    ui.step("done")
    ui.kv_table("summary", [("config", cfg_path), ("output", cfg["output_dir"]), ("log", log_path),
                            ("uploaded", ", ".join(urls) or "-")])
    notifier.finished({"final loss": summary.get("final_loss"), "uploaded": "\n".join(urls) or "-"})
    return 0


def _finish_hook(settings: Settings) -> None:
    if settings.on_finish:
        ui.info(f"running FORGE_ON_FINISH_CMD: {settings.on_finish}")
        subprocess.call(settings.on_finish, shell=True)


def prepare(settings: Settings, log_path: Path):
    """Everything up to an accepted config. Returns (config path, model, data, hw, drafter)."""
    hw = hardware.detect(settings.home)
    show_hardware(hw)
    if not hw.gpus:
        ui.warn("no NVIDIA GPU visible - configs can still be generated, training will not run")
    if hw.disk_free_gb and hw.disk_free_gb < 30:
        ui.warn(f"only {hw.disk_free_gb:.0f} GB free in {settings.home} - large models will not fit")

    if not settings.model:
        settings.model = ui.ask("Model link (HF id like Qwen/Qwen3-8B, HF url, https://, s3://, local path)", env_hint="FORGE_MODEL")
    if not settings.dataset:
        settings.dataset = ui.ask("Dataset link (HF id/url, https:// .jsonl/.parquet/.zip, Google Drive, s3://, local path)", env_hint="FORGE_DATASET")
    if ui.interactive() and not settings.config and not settings.goal:
        settings.goal = ui.ask("Anything the AI should know? (goal, epochs, context length...) - optional", "")

    ui.step("model")
    model = fetch.fetch_model(settings.model, settings)
    ui.step("dataset")
    data = fetch.fetch_dataset(settings.dataset, settings)
    run_name = autoconfig.run_name_for(settings, model, data)

    if settings.config:
        ui.step("config")
        cfg_path = _resolve_config(settings, run_name)
        ui.show_yaml(cfg_path.read_text(encoding="utf-8"), str(cfg_path))
        return cfg_path, model, data, hw, None

    ui.step(f"{settings.sample_count} random sectors of the dataset")
    prof = sampler.profile(data, settings.sample_count, model_dir=model.local_dir)
    show_profile(prof)

    ui.step("AI config")
    provider = _choose_provider(settings) if settings.provider != "none" else None
    cfg_path, drafter = autoconfig.generate(settings, hw, model, data, prof, provider, run_name)
    ui.ok(f"config saved: {cfg_path}")
    return cfg_path, model, data, hw, drafter


def run(settings: Settings) -> int:
    log_path = start_log(settings)
    ui.banner(f"run log: {log_path}")
    notifier = notify.Discord(settings.discord_webhook)
    try:
        cfg_path, model, data, hw, drafter = prepare(settings, log_path)
        _plan_upload(settings, cfg_path, data.link)
        rc = train_and_ship(settings, cfg_path, data.link, data.hf_repo, hw, log_path, drafter)
    except (ui.NeedsInput, ui.CmdExit) as exc:
        if str(exc):
            ui.err(str(exc))
        return 2
    except KeyboardInterrupt:
        ui.err("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001
        ui.err(f"{type(exc).__name__}: {exc}")
        notifier.failed("run", f"{type(exc).__name__}: {exc}")
        if os.environ.get("FORGE_DEBUG"):
            raise
        return 1
    finally:
        _finish_hook(settings)
    return rc


def config_only(settings: Settings) -> int:
    log_path = start_log(settings, "config")
    ui.banner("config generator")
    try:
        cfg_path, *_ = prepare(settings, log_path)
    except (ui.NeedsInput, ui.CmdExit) as exc:
        if str(exc):
            ui.err(str(exc))
        return 2
    ui.info(f"train it with: forge train {cfg_path}")
    return 0


def train_existing(settings: Settings, cfg_path: Path) -> int:
    log_path = start_log(settings, "train")
    ui.banner(f"training {cfg_path.name}")
    hw = hardware.detect(settings.home)
    show_hardware(hw)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ds = (cfg.get("datasets") or [{}])[0]
    dataset_link = settings.dataset or str(ds.get("path", "dataset"))
    try:
        _plan_upload(settings, cfg_path, dataset_link)
        return train_and_ship(settings, cfg_path, dataset_link, None, hw, log_path)
    finally:
        _finish_hook(settings)
