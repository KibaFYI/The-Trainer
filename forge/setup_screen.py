"""The first-boot setup screen.

The wolfgon sits in the middle of the terminal with one question under it at a time:
HF token -> AI key -> model -> dataset -> Claude Sonnet/Opus -> model name -> where the LoRA
goes -> where the merged model goes -> Discord webhook. Then: fetch -> Claude writes the
config -> you look through / edit it -> training starts; a crash goes back to Claude for a
fix (3 attempts, then a Discord alert) -> merge -> uploads -> Discord -> back to the start.
Type cmd-exit at any prompt to leave.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.align import Align
from rich.console import Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import art, autoconfig, fetch, hardware, notify, pipeline, providers, sampler, ui, upload
from .settings import Settings

DISMISSED = Path(os.environ.get("FORGE_SETUP_DISMISSED", "/tmp/forge-setup-dismissed"))
SKIP = "__skip__"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
CLAUDE = {
    "sonnet": {"label": "Claude Sonnet 5.5", "anthropic": "claude-sonnet-5-5", "openrouter": "anthropic/claude-sonnet-5.5",
               "hint": "fast and cheaper - great for most configs"},
    "opus": {"label": "Claude Opus 5.5", "anthropic": "claude-opus-5-5", "openrouter": "anthropic/claude-opus-5.5",
             "hint": "the strongest - best for tricky models, MoE and multi-GPU"},
}
SAVED_FIELDS = ("model_url", "dataset_url", "claude", "name", "lora_ns", "merged_ns", "private", "goal")


def _mask(secret: str | None) -> str:
    if not secret:
        return "-"
    return secret[:6] + "…" + secret[-4:] if len(secret) > 12 else "…" * 3


def _short(exc: Exception) -> str:
    return re.sub(r"\s+", " ", str(exc))[:160] or type(exc).__name__


# ── answers ───────────────────────────────────────────────────────────────────────

@dataclass
class Answers:
    hf_token: str | None = None
    ai_key: str | None = None
    model_url: str | None = None
    dataset_url: str | None = None
    claude: str = "sonnet"
    name: str | None = None
    lora_ns: str | None = None
    merged_ns: str | None = None
    webhook: str | None = None
    private: bool = True
    goal: str = ""
    hf_user: str | None = None
    hf_orgs: list[str] = field(default_factory=list)

    @classmethod
    def initial(cls, settings: Settings) -> "Answers":
        a = cls(
            hf_token=settings.hf_token,
            ai_key=os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
            model_url=settings.model,
            dataset_url=settings.dataset,
            webhook=settings.discord_webhook,
            private=settings.hub_private,
            goal=settings.goal,
        )
        if settings.llm_model and "opus" in settings.llm_model:
            a.claude = "opus"
        saved = settings.home / "setup.json"
        if saved.exists():
            try:
                data = json.loads(saved.read_text())
                for k in SAVED_FIELDS:
                    if k in data and getattr(a, k) in (None, "", "sonnet", True):
                        setattr(a, k, data[k])
            except (ValueError, OSError):
                pass
        return a

    def save(self, settings: Settings) -> None:
        """Remember the non-secret answers for the next boot (keys are never written to disk)."""
        try:
            settings.home.mkdir(parents=True, exist_ok=True)
            (settings.home / "setup.json").write_text(json.dumps({k: getattr(self, k) for k in SAVED_FIELDS}, indent=2))
        except OSError:
            pass

    @property
    def ai_kind(self) -> str | None:
        if not self.ai_key:
            return None
        return "openrouter" if self.ai_key.startswith("sk-or-") else "anthropic"

    @property
    def lora_repo(self) -> str | None:
        if not self.hf_token or not self.name or self.lora_ns in (None, SKIP):
            return None
        return f"{self.lora_ns}/{self.name}-lora"

    @property
    def merged_repo(self) -> str | None:
        if not self.hf_token or not self.name or self.merged_ns in (None, SKIP):
            return None
        return f"{self.merged_ns}/{self.name}"


# ── the screen ────────────────────────────────────────────────────────────────────

class Screen:
    def __init__(self) -> None:
        self.status = ""
        self._last: tuple | None = None
        if hasattr(signal, "SIGWINCH"):
            try:
                signal.signal(signal.SIGWINCH, self._on_resize)
            except (ValueError, OSError):
                pass

    @property
    def panel_width(self) -> int:
        return max(40, min(80, ui.console.width - 4))

    def prompt_prefix(self) -> str:
        pad = max(0, (ui.console.width - self.panel_width) // 2 + 2)
        return " " * pad + f"[bold {ui.HOT}]❯[/] "

    def _on_resize(self, *_args) -> None:
        if self._last:
            self.draw(*self._last)
            ui.console.print(self.prompt_prefix(), end="")

    def deactivate(self) -> None:
        self._last = None

    def draw(self, step: str, title: str, body, error: str | None = None) -> None:
        self._last = (step, title, body, error)
        c = ui.console
        w, h = c.size
        parts = []
        if isinstance(body, list):
            parts.append(Text.from_markup("\n".join(body)))
        else:
            parts.append(body)
        if error:
            parts.append(Text.from_markup(f"\n[bold {ui.ERR}]✖ {escape(error)}[/]"))
        panel = Panel(
            Group(*parts),
            title=f"[bold {ui.HOT}] {escape(title)} [/]",
            subtitle=f"[{ui.DIM}] {escape(step)} [/]" if step else None,
            border_style=ui.DEEP,
            width=self.panel_width,
            padding=(1, 2),
        )
        panel_h = len(c.render_lines(panel, c.options.update(width=self.panel_width), pad=False))
        show_art = w >= art.width() + 2
        total = (art.height() + 1 if show_art else 1) + 2 + panel_h + 2
        c.clear()
        top = max(0, (h - total) // 2)
        if top:
            c.print("\n" * (top - 1))
        if show_art:
            art.print_centered(c, w)
        else:
            c.print(Align.center(Text("🐺 axolotl-forge", style=f"bold {ui.HOT}")))
        c.print()
        c.print(Align.center(Text("a x o l o t l · f o r g e", style=f"bold {ui.PURPLE}")))
        if self.status:
            c.print(Align.center(Text(self.status, style=ui.DIM)))
        c.print(Align.center(panel))
        c.print(Align.center(Text("type cmd-exit at any prompt to leave", style=f"italic {ui.DIM}")))

    def ask(self, step: str, title: str, body, error: str | None = None, password: bool = False) -> str:
        self.draw(step, title, body, error)
        return ui.read_line(self.prompt_prefix(), password=password).strip()

    def pick(self, step: str, title: str, intro: list[str], options: list[tuple[str, str]], default: int = 0,
             error: str | None = None) -> str:
        while True:
            body = list(intro) + [""] + [
                f"  [{ui.HOT}]{'❯' if i == default else ' '}[/] [bold {ui.DEEP}]{i + 1}[/]  {label}"
                for i, (_, label) in enumerate(options)
            ] + ["", f"[{ui.DIM}]number, or Enter for {default + 1}[/]"]
            raw = self.ask(step, title, body, error)
            if not raw:
                return options[default][0]
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1][0]
            error = f"pick 1-{len(options)}"


# ── the questions ─────────────────────────────────────────────────────────────────

def _check_hf_repo(link: str, kind: str, token: str | None) -> str | None:
    """Fail fast on typos / gated repos. Returns an error message or None."""
    try:
        src = fetch.classify(link, kind)
    except ValueError as exc:
        return str(exc)
    if src.kind != "hf":
        return None
    if kind == "model" and src.repo_type == "dataset":
        return "that is a dataset link - paste a model"
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        if kind == "model":
            api.model_info(src.ref, revision=src.revision)
        else:
            api.dataset_info(src.ref, revision=src.revision)
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if "Gated" in name:
            return f"{src.ref} is gated - accept its terms on huggingface.co with this account"
        if "NotFound" in name or "404" in str(exc):
            return f"{src.ref} not found (typo? private repos need a token with access)"
        return None  # network hiccups: let the download try
    return None


def step_hf(screen: Screen, a: Answers, step: str) -> None:
    error = None
    while True:
        body = [
            f"Paste your [bold {ui.HOT}]Hugging Face token[/] - it needs [b]write[/b] access to upload.",
            f"[{ui.DIM}]huggingface.co/settings/tokens · type skip to train without uploading[/]",
        ]
        if a.hf_token:
            who = f" · logged in as [bold {ui.HOT}]{escape(a.hf_user)}[/]" if a.hf_user else ""
            body += ["", f"saved: {_mask(a.hf_token)}{who}  [{ui.DIM}](Enter keeps it)[/]"]
        value = screen.ask(step, "Hugging Face token", body, error, password=True) or (a.hf_token or "")
        if value.lower() == "skip":
            a.hf_token, a.hf_user, a.hf_orgs = None, None, []
            a.lora_ns = a.merged_ns = SKIP
            return
        if not value:
            error = "a token is needed (or type skip)"
            continue
        try:
            with ui.console.status("checking the token with Hugging Face…"):
                ident = upload.identity(value)
        except Exception as exc:  # noqa: BLE001
            error = f"Hugging Face rejected that token: {_short(exc)}"
            continue
        if ident.token_role and ident.token_role not in ("write", "fineGrained", "admin"):
            error = f"that token is '{ident.token_role}' - uploads need a write token (or type skip)"
            continue
        a.hf_token, a.hf_user, a.hf_orgs = value, ident.user, ident.orgs
        os.environ["HF_TOKEN"] = value
        return


def _check_ai_key(key: str) -> str | None:
    if key.startswith("sk-or-"):
        import requests

        try:
            resp = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=20)
        except Exception as exc:  # noqa: BLE001
            return f"could not reach OpenRouter: {_short(exc)}"
        return None if resp.status_code == 200 else f"OpenRouter rejected the key ({resp.status_code})"
    try:
        providers.AnthropicProvider(key, CLAUDE["sonnet"]["anthropic"]).list_models()
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if "Authentication" in name:
            return "Anthropic rejected that key"
        if "Connection" in name:
            return "could not reach the Anthropic API"
        return f"key check failed: {_short(exc)}"
    return None


def step_ai(screen: Screen, a: Answers, step: str) -> None:
    error = None
    while True:
        body = [
            f"Paste your [bold {ui.HOT}]Anthropic API key[/] (sk-ant-…). An OpenRouter key (sk-or-…) works too.",
            f"[{ui.DIM}]Claude writes the training config and fixes it if training crashes.[/]",
            f"[{ui.DIM}]type skip to use forge's rules engine instead (no AI fixes)[/]",
        ]
        if a.ai_key:
            kind = "OpenRouter" if a.ai_kind == "openrouter" else "Anthropic"
            body += ["", f"saved: {_mask(a.ai_key)} ({kind})  [{ui.DIM}](Enter keeps it)[/]"]
        value = screen.ask(step, "AI key", body, error, password=True) or (a.ai_key or "")
        if value.lower() == "skip":
            a.ai_key = None
            return
        if not value:
            error = "paste a key (or type skip)"
            continue
        with ui.console.status("checking the key…"):
            error = _check_ai_key(value)
        if error:
            continue
        a.ai_key = value
        return


def _ask_link(screen: Screen, a: Answers, step: str, kind: str) -> None:
    attr = "model_url" if kind == "model" else "dataset_url"
    title = "Model to train" if kind == "model" else "Dataset"
    examples = (
        ["Qwen/Qwen3-8B", "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"]
        if kind == "model"
        else ["https://huggingface.co/datasets/mlabonne/FineTome-100k", "https://example.com/chats.jsonl (.parquet .zip ok)"]
    )
    error = None
    while True:
        current = getattr(a, attr)
        body = [
            f"Paste the [bold {ui.HOT}]{kind}[/] link.",
            f"[{ui.DIM}]HF id or url · https:// file · Google Drive · s3:// · local path[/]",
            "",
            *[f"[{ui.DIM}]e.g. {escape(e)}[/]" for e in examples],
        ]
        if current:
            body += ["", f"current: [bold {ui.HOT}]{escape(current)}[/]  [{ui.DIM}](Enter keeps it)[/]"]
        value = screen.ask(step, title, body, error) or (current or "")
        if not value:
            error = f"a {kind} link is needed"
            continue
        with ui.console.status("checking the link…"):
            error = _check_hf_repo(value, kind, a.hf_token)
        if error:
            continue
        setattr(a, attr, value)
        return


def step_model(screen, a, step):
    _ask_link(screen, a, step, "model")


def step_dataset(screen, a, step):
    _ask_link(screen, a, step, "dataset")


def step_claude(screen: Screen, a: Answers, step: str) -> None:
    if not a.ai_key:
        return
    options = [(k, f"{v['label']}  [{ui.DIM}]{v['hint']}[/]") for k, v in CLAUDE.items()]
    default = list(CLAUDE).index(a.claude) if a.claude in CLAUDE else 0
    a.claude = screen.pick(step, "Which Claude writes the config?", ["It reads 5 random sectors of your dataset, the installed plugins and this machine's GPUs."], options, default)


def _default_name(a: Answers) -> str:
    base = upload.default_repo_name(a.model_url or "model", a.dataset_url or "data", "full")
    return base.lower()


def step_name(screen: Screen, a: Answers, step: str) -> None:
    error = None
    while True:
        default = a.name or _default_name(a)
        body = [
            f"What should the [bold {ui.HOT}]fine-tuned model[/] be called?",
            f"[{ui.DIM}]merged model -> <place>/{escape(default)} · LoRA -> <place>/{escape(default)}-lora[/]",
            "",
            f"suggested: [bold {ui.HOT}]{escape(default)}[/]  [{ui.DIM}](Enter keeps it)[/]",
        ]
        value = screen.ask(step, "Model name", body, error) or default
        if not NAME_RE.match(value):
            error = "letters, numbers, . _ - only (max 96)"
            continue
        a.name = value
        return


def _place_options(a: Answers, skip_label: str) -> list[tuple[str, str]]:
    opts = []
    if a.hf_user:
        opts.append((a.hf_user, f"{escape(a.hf_user)}  [{ui.DIM}](you)[/]"))
    opts += [(o, f"{escape(o)}  [{ui.DIM}](org)[/]") for o in a.hf_orgs]
    opts.append((SKIP, f"[{ui.DIM}]{skip_label}[/]"))
    return opts


def step_lora(screen: Screen, a: Answers, step: str) -> None:
    if not a.hf_token:
        a.lora_ns = SKIP
        return
    opts = _place_options(a, "don't upload the LoRA")
    values = [v for v, _ in opts]
    default = values.index(a.lora_ns) if a.lora_ns in values else 0
    a.lora_ns = screen.pick(step, "Where does the LoRA go?", [
        f"The LoRA adapter uploads as [bold {ui.HOT}]<place>/{escape(a.name or '')}-lora[/].",
        f"[{ui.DIM}]your account and every org your token can write to:[/]",
    ], opts, default)


def step_merged(screen: Screen, a: Answers, step: str) -> None:
    if not a.hf_token:
        a.merged_ns = SKIP
        return
    opts = _place_options(a, "don't merge / don't upload a merged model")
    values = [v for v, _ in opts]
    default = values.index(a.merged_ns) if a.merged_ns in values else 0
    a.merged_ns = screen.pick(step, "Where does the merged model go?", [
        f"The LoRA gets merged into the base and uploads as [bold {ui.HOT}]<place>/{escape(a.name or '')}[/].",
    ], opts, default)


def step_webhook(screen: Screen, a: Answers, step: str) -> None:
    error = None
    while True:
        body = [
            f"[bold {ui.HOT}]Discord webhook[/] for pings: training done · uploaded · all finished · failed 3x",
            f"[{ui.DIM}]Server settings -> Integrations -> Webhooks -> copy URL · Enter to skip · none to turn off[/]",
        ]
        if a.webhook:
            body += ["", f"saved: {_mask(a.webhook)}  [{ui.DIM}](Enter keeps it)[/]"]
        value = screen.ask(step, "Discord webhook (optional)", body, error, password=True)
        if not value:
            return
        if value.lower() in ("none", "off", "skip"):
            a.webhook = None
            return
        if not notify.valid_webhook(value):
            error = "that does not look like https://discord.com/api/webhooks/…"
            continue
        a.webhook = value
        return


STEPS = [
    ("Hugging Face token", step_hf),
    ("AI key", step_ai),
    ("model", step_model),
    ("dataset", step_dataset),
    ("Claude", step_claude),
    ("name", step_name),
    ("LoRA upload", step_lora),
    ("merged upload", step_merged),
    ("Discord", step_webhook),
]


def _summary_table(a: Answers) -> Table:
    t = Table(show_header=False, box=None, padding=(0, 1))
    t.add_column(style=f"bold {ui.DEEP}", no_wrap=True)
    t.add_column(style=f"bold {ui.HOT}", no_wrap=True)
    t.add_column(style=ui.PURPLE, overflow="fold")
    ai = "-"
    if a.ai_key:
        ai = f"{'OpenRouter' if a.ai_kind == 'openrouter' else 'Anthropic'} {_mask(a.ai_key)}"
    rows = [
        ("1", "Hugging Face", f"{a.hf_user} ✔" if a.hf_user else "skipped (no uploads)"),
        ("2", "AI key", ai if a.ai_key else "skipped (rules engine)"),
        ("3", "model", a.model_url or "-"),
        ("4", "dataset", a.dataset_url or "-"),
        ("5", "Claude", CLAUDE[a.claude]["label"] if a.ai_key else "-"),
        ("6", "name", a.name or "-"),
        ("7", "LoRA ->", a.lora_repo or "not uploaded"),
        ("8", "merged ->", a.merged_repo or "not merged / not uploaded"),
        ("9", "Discord", "on ✔" if a.webhook else "off"),
        ("n", "notes for AI", a.goal or "-"),
        ("p", "visibility", "private" if a.private else "public"),
    ]
    for key, label, value in rows:
        t.add_row(key, label, escape(str(value)))
    return t


def collect(screen: Screen, a: Answers) -> None:
    total = len(STEPS)
    for i, (_, fn) in enumerate(STEPS, start=1):
        fn(screen, a, f"step {i}/{total}")
    error = None
    while True:
        body = Group(
            _summary_table(a),
            Text.from_markup(f"\n[{ui.HOT}]Enter[/] start  ·  [{ui.HOT}]1-9 n p[/] change something"),
        )
        raw = screen.ask("ready", "Ready to forge", body, error).lower()
        error = None
        if raw == "":
            if not a.model_url or not a.dataset_url:
                error = "model and dataset are needed"
                continue
            return
        if raw == "p":
            a.private = not a.private
        elif raw == "n":
            a.goal = screen.ask("notes", "Notes for the AI (optional)", [
                "Anything Claude should know? e.g. [i]roleplay chat, 8k context, 3 epochs[/i]",
                f"[{ui.DIM}]Enter keeps: {escape(a.goal or '-')} · none clears[/]",
            ]) or a.goal
            if a.goal.lower() == "none":
                a.goal = ""
        elif raw.isdigit() and 1 <= int(raw) <= total:
            STEPS[int(raw) - 1][1](screen, a, f"step {raw}/{total}")
        else:
            error = "Enter, 1-9, n or p"


# ── the run ───────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    ui.console.clear()
    ui.console.print(Align.center(Text("a x o l o t l · f o r g e", style=f"bold {ui.PURPLE}")))
    ui.console.print(Align.center(Text(title, style=ui.DIM)))


def review_config(screen: Screen, drafter: autoconfig.Drafter, prof: sampler.DatasetProfile) -> Path:
    """Show the config; Enter starts training. e = edit, d = dataset samples, r = ask the AI, b = baseline."""
    note = ""
    while True:
        _header("look through your training config · Enter starts training")
        ui.show_yaml(drafter.path.read_text(encoding="utf-8"), title=f"{drafter.path.name} · written by {drafter.source}")
        if note:
            ui.console.print(f"  {note}")
        ai = "[hot]r[/hot] ask Claude to change something  ·  " if drafter.provider else ""
        ui.console.print(
            f"  [hot]Enter[/hot] start training  ·  [hot]e[/hot] edit in {os.environ.get('EDITOR', 'nano')}  ·  "
            f"[hot]d[/hot] dataset samples  ·  {ai}[hot]b[/hot] rules-engine baseline  ·  [dim]cmd-exit[/dim]"
        )
        choice = ui.read_line(f"  [bold {ui.HOT}]❯[/] ").strip().lower()
        note = ""
        if choice == "":
            return drafter.path
        if choice == "e":
            subprocess.call([os.environ.get("EDITOR", "nano"), str(drafter.path)])
            with ui.console.status("validating your edit…"):
                ok, err = autoconfig.validate(drafter.path)
            if ok is False:
                ui.panel(escape(err[-2500:]), "axolotl says")
                ui.read_line(f"  [{ui.DIM}]fix it with e again - Enter to continue[/] ")
            elif ok:
                note = f"[ok]✔ axolotl accepted your edit[/ok]"
        elif choice == "d":
            _header("5 random sectors of your dataset")
            pipeline.show_profile(prof)
            ui.read_line(f"  [{ui.DIM}]Enter to go back to the config[/] ")
        elif choice == "r" and drafter.provider:
            what = ui.read_line(f"  [bold {ui.HOT}]what should Claude change?[/] ").strip()
            if what:
                drafter.draft(what)
        elif choice == "b":
            drafter.use_baseline()
            note = "[dim]switched to the rules-engine baseline[/dim]"


def execute(screen: Screen, a: Answers, base: Settings) -> str:
    s = dataclasses.replace(base)
    s.model, s.dataset = a.model_url, a.dataset_url
    s.method = "adapter"
    s.goal = a.goal
    s.hf_token = a.hf_token
    s.hub_private = a.private
    s.discord_webhook = a.webhook
    s.upload = bool(a.lora_repo or a.merged_repo)
    s.run_name = f"{a.name}-{_dt.datetime.now():%m%d-%H%M}"
    if a.ai_kind == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = a.ai_key or ""
        s.provider, s.llm_model = "anthropic", CLAUDE[a.claude]["anthropic"]
    elif a.ai_kind == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = a.ai_key or ""
        s.provider, s.llm_model = "openrouter", CLAUDE[a.claude]["openrouter"]
    else:
        s.provider = "none"
    if a.hf_token:
        os.environ["HF_TOKEN"] = a.hf_token
    a.save(s)

    screen.deactivate()
    log_path = pipeline.start_log(s, "setup")
    notifier = notify.Discord(a.webhook, s.run_name)
    _header(f"run {s.run_name} · log {log_path}")

    hw = hardware.detect(s.home)
    pipeline.show_hardware(hw)
    ui.step("model")
    model = fetch.fetch_model(a.model_url, s)
    ui.step("dataset")
    data = fetch.fetch_dataset(a.dataset_url, s)
    ui.step(f"{s.sample_count} random sectors")
    prof = sampler.profile(data, s.sample_count, model_dir=model.local_dir)
    ui.ok(f"{prof.detected.get('format')} · ~{prof.est_rows or '?'} rows · p99 {prof.lengths['p99']} tokens")

    ui.step("config")
    provider = providers.build(s) if s.provider != "none" else None
    drafter = autoconfig.Drafter(s, hw, model, data, prof, provider, s.run_name)
    drafter.draft()
    cfg_path = review_config(screen, drafter, prof)

    _header(f"training {s.run_name} · Ctrl+C stops it")
    attempts = pipeline.train_with_fixes(s, cfg_path, hw, log_path, drafter, notifier)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    summary = upload.train_summary(Path(cfg["output_dir"]))
    notifier.training_done(summary, attempts)

    merged_ok = pipeline.merge_lora(s, cfg_path, log_path) if a.merged_repo else False
    urls: list[str] = []
    if a.lora_repo or (a.merged_repo and merged_ok):
        ui.step("upload")
        urls = upload.upload_targets(s, cfg_path, a.lora_repo, a.merged_repo if merged_ok else None,
                                     a.dataset_url or "", data.hf_repo, hw.gpu_label)
        notifier.uploaded(urls)
    notifier.finished({
        "attempts": attempts,
        "final loss": summary.get("final_loss"),
        "hours": summary.get("runtime_h"),
        "uploaded": "\n".join(urls) or "kept on disk",
    })
    finish_screen(screen, s.run_name, summary, urls, cfg["output_dir"])
    return f"last run ✔ {s.run_name} · loss {summary.get('final_loss', '?')}"


def _wait_or_enter(seconds: int) -> None:
    """Back to the start screen after `seconds`, or right away on Enter. cmd-exit leaves."""
    if os.name == "posix" and sys.stdin.isatty():
        import select

        ready, _, _ = select.select([sys.stdin], [], [], seconds)
        if ready and sys.stdin.readline().strip().lower() == ui.EXIT_WORD:
            raise ui.CmdExit()
        return
    ui.read_line("")


def finish_screen(screen: Screen, run_name: str, summary: dict, urls: list[str], output_dir: str) -> None:
    body = [
        f"[bold {ui.HOT}]✔ {escape(run_name)} is done[/]",
        "",
        f"final loss  [bold {ui.HOT}]{summary.get('final_loss', '?')}[/]   eval  {summary.get('final_eval_loss') or '-'}   "
        f"steps  {summary.get('steps', '?')}   hours  {summary.get('runtime_h', '?')}",
        "",
        *([f"📦 {escape(u)}" for u in urls] or ["kept on disk (nothing uploaded)"]),
        f"[{ui.DIM}]{escape(output_dir)}[/]",
        "",
        f"[{ui.DIM}]back to the start screen in 60 s · Enter now · cmd-exit to leave[/]",
    ]
    screen.draw("finished", "All finished", body)
    ui.console.print(screen.prompt_prefix(), end="")
    _wait_or_enter(60)


def error_screen(screen: Screen, title: str, message: str) -> None:
    tail = "\n".join(message.strip().splitlines()[-14:])[-1400:]
    body = [
        f"[bold {ui.ERR}]{escape(tail) or 'unknown error'}[/]",
        "",
        f"[{ui.DIM}]the full log is in /workspace/forge/logs/latest.log[/]",
        f"[{ui.HOT}]Enter[/] back to the start screen  ·  [{ui.DIM}]cmd-exit to leave[/]",
    ]
    screen.draw("", title, body)
    ui.read_line(screen.prompt_prefix())


def leave() -> int:
    try:
        DISMISSED.parent.mkdir(parents=True, exist_ok=True)
        DISMISSED.touch()
    except OSError:
        pass
    pipeline.stop_log()
    ui.console.clear()
    ui.console.print(f"[bold {ui.HOT}]🐺 bye[/] [dim]- `forge setup` brings the setup screen back[/dim]")
    return 0


def run_setup(settings: Settings) -> int:
    try:
        DISMISSED.unlink()
    except OSError:
        pass
    ui.set_yes(False)
    screen = Screen()
    a = Answers.initial(settings)
    gpus = hardware.detect(settings.home).gpu_label
    screen.status = gpus
    while True:
        try:
            collect(screen, a)
            status = execute(screen, a, settings)
            screen.status = f"{gpus} · {status}"
        except ui.CmdExit:
            return leave()
        except Exception as exc:  # noqa: BLE001  (KeyboardInterrupt handled below)
            pipeline.stop_log()
            if isinstance(exc, pipeline.TrainingFailed):
                title, message = f"training failed {exc.attempts}x", exc.error  # Discord already pinged
            else:
                title, message = "something failed", f"{type(exc).__name__}: {exc}"
                notify.Discord(a.webhook).failed("setup run", message)
            screen.status = f"{gpus} · last run ✖ {title}"
            try:
                error_screen(screen, title, message)
            except (ui.CmdExit, KeyboardInterrupt):
                return leave()
        except KeyboardInterrupt:
            pipeline.stop_log()
            screen.status = f"{gpus} · last run stopped"
            try:
                error_screen(screen, "stopped", "you stopped it with Ctrl+C")
            except (ui.CmdExit, KeyboardInterrupt):
                return leave()
        finally:
            pipeline.stop_log()
