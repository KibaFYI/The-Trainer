"""Purple terminal UI: themed rich console, prompts, and a recolouring process runner."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


PURPLE = os.environ.get("FORGE_PURPLE", "#B388FF")
DEEP = os.environ.get("FORGE_PURPLE_DEEP", "#7C4DFF")
SOFT = "#D1B3FF"
DIM = "#9575CD"
HOT = "#EA80FC"
ERR = "#FF5CCB"
INK = "#1A1028"

_THEME = Theme(
    {
        "info": PURPLE,
        "ok": f"bold {HOT}",
        "warn": f"bold {SOFT}",
        "err": f"bold {ERR}",
        "dim": DIM,
        "accent": f"bold {DEEP}",
        "hot": HOT,
        "soft": SOFT,
        "key": f"bold {HOT}",
        # rich internals -> purple
        "repr.number": SOFT,
        "repr.str": SOFT,
        "repr.bool_true": HOT,
        "repr.bool_false": DIM,
        "repr.none": DIM,
        "repr.path": SOFT,
        "repr.filename": HOT,
        "repr.url": f"underline {SOFT}",
        "rule.line": DEEP,
        "rule.text": f"bold {HOT}",
        "table.header": f"bold {HOT}",
        "table.title": f"bold {DEEP}",
        "prompt": PURPLE,
        "prompt.choices": HOT,
        "prompt.default": DIM,
        "prompt.invalid": ERR,
        "bar.back": "#3A2A55",
        "bar.complete": HOT,
        "bar.finished": PURPLE,
        "bar.pulse": SOFT,
        "progress.description": PURPLE,
        "progress.percentage": SOFT,
        "progress.download": SOFT,
        "progress.data.speed": DIM,
        "progress.elapsed": DIM,
        "progress.remaining": DIM,
        "progress.spinner": HOT,
        "status.spinner": HOT,
    }
)

console = Console(theme=_THEME, style=PURPLE, highlight=False, soft_wrap=False)

_YES = os.environ.get("FORGE_YES", "").lower() in ("1", "true", "yes")


def set_yes(value: bool) -> None:
    global _YES
    _YES = value or _YES


EXIT_WORD = "cmd-exit"


class CmdExit(Exception):
    """The user typed cmd-exit at a prompt."""


def interactive() -> bool:
    """True when we can ask the user questions."""
    if os.environ.get("FORGE_FORCE_INTERACTIVE") == "1":  # scripted tests feed answers through a pipe
        return True
    return not _YES and sys.stdin.isatty() and sys.stdout.isatty()


def read_line(prompt: str = "", password: bool = False) -> str:
    """One line of input. Typing cmd-exit anywhere raises CmdExit."""
    try:
        value = console.input(prompt, password=password and sys.stdin.isatty())
    except EOFError:
        raise CmdExit() from None
    if value.strip().lower() == EXIT_WORD:
        raise CmdExit()
    return value


# ── output helpers ────────────────────────────────────────────────────────────────

def banner(subtitle: str | None = None) -> None:
    """The wolfgon, centered, with the name under it."""
    from rich.align import Align

    from . import art

    if console.width >= art.width() + 2:
        art.print_centered(console)
    console.print(Align.center(Text("a x o l o t l · f o r g e", style=f"bold {PURPLE}")))
    if subtitle:
        console.print(Align.center(Text(subtitle, style=DIM)))
    console.print()


def step(title: str) -> None:
    console.rule(Text(f" {title} ", style=f"bold {HOT}"), style=DEEP)


def info(msg: str) -> None:
    console.print(f"[accent]•[/accent] {msg}")


def ok(msg: str) -> None:
    console.print(f"[ok]✔[/ok] {msg}")


def warn(msg: str) -> None:
    console.print(f"[warn]![/warn] [soft]{msg}[/soft]")


def err(msg: str) -> None:
    console.print(f"[err]✖ {msg}[/err]")


def kv_table(title: str, rows: Sequence[tuple[str, Any]]) -> None:
    table = Table(title=title, show_header=False, border_style=DEEP, title_justify="left")
    table.add_column(style=f"bold {HOT}", no_wrap=True)
    table.add_column(style=PURPLE)
    for key, value in rows:
        table.add_row(str(key), str(value))
    console.print(table)


def table(title: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    t = Table(title=title, border_style=DEEP, header_style=f"bold {HOT}", title_justify="left")
    for col in columns:
        t.add_column(col, style=PURPLE, overflow="fold")
    for row in rows:
        t.add_row(*[str(c) for c in row])
    console.print(t)


def panel(body: str | Text, title: str = "") -> None:
    console.print(Panel(body, title=title, border_style=DEEP, title_align="left", style=PURPLE))


_YAML_KEY = re.compile(r"^(\s*-?\s*)([A-Za-z0-9_.\-]+)(:)(.*)$")


def show_yaml(text: str, title: str = "config") -> None:
    """Tiny purple YAML highlighter (no pygments needed)."""
    out = Text()
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out.append(line + "\n", style=f"italic {DIM}")
            continue
        m = _YAML_KEY.match(line)
        if m:
            out.append(m.group(1), style=DEEP)
            out.append(m.group(2), style=f"bold {HOT}")
            out.append(m.group(3), style=DEEP)
            out.append(m.group(4) + "\n", style=SOFT)
        else:
            out.append(line + "\n", style=SOFT)
    console.print(Panel(out, title=title, border_style=DEEP, title_align="left"))


# ── prompts ───────────────────────────────────────────────────────────────────────

class NeedsInput(RuntimeError):
    """Raised when a value is required but we cannot ask (non-interactive)."""


def ask(question: str, default: str | None = None, env_hint: str | None = None, password: bool = False) -> str:
    if not interactive():
        if default is not None:
            return default
        hint = f" (set {env_hint})" if env_hint else ""
        raise NeedsInput(f"missing value for: {question}{hint}")
    shown = ""
    if default not in (None, ""):
        shown = " [dim](Enter keeps the saved one)[/dim]" if password else f" [dim]({default})[/dim]"
    while True:
        answer = read_line(f"[bold {HOT}]?[/] [bold {PURPLE}]{question}[/]{shown} ", password=password).strip()
        if answer:
            return answer
        if default is not None:
            return default


def confirm(question: str, default: bool = True) -> bool:
    if not interactive():
        return default
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = read_line(f"[bold {HOT}]?[/] [bold {PURPLE}]{question}[/] [dim]{hint}[/dim] ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False


def choose(
    question: str,
    options: Sequence[tuple[str, str]],
    default: int = 0,
    allow_custom: bool = False,
    custom_label: str = "type something else",
) -> str:
    """Numbered picker. options = [(value, label)]. Returns the chosen value."""
    if not options and not allow_custom:
        raise ValueError("no options to choose from")
    if not interactive():
        if options:
            return options[min(default, len(options) - 1)][0]
        raise NeedsInput(f"missing value for: {question}")
    console.print(f"[bold {HOT}]?[/] [bold {PURPLE}]{question}[/]")
    for i, (_, label) in enumerate(options, start=1):
        marker = "❯" if i - 1 == default else " "
        console.print(f"  [hot]{marker}[/hot] [accent]{i:>2}[/accent]  {label}")
    custom_idx = len(options) + 1
    if allow_custom:
        console.print(f"    [accent]{custom_idx:>2}[/accent]  [dim]{custom_label}[/dim]")
    while True:
        raw = read_line(f"  [dim]pick a number ({default + 1})[/dim] " if options else "  ").strip()
        if not raw and options:
            return options[default][0]
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(options):
                return options[n - 1][0]
            if allow_custom and n == custom_idx:
                return ask(custom_label)
        elif allow_custom and raw:
            return raw
        err("not a valid choice")


# ── recolouring process runner ──────────────────────────────────────────────────

_SGR = re.compile(rb"\x1b\[([0-9;:]*)m")
_STRICT = os.environ.get("FORGE_PURPLE_STRICT", "1") != "0"


def _shade_for_rgb(r: int, g: int, b: int) -> tuple[int, int, int]:
    """Map any colour to a purple shade, keeping 'red-ish' as hot pink for errors."""
    if r > 150 and r > g * 1.6 and r > b * 1.3:
        return _rgb(ERR)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum > 190:
        return _rgb(SOFT)
    if lum > 110:
        return _rgb(PURPLE)
    if lum > 50:
        return _rgb(HOT)
    return _rgb(DEEP)


_ANSI16 = {
    0: (40, 40, 40), 1: (205, 49, 49), 2: (13, 188, 121), 3: (229, 229, 16),
    4: (36, 114, 200), 5: (188, 63, 188), 6: (17, 168, 205), 7: (229, 229, 229),
    8: (102, 102, 102), 9: (241, 76, 76), 10: (35, 209, 139), 11: (245, 245, 67),
    12: (59, 142, 234), 13: (214, 112, 214), 14: (41, 184, 219), 15: (255, 255, 255),
}


def _xterm256(idx: int) -> tuple[int, int, int]:
    if idx < 16:
        return _ANSI16[idx]
    if idx < 232:
        idx -= 16
        steps = [0, 95, 135, 175, 215, 255]
        return steps[idx // 36], steps[(idx // 6) % 6], steps[idx % 6]
    level = 8 + (idx - 232) * 10
    return level, level, level


def _fg(rgb: tuple[int, int, int]) -> list[str]:
    return ["38", "2", str(rgb[0]), str(rgb[1]), str(rgb[2])]


def _rewrite_sgr(params: str) -> str:
    primary = _fg(_rgb(PURPLE))
    parts = re.split(r"[;:]", params) if params else ["0"]
    out: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        n = int(p) if p.isdigit() else (0 if p == "" else None)
        if n == 0:
            out += ["0", *primary]
        elif n == 39:
            out += primary
        elif not _STRICT:
            out.append(p)
        elif n is not None and (30 <= n <= 37 or 90 <= n <= 97):
            base = n - 30 if n < 90 else n - 90 + 8
            out += _fg(_shade_for_rgb(*_ANSI16[base]))
        elif n == 38 and i + 1 < len(parts) and parts[i + 1] == "5" and i + 2 < len(parts):
            idx = int(parts[i + 2]) if parts[i + 2].isdigit() else 0
            out += _fg(_shade_for_rgb(*_xterm256(idx)))
            i += 2
        elif n == 38 and i + 1 < len(parts) and parts[i + 1] == "2" and i + 4 < len(parts):
            try:
                r, g, b = (int(x) for x in parts[i + 2 : i + 5])
                out += _fg(_shade_for_rgb(r, g, b))
            except ValueError:
                out += primary
            i += 4
        else:
            out.append(p)
        i += 1
    return "\x1b[" + ";".join(out) + "m"


class Recolorizer:
    """Streaming SGR rewriter; safe across chunk boundaries."""

    def __init__(self) -> None:
        self._tail = b""
        self.started = False

    def feed(self, data: bytes) -> bytes:
        data = self._tail + data
        self._tail = b""
        esc = data.rfind(b"\x1b")
        if esc != -1 and esc >= len(data) - 32 and not re.match(rb"\x1b\[[0-9;:]*[A-Za-z]|\x1b[^\[]", data[esc:]):
            self._tail = data[esc:]
            data = data[:esc]
        out = _SGR.sub(lambda m: _rewrite_sgr(m.group(1).decode()).encode(), data)
        if not self.started:
            self.started = True
            out = ("\x1b[" + ";".join(_fg(_rgb(PURPLE))) + "m").encode() + out
        return out

    def flush(self) -> bytes:
        out, self._tail = self._tail, b""
        return out


USER_CANCELLED = 130  # run_purple's return code when the user pressed Ctrl+C


def run_purple(
    cmd: Sequence[str],
    log_path: str | Path | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> int:
    """Run a command in a pseudo-terminal, recolour its output purple, tee to a log."""
    env = dict(env or os.environ)
    log = open(log_path, "ab") if log_path else None
    try:
        if os.name != "posix":
            proc = subprocess.Popen(list(cmd), env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            rc = Recolorizer()
            assert proc.stdout is not None
            for chunk in iter(lambda: proc.stdout.read1(65536), b""):  # type: ignore[attr-defined]
                out = rc.feed(chunk)
                sys.stdout.buffer.write(out)
                sys.stdout.flush()
                if log:
                    log.write(out)
            return proc.wait()

        import fcntl
        import pty
        import struct
        import termios

        pid, fd = pty.fork()
        if pid == 0:  # child
            try:
                if cwd:
                    os.chdir(cwd)
                os.execvpe(cmd[0], list(cmd), env)
            except Exception as exc:  # pragma: no cover
                sys.stderr.write(f"forge: could not exec {cmd[0]}: {exc}\n")
            finally:
                os._exit(127)

        try:  # give the child our terminal size so progress bars fit
            size = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        except OSError:
            size = struct.pack("HHHH", 50, 200, 0, 0)
        try:
            fcntl.ioctl(fd, termios.TIOCSWINSZ, size)
        except OSError:
            pass

        rc = Recolorizer()
        interrupted = False
        while True:
            try:
                data = os.read(fd, 65536)
            except KeyboardInterrupt:
                interrupted = True
                os.kill(pid, signal.SIGINT)
                continue
            except OSError:
                break
            if not data:
                break
            out = rc.feed(data)
            sys.stdout.buffer.write(out)
            sys.stdout.flush()
            if log:
                log.write(out)
                log.flush()
        tail = rc.flush()
        if tail:
            sys.stdout.buffer.write(tail)
        while True:
            try:
                _, status = os.waitpid(pid, 0)
                break
            except KeyboardInterrupt:
                interrupted = True
                os.kill(pid, signal.SIGINT)
        os.close(fd)
        sys.stdout.write("\x1b[0m" + "\x1b[" + ";".join(_fg(_rgb(PURPLE))) + "m")
        sys.stdout.flush()
        code = os.waitstatus_to_exitcode(status)
        return USER_CANCELLED if interrupted and code != 0 else code
    finally:
        if log:
            log.close()
