"""The wolfgon. The art file is used byte-for-byte: lines are only shifted as a whole block
(same left padding on every line) and coloured per character - never re-spaced."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rich.text import Text

ART_PATH = Path(__file__).parent / "assets" / "wolfgon.txt"

# character -> purple shade (colour only; characters and spacing are untouched)
_HOT, _DEEP, _MID = "bold #EA80FC", "#7C4DFF", "#B388FF"


@lru_cache(maxsize=1)
def lines() -> tuple[str, ...]:
    raw = ART_PATH.read_bytes().decode("utf-8").replace("\r\n", "\n")
    out = raw.split("\n")
    while out and not out[-1].strip():
        out.pop()
    return tuple(out)


def width() -> int:
    return max(len(line.rstrip()) for line in lines())


def height() -> int:
    return len(lines())


def _style(ch: str) -> str:
    if ch in "@%#*+":
        return _HOT
    if ch in "=-":
        return _DEEP
    return _MID


def render_line(line: str, pad: int) -> Text:
    text = Text(" " * pad, no_wrap=True, overflow="crop", end="\n")
    run, style = "", None
    for ch in line:
        s = _style(ch) if ch != " " else style
        if s != style and run:
            text.append(run, style=style)
            run = ""
        style = s
        run += ch
    if run:
        text.append(run, style=style)
    return text


def print_centered(console, term_width: int | None = None) -> None:
    w = term_width or console.width
    pad = max(0, (w - width()) // 2)
    for line in lines():
        console.print(render_line(line, pad), no_wrap=True, crop=True, highlight=False, markup=False)
