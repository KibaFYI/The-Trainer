"""Read N random "sectors" of a dataset (jsonl / json / parquet / csv / arrow) and profile it.

The dataset is split into N equal slices (by bytes for jsonl, by rows for columnar
formats) and one record is read from a random spot inside each slice, so even a
100 GB file is sampled in milliseconds and the samples are spread across the data.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fetch import DatasetInfo

CHAT_KEYS = ("messages", "conversations", "conversation", "chat", "dialogue", "dialog", "turns")


@dataclass
class Sample:
    where: str
    record: dict


# ── readers ───────────────────────────────────────────────────────────────────────

class _JsonlReader:
    def __init__(self, files: list[Path]):
        self.files = files
        self.sizes = [f.stat().st_size for f in files]
        self.total = sum(self.sizes) or 1
        self.line_bytes: list[int] = []

    def _locate(self, frac: float) -> tuple[int, int]:
        pos = int(frac * self.total)
        for i, size in enumerate(self.sizes):
            if pos < size or i == len(self.sizes) - 1:
                return i, min(pos, max(size - 1, 0))
            pos -= size
        return len(self.files) - 1, 0

    def read(self, fracs: list[float]) -> list[Sample]:
        out: list[Sample] = []
        for frac in fracs:
            fi, off = self._locate(frac)
            path = self.files[fi]
            with open(path, "rb") as fh:
                fh.seek(off)
                if off > 0:
                    fh.readline()  # drop the partial line we landed in
                for _attempt in range(8):
                    line = fh.readline()
                    if not line:
                        fh.seek(0)
                        line = fh.readline()
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    self.line_bytes.append(len(line))
                    out.append(Sample(f"{path.name} @ {100 * frac:.1f}%", rec if isinstance(rec, dict) else {"value": rec}))
                    break
        return out

    def estimate_rows(self) -> int | None:
        if not self.line_bytes:
            return None
        return int(self.total / statistics.mean(self.line_bytes))


class _ParquetReader:
    def __init__(self, files: list[Path]):
        import pyarrow.parquet as pq

        self.pf = [pq.ParquetFile(f) for f in files]
        self.files = files
        self.rows = [p.metadata.num_rows for p in self.pf]
        self.total = sum(self.rows) or 1

    def read(self, fracs: list[float]) -> list[Sample]:
        wanted: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for frac in fracs:
            row = min(int(frac * self.total), self.total - 1)
            for fi, n in enumerate(self.rows):
                if row < n:
                    break
                row -= n
            pf = self.pf[fi]
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                if row < n:
                    break
                row -= n
            wanted.setdefault((fi, rg), []).append((row, frac))
        out: list[Sample] = []
        for (fi, rg), rows in wanted.items():
            table = self.pf[fi].read_row_group(rg)
            for row, frac in rows:
                rec = table.slice(min(row, table.num_rows - 1), 1).to_pylist()[0]
                out.append(Sample(f"{self.files[fi].name} @ {100 * frac:.1f}%", rec))
        return out

    def estimate_rows(self) -> int:
        return self.total


class _TableReader:
    """Formats we load whole (memory-mapped where possible): json arrays, csv, arrow."""

    def __init__(self, files: list[Path], fmt: str):
        import pyarrow as pa

        tables = []
        for f in files:
            tables.append(self._load(f, fmt, pa))
        self.table = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
        self.name = files[0].name if len(files) == 1 else f"{len(files)} files"

    @staticmethod
    def _load(f: Path, fmt: str, pa):
        if fmt == "csv":
            import pyarrow.csv as pcsv

            opts = pcsv.ParseOptions(delimiter="\t") if f.suffix.lower() == ".tsv" else None
            return pcsv.read_csv(f, parse_options=opts)
        if fmt in ("arrow", "hf_disk"):
            src = pa.memory_map(str(f))
            try:
                return pa.ipc.open_stream(src).read_all()
            except pa.ArrowInvalid:
                return pa.ipc.open_file(pa.memory_map(str(f))).read_all()
        # json: either a JSON array or JSONL saved as .json
        text_start = f.open("rb").read(64).lstrip()
        if text_start.startswith(b"["):
            data = json.loads(f.read_text(encoding="utf-8"))
            return pa.Table.from_pylist([r if isinstance(r, dict) else {"value": r} for r in data])
        if text_start.startswith(b"{") and b"\n" not in f.open("rb").read(1 << 20).strip():
            data = json.loads(f.read_text(encoding="utf-8"))
            for v in data.values():
                if isinstance(v, list):
                    return pa.Table.from_pylist([r if isinstance(r, dict) else {"value": r} for r in v])
            return pa.Table.from_pylist([data])
        rows = [json.loads(line) for line in f.open(encoding="utf-8") if line.strip()]
        return pa.Table.from_pylist(rows)

    def read(self, fracs: list[float]) -> list[Sample]:
        n = self.table.num_rows
        out = []
        for frac in fracs:
            idx = min(int(frac * n), n - 1)
            out.append(Sample(f"{self.name} row {idx}", self.table.slice(idx, 1).to_pylist()[0]))
        return out

    def estimate_rows(self) -> int:
        return self.table.num_rows


def _reader(info: DatasetInfo):
    if info.fmt == "jsonl":
        return _JsonlReader(info.files)
    if info.fmt == "json":
        # .json files are often JSONL in disguise; sniff the first byte
        first = info.files[0].open("rb").read(64).lstrip()[:1]
        if first == b"{":
            head = info.files[0].open("rb").read(1 << 16)
            if head.count(b"\n") > 1:
                return _JsonlReader(info.files)
        return _TableReader(info.files, "json")
    if info.fmt == "parquet":
        return _ParquetReader(info.files)
    return _TableReader(info.files, info.fmt)


def _sector_fracs(n: int, rng: random.Random) -> list[float]:
    return [(i + rng.random()) / n for i in range(n)]


# ── profiling ─────────────────────────────────────────────────────────────────────

def _truncate(value: Any, max_chars: int, depth: int = 0) -> Any:
    if isinstance(value, str):
        if len(value) > max_chars:
            return value[:max_chars] + f" …[+{len(value) - max_chars} chars]"
        return value
    if isinstance(value, bytes):
        return f"<binary {len(value)} bytes>"
    if isinstance(value, dict):
        if "bytes" in value and isinstance(value.get("bytes"), (bytes, bytearray)):
            return f"<image/audio {len(value['bytes'])} bytes>"
        return {k: _truncate(v, max_chars, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_truncate(v, max_chars, depth + 1) for v in value[:16]]
        if len(value) > 16:
            items.append(f"…[+{len(value) - 16} more items]")
        return items
    return value


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def _chat_key(rec: dict) -> str | None:
    for k in CHAT_KEYS:
        v = rec.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return k
    for k, v in rec.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and ({"role", "content"} <= set(v[0]) or {"from", "value"} <= set(v[0])):
            return k
    return None


def detect_format(records: list[dict]) -> dict:
    """Guess the dataset shape and the matching axolotl `datasets:` entry."""
    if not records:
        return {"format": "unknown", "axolotl": {}}
    counts = Counter(k for r in records for k in r.keys())
    common = {k for k, c in counts.items() if c >= 0.8 * len(records)}
    chat = _chat_key(records[0])
    if chat:
        turns = [t for r in records for t in (r.get(chat) or []) if isinstance(t, dict)]
        inner = Counter(k for t in turns for k in t.keys())
        role_key = "role" if "role" in inner else ("from" if "from" in inner else None)
        content_key = "content" if "content" in inner else ("value" if "value" in inner else ("text" if "text" in inner else None))
        roles = sorted({str(t.get(role_key)) for t in turns if role_key and t.get(role_key) is not None})
        entry: dict[str, Any] = {"type": "chat_template", "field_messages": chat}
        if (role_key, content_key) != ("role", "content") and role_key and content_key:
            entry["message_property_mappings"] = {"role": role_key, "content": content_key}
        std = {"system", "user", "assistant", "tool"}
        if roles and not set(roles) <= std:
            mapping: dict[str, list[str]] = {}
            for r in roles:
                low = r.lower()
                if low in ("human", "user", "prompter", "question", "customer"):
                    mapping.setdefault("user", []).append(r)
                elif low in ("gpt", "assistant", "bot", "model", "chatgpt", "answer", "agent"):
                    mapping.setdefault("assistant", []).append(r)
                elif low in ("system", "sys"):
                    mapping.setdefault("system", []).append(r)
                elif low in ("tool", "function", "observation", "ipython"):
                    mapping.setdefault("tool", []).append(r)
            if mapping:
                entry["roles"] = mapping
        fmt = "chat (openai messages)" if (role_key, content_key) == ("role", "content") else "chat (sharegpt-style)"
        if {"chosen", "rejected"} <= common:
            return {"format": "preference (chat)", "rl": "dpo", "roles_seen": roles, "axolotl": {"type": "chat_template.default", "field_messages": chat}}
        return {"format": fmt, "roles_seen": roles, "axolotl": entry}
    if {"chosen", "rejected"} <= common:
        prompt = "prompt" if "prompt" in common else ("question" if "question" in common else None)
        return {"format": "preference (dpo)", "rl": "dpo", "axolotl": {"type": "chatml.prompt_pairs" if prompt == "prompt" else "chatml.intel"}}
    if {"prompt", "completion", "label"} <= common:
        return {"format": "kto (prompt/completion/label)", "rl": "kto", "axolotl": {"type": "llama3.ultra"}}
    if {"instruction", "output"} <= common:
        return {"format": "alpaca (instruction/input/output)", "axolotl": {"type": "alpaca"}}
    for a, b in (("prompt", "completion"), ("prompt", "response"), ("question", "answer"), ("input", "output"), ("query", "response"), ("problem", "solution")):
        if {a, b} <= common:
            entry = {
                "type": {
                    "field_instruction": a,
                    "field_output": b,
                    "format": "{instruction}\n",
                    "no_input_format": "{instruction}\n",
                }
            }
            if "system" in common:
                entry["type"]["field_system"] = "system"
            return {"format": f"single-turn ({a}/{b})", "axolotl": entry}
    if "input_ids" in common:
        return {"format": "pre-tokenized (input_ids)", "axolotl": {"type": None}}
    text_keys = [k for k in ("text", "content", "body", "document") if k in common]
    if text_keys:
        return {"format": "raw text (completion / continued pretraining)", "axolotl": {"type": "completion", "field": text_keys[0]}}
    return {"format": "unknown", "fields": sorted(common), "axolotl": {}}


@dataclass
class DatasetProfile:
    fmt: str
    files: list[str]
    total_mb: float
    est_rows: int | None
    fields: dict[str, float]
    detected: dict
    samples: list[Sample]
    lengths: dict
    token_counter: str
    profile_records: int = 0
    extra: dict = field(default_factory=dict)

    def to_prompt(self, max_chars: int = 1500) -> dict:
        return {
            "file_format": self.fmt,
            "files": self.files[:20] + ([f"... +{len(self.files) - 20} more"] if len(self.files) > 20 else []),
            "total_size_mb": self.total_mb,
            "estimated_rows": self.est_rows,
            "top_level_fields_presence": self.fields,
            "detected": self.detected,
            "token_length_stats": self.lengths,
            "token_counter": self.token_counter,
            "random_sectors": [{"where": s.where, "record": _truncate(s.record, max_chars)} for s in self.samples],
        }


def _token_lengths(records: list[dict], detected: dict, model_dir: Path | None) -> tuple[list[int], str]:
    texts = ["\n".join(_strings(r)) for r in records]
    if model_dir:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(model_dir))
            lengths: list[int] = []
            chat_key = detected.get("axolotl", {}).get("field_messages")
            mapping = detected.get("axolotl", {}).get("message_property_mappings") or {}
            for rec, text in zip(records, texts):
                n = None
                if chat_key and getattr(tok, "chat_template", None):
                    try:
                        msgs = [
                            {"role": _norm_role(str(t.get(mapping.get("role", "role"), "user"))), "content": str(t.get(mapping.get("content", "content"), ""))}
                            for t in rec.get(chat_key) or []
                            if isinstance(t, dict)
                        ]
                        n = len(tok.apply_chat_template(msgs, tokenize=True))
                    except Exception:
                        n = None
                if n is None:
                    n = len(tok(text, add_special_tokens=True)["input_ids"])
                lengths.append(n)
            return lengths, "model tokenizer"
        except Exception:
            pass
    return [math.ceil(len(t) / 3.6) for t in texts], "chars / 3.6 (approx)"


def _norm_role(role: str) -> str:
    low = role.lower()
    if low in ("human", "user", "prompter", "question", "customer"):
        return "user"
    if low in ("gpt", "assistant", "bot", "model", "chatgpt", "answer", "agent"):
        return "assistant"
    if low in ("system", "sys"):
        return "system"
    return "user"


def _pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def profile(
    info: DatasetInfo,
    n_sectors: int = 5,
    profile_size: int = 256,
    model_dir: Path | None = None,
    seed: int | None = None,
) -> DatasetProfile:
    rng = random.Random(seed)
    reader = _reader(info)
    samples = reader.read(_sector_fracs(n_sectors, rng))
    stats_records = [s.record for s in reader.read(sorted(rng.random() for _ in range(profile_size)))]
    stats_records = stats_records or [s.record for s in samples]
    detected = detect_format(stats_records)
    counts = Counter(k for r in stats_records for k in r.keys())
    fields = {k: round(100 * c / max(len(stats_records), 1), 1) for k, c in counts.most_common(40)}
    lengths, counter = _token_lengths(stats_records, detected, model_dir)
    length_stats = {
        "p50": _pct(lengths, 0.5),
        "p90": _pct(lengths, 0.9),
        "p95": _pct(lengths, 0.95),
        "p99": _pct(lengths, 0.99),
        "max": max(lengths) if lengths else 0,
        "mean": int(statistics.mean(lengths)) if lengths else 0,
    }
    est_rows = reader.estimate_rows()
    if est_rows and lengths:
        length_stats["est_total_tokens_millions"] = round(est_rows * statistics.mean(lengths) / 1e6, 2)
    return DatasetProfile(
        fmt=info.fmt,
        files=[str(f) for f in info.files],
        total_mb=round(info.total_bytes / 1024**2, 2),
        est_rows=est_rows,
        fields=fields,
        detected=detected,
        samples=samples,
        lengths=length_stats,
        token_counter=counter,
        profile_records=len(stats_records),
    )
