"""Fetch models and datasets from whatever link you paste.

Supported links (models and datasets):
  org/name                                   Hugging Face repo id
  https://huggingface.co/[datasets/]org/name[/tree|blob|resolve/<rev>/<path>]
  hf://[datasets/]org/name[@rev][/path]
  https://.../file.jsonl | .parquet | .zip | .tar.gz ...   direct download (resumable)
  https://drive.google.com/file/d/<id>/...                 public Google Drive file
  s3:// gs:// az:// abfs:// oci://                         cloud storage (uses the usual env creds)
  /local/path or ./relative/path                           already on disk
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import struct
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import ui

DATA_EXTS = (".jsonl", ".json", ".parquet", ".csv", ".tsv", ".arrow")
COMPRESSED = (".gz", ".zst", ".zstd")
ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.zst", ".tar.xz", ".tar.bz2")
HF_DATA_PATTERNS = [f"*{e}" for e in DATA_EXTS] + [f"*{e}{c}" for e in DATA_EXTS for c in COMPRESSED] + [
    "**/*" + e for e in DATA_EXTS
] + ["**/*" + e + c for e in DATA_EXTS for c in COMPRESSED] + ["README.md", "*.json", "**/dataset_info.json", "**/state.json"]
MODEL_LIGHT_PATTERNS = ["*.json", "*.model", "tokenizer*", "*.txt", "*.jinja", "*.tiktoken", "*.py", "*.md"]
MODEL_IGNORE = ["*.pth", "*.pt", "original/*", "*.gguf", "*.onnx", "*.msgpack", "*.h5", "*.ot", "consolidated*", "*.mlmodel"]

_HF_URL = re.compile(
    r"^https?://(?:www\.)?(?:huggingface\.co|hf\.co)/(?P<datasets>datasets/)?(?P<repo>[^/\s]+/[^/\s?#]+)"
    r"(?:/(?P<mode>tree|blob|resolve)/(?P<rev>[^/\s]+)(?:/(?P<sub>[^?#]+))?)?"
)
_HF_SCHEME = re.compile(r"^hf://(?P<datasets>datasets/)?(?P<repo>[^/\s]+/[^/\s@]+)(?:@(?P<rev>[^/\s]+))?(?:/(?P<sub>.+))?$")
_BARE_REPO = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")
_GDRIVE = re.compile(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:export=download&)?id=)([\w-]+)")
_REMOTE = ("s3://", "gs://", "gcs://", "az://", "abfs://", "abfss://", "oci://")


@dataclass
class Source:
    kind: str  # hf | url | gdrive | remote | local
    ref: str
    repo_type: str = "model"
    revision: str | None = None
    subpath: str | None = None


def classify(link: str, kind: str) -> Source:
    link = link.strip().strip('"').strip("'")
    if not link:
        raise ValueError("empty link")
    expanded = Path(os.path.expanduser(link))
    if expanded.exists():
        return Source("local", str(expanded.resolve()))
    m = _HF_SCHEME.match(link) or _HF_URL.match(link)
    if m:
        repo_type = "dataset" if m.group("datasets") else "model"
        sub = m.groupdict().get("sub")
        return Source("hf", m.group("repo"), repo_type, m.group("rev"), unquote(sub) if sub else None)
    g = _GDRIVE.search(link)
    if g:
        return Source("gdrive", g.group(1))
    if link.startswith(_REMOTE):
        return Source("remote", link)
    if link.startswith(("http://", "https://")):
        return Source("url", link)
    if _BARE_REPO.match(link):
        return Source("hf", link, kind)
    raise ValueError(f"don't know how to fetch {link!r} (and it is not a local path)")


def slugify(text: str, limit: int = 60) -> str:
    text = re.sub(r"^(https?://|hf://|s3://|gs://)", "", text)
    text = text.replace("/", "__")
    text = re.sub(r"[^A-Za-z0-9_.\-]+", "-", text).strip("-._")
    return (text or "item")[:limit]


# ── generic downloading / extracting ──────────────────────────────────────────────

def _filename_from_response(resp, url: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r"filename\*=(?:UTF-8'')?([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    if m:
        return unquote(m.group(1)).strip()
    name = Path(unquote(urlparse(resp.url or url).path)).name
    return name or "download"


def download_url(url: str, dest_dir: Path, token: str | None = None) -> Path:
    """Stream a URL to disk with a purple progress bar. Resumes partial downloads."""
    import requests
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

    dest_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "axolotl-forge/1.0"}
    if token and ("huggingface.co" in url or "hf.co" in url):
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, stream=True, timeout=60, allow_redirects=True, headers=headers) as head:
        head.raise_for_status()
        name = _filename_from_response(head, url)
        ctype = head.headers.get("content-type", "")
        if "text/html" in ctype and not name.endswith((".html", ".htm")):
            raise RuntimeError(
                f"{url} returned a web page, not a file. Make sure the link is a direct download "
                "(Google Drive files must be shared as 'anyone with the link')."
            )
        total = int(head.headers.get("content-length") or 0)
        accept_ranges = head.headers.get("accept-ranges", "") == "bytes"
        target = dest_dir / name
        part = dest_dir / (name + ".part")
        if target.exists() and total and target.stat().st_size == total:
            ui.ok(f"already downloaded: {target}")
            return target
        resume_from = part.stat().st_size if part.exists() and accept_ranges else 0
        stream = head
        if resume_from:
            head.close()
            stream = requests.get(
                url, stream=True, timeout=60, allow_redirects=True, headers={**headers, "Range": f"bytes={resume_from}-"}
            )
            if stream.status_code != 206:
                resume_from = 0
                stream = requests.get(url, stream=True, timeout=60, allow_redirects=True, headers=headers)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=ui.console,
        ) as progress:
            task = progress.add_task(f"⬇ {name}", total=total or None, completed=resume_from)
            with open(part, "ab" if resume_from else "wb") as fh:
                for chunk in stream.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
                        progress.update(task, advance=len(chunk))
        if stream is not head:
            stream.close()
    part.replace(target)
    return target


def _gdrive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _safe_zip_extract(path: Path, dest: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            resolved = (dest / member).resolve()
            if not str(resolved).startswith(str(dest.resolve())):
                raise RuntimeError(f"refusing to extract unsafe path {member!r} from {path.name}")
        zf.extractall(dest)


def _decompress_single(path: Path) -> Path:
    """foo.jsonl.gz -> foo.jsonl (next to it)."""
    out = path.with_suffix("")
    if out.exists() and out.stat().st_size > 0:
        return out
    ui.info(f"decompressing {path.name}")
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    else:
        import zstandard

        with open(path, "rb") as src, open(out, "wb") as dst:
            zstandard.ZstdDecompressor().copy_stream(src, dst)
    return out


def extract_if_archive(path: Path) -> Path:
    """Extract archives into a sibling folder; decompress single .gz/.zst files. Returns the usable path."""
    name = path.name.lower()
    if path.is_dir():
        return path
    if name.endswith(ARCHIVES):
        dest = path.parent / re.sub(r"(\.tar)?\.(zip|tar|gz|tgz|zst|xz|bz2)$", "", path.name)
        if dest.exists() and any(dest.iterdir()):
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        ui.info(f"extracting {path.name}")
        if name.endswith(".zip"):
            _safe_zip_extract(path, dest)
        elif name.endswith(".tar.zst"):
            import zstandard

            with open(path, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(dest, filter="data")
        else:
            with tarfile.open(path) as tar:
                tar.extractall(dest, filter="data")
        return dest
    if name.endswith(COMPRESSED):
        return _decompress_single(path)
    return path


def _fetch_remote(url: str, dest: Path) -> Path:
    import fsspec

    fs, _, paths = fsspec.get_fs_token_paths(url)
    dest.mkdir(parents=True, exist_ok=True)
    ui.info(f"copying {url}")
    if fs.isdir(paths[0]):
        fs.get(paths[0].rstrip("/") + "/", str(dest), recursive=True)
        return dest
    target = dest / Path(paths[0]).name
    fs.get(paths[0], str(target))
    return target


def _hf_errors():
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:  # pragma: no cover
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError  # type: ignore
    return GatedRepoError, RepositoryNotFoundError


# ── datasets ──────────────────────────────────────────────────────────────────────

@dataclass
class DatasetInfo:
    link: str
    root: Path
    files: list[Path]
    fmt: str  # jsonl | json | parquet | csv | arrow | hf_disk
    hf_repo: str | None = None

    @property
    def ds_type(self) -> str:
        return {"jsonl": "json", "json": "json", "parquet": "parquet", "csv": "csv", "arrow": "arrow"}.get(self.fmt, "json")

    @property
    def total_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.files if f.exists())

    def axolotl_entry(self) -> dict:
        """The path / ds_type / data_files part of an axolotl `datasets:` entry."""
        if self.fmt == "hf_disk":
            return {"path": str(self.root)}
        if len(self.files) == 1:
            return {"path": str(self.files[0]), "ds_type": self.ds_type}
        common = Path(os.path.commonpath([str(f.parent) for f in self.files]))
        return {
            "path": str(common),
            "ds_type": self.ds_type,
            "data_files": [str(f) for f in self.files],
            "split": "train",
        }


def _fmt_of(path: Path) -> str | None:
    name = path.name.lower()
    for ext, fmt in ((".jsonl", "jsonl"), (".parquet", "parquet"), (".csv", "csv"), (".tsv", "csv"), (".arrow", "arrow")):
        if name.endswith(ext):
            return fmt
    if name.endswith(".json") and name not in {"dataset_info.json", "state.json", "dataset_dict.json", "config.json"}:
        return "json"
    return None


def find_data_files(root: Path) -> tuple[list[Path], str]:
    root = extract_if_archive(root)
    if root.is_file():
        fmt = _fmt_of(root)
        if not fmt:
            raise RuntimeError(f"{root.name}: unsupported dataset file (use jsonl / json / parquet / csv / arrow)")
        return [root], fmt
    # datasets.save_to_disk() folder (or DatasetDict with a train split)
    for cand in (root, root / "train"):
        if (cand / "state.json").exists() and (cand / "dataset_info.json").exists():
            return sorted(cand.glob("*.arrow")), "hf_disk"
    for archive in [p for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(ARCHIVES + COMPRESSED)]:
        extract_if_archive(archive)
    found: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts):
            fmt = _fmt_of(p)
            if fmt:
                found.setdefault(fmt, []).append(p)
    if not found:
        raise RuntimeError(f"no jsonl / json / parquet / csv / arrow files found in {root}")
    fmt = max(found, key=lambda k: sum(f.stat().st_size for f in found[k]))
    files = found[fmt]
    rel = {f: f.relative_to(root).as_posix().lower() for f in files}
    train = [f for f in files if re.search(r"(^|[/_\-.])train", rel[f])]
    held_out = [f for f in files if re.search(r"(^|[/_\-.])(test|valid|validation|eval|dev)", rel[f])]
    if train and held_out:
        files = train
    return files, fmt


def fetch_dataset(link: str, settings) -> DatasetInfo:
    src = classify(link, "dataset")
    dest = settings.datasets_dir / slugify(src.ref if src.kind != "local" else Path(src.ref).name)
    hf_repo = None
    if src.kind == "local":
        root = Path(src.ref)
    elif src.kind == "hf":
        from huggingface_hub import hf_hub_download, snapshot_download

        GatedRepoError, RepositoryNotFoundError = _hf_errors()
        hf_repo = src.ref
        if src.repo_type != "dataset":
            ui.warn(f"{src.ref} looks like a model link; trying it as a dataset repo")
        ui.info(f"downloading dataset [hot]{src.ref}[/hot] from the Hugging Face Hub")
        try:
            if src.subpath and _fmt_of(Path(src.subpath)) or (src.subpath or "").endswith(COMPRESSED + ARCHIVES):
                root = Path(
                    hf_hub_download(
                        src.ref, src.subpath, repo_type="dataset", revision=src.revision, local_dir=dest, token=settings.hf_token
                    )
                )
            else:
                patterns = HF_DATA_PATTERNS
                if src.subpath:
                    patterns = [f"{src.subpath.rstrip('/')}/{p}" for p in HF_DATA_PATTERNS]
                snapshot_download(
                    src.ref, repo_type="dataset", revision=src.revision, local_dir=dest,
                    allow_patterns=patterns, token=settings.hf_token,
                )
                root = dest / src.subpath if src.subpath else dest
        except GatedRepoError as exc:
            raise RuntimeError(f"{src.ref} is gated: accept its terms on huggingface.co and set HF_TOKEN") from exc
        except RepositoryNotFoundError as exc:
            raise RuntimeError(f"dataset {src.ref} not found (private? set HF_TOKEN)") from exc
    elif src.kind in ("url", "gdrive"):
        url = _gdrive_url(src.ref) if src.kind == "gdrive" else src.ref
        root = download_url(url, dest, token=settings.hf_token)
    else:
        root = _fetch_remote(src.ref, dest)
    files, fmt = find_data_files(root)
    if fmt == "hf_disk" and files:
        root_dir = files[0].parent
    else:
        root_dir = root if root.is_dir() else root.parent
    info = DatasetInfo(link=link, root=root_dir, files=files, fmt=fmt, hf_repo=hf_repo)
    ui.ok(f"dataset ready: {len(files)} {fmt} file(s), {info.total_bytes / 1024**2:,.1f} MB")
    return info


# ── models ────────────────────────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    link: str
    base_model: str
    local_dir: Path | None
    hf_repo: str | None = None
    revision: str | None = None
    config: dict = field(default_factory=dict)
    param_count: int | None = None
    has_chat_template: bool = False
    weights_downloaded: bool = False

    def _cfg(self, *keys, default=None):
        sources = [self.config, self.config.get("text_config") or {}]
        for src in sources:
            for k in keys:
                if k in src and src[k] is not None:
                    return src[k]
        return default

    @property
    def model_type(self) -> str:
        tc = self.config.get("text_config") or {}
        return str(self.config.get("model_type") or tc.get("model_type") or "unknown")

    @property
    def architectures(self) -> list[str]:
        return list(self.config.get("architectures") or [])

    @property
    def num_experts(self) -> int:
        return int(self._cfg("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts", default=0) or 0)

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1

    @property
    def is_multimodal(self) -> bool:
        return bool(self.config.get("vision_config") or self.config.get("audio_config"))

    @property
    def params_b(self) -> float | None:
        return round(self.param_count / 1e9, 2) if self.param_count else None

    @property
    def quantization(self) -> str | None:
        q = self.config.get("quantization_config")
        if isinstance(q, dict):
            return str(q.get("quant_method") or q.get("quant_type") or "quantized")
        return None

    def to_prompt(self) -> dict:
        return {
            "base_model": self.base_model,
            "model_type": self.model_type,
            "architectures": self.architectures,
            "params_billion": self.params_b,
            "is_moe": self.is_moe,
            "num_experts": self.num_experts or None,
            "experts_per_token": self._cfg("num_experts_per_tok", "moe_topk", "top_k"),
            "hidden_size": self._cfg("hidden_size", "d_model"),
            "num_hidden_layers": self._cfg("num_hidden_layers", "n_layer"),
            "num_attention_heads": self._cfg("num_attention_heads"),
            "num_key_value_heads": self._cfg("num_key_value_heads"),
            "vocab_size": self._cfg("vocab_size"),
            "max_position_embeddings": self._cfg("max_position_embeddings", "max_seq_len", "seq_length"),
            "tie_word_embeddings": self.config.get("tie_word_embeddings"),
            "torch_dtype": self._cfg("torch_dtype", "dtype"),
            "quantization": self.quantization,
            "multimodal": self.is_multimodal,
            "tokenizer_has_chat_template": self.has_chat_template,
        }


def _count_safetensors_params(folder: Path) -> int | None:
    total = 0
    found = False
    for f in folder.rglob("*.safetensors"):
        try:
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
        except (OSError, ValueError, struct.error):
            continue
        found = True
        for key, meta in header.items():
            if key == "__metadata__" or not isinstance(meta, dict):
                continue
            count = 1
            for dim in meta.get("shape", []):
                count *= int(dim)
            total += count
    return total if found else None


def _estimate_params(cfg: dict) -> int | None:
    c = {**(cfg.get("text_config") or {}), **{k: v for k, v in cfg.items() if not isinstance(v, dict)}}
    try:
        h = int(c["hidden_size"])
        layers = int(c["num_hidden_layers"])
        vocab = int(c.get("vocab_size", 32000))
        inter = int(c.get("intermediate_size") or c.get("moe_intermediate_size") or 4 * h)
        heads = int(c.get("num_attention_heads", h // 128 or 1))
        kv = int(c.get("num_key_value_heads", heads))
        head_dim = int(c.get("head_dim") or h // heads)
        experts = int(c.get("num_experts") or c.get("num_local_experts") or c.get("n_routed_experts") or 1)
        moe_inter = int(c.get("moe_intermediate_size") or inter)
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None
    attn = h * heads * head_dim * 2 + h * kv * head_dim * 2
    mlp = 3 * h * (moe_inter * experts if experts > 1 else inter)
    embed = vocab * h * (1 if c.get("tie_word_embeddings") else 2)
    return int(layers * (attn + mlp) + embed)


def _read_model_folder(folder: Path, info: ModelInfo) -> None:
    cfg_path = folder / "config.json"
    if cfg_path.exists():
        info.config = json.loads(cfg_path.read_text(encoding="utf-8"))
    tok_cfg = folder / "tokenizer_config.json"
    has_tpl = (folder / "chat_template.jinja").exists() or (folder / "chat_template.json").exists()
    if tok_cfg.exists():
        try:
            has_tpl = has_tpl or bool(json.loads(tok_cfg.read_text(encoding="utf-8")).get("chat_template"))
        except ValueError:
            pass
    info.has_chat_template = has_tpl
    info.param_count = _count_safetensors_params(folder)
    info.weights_downloaded = info.param_count is not None


def _find_model_dir(root: Path) -> Path:
    if (root / "config.json").exists():
        return root
    for cfg in sorted(root.rglob("config.json"), key=lambda p: len(p.parts)):
        return cfg.parent
    raise RuntimeError(f"no config.json found under {root} - is this a Hugging Face model folder?")


def fetch_model(link: str, settings, light: bool | None = None) -> ModelInfo:
    light = settings.skip_model_download if light is None else light
    src = classify(link, "model")
    if src.kind == "hf":
        from huggingface_hub import HfApi, snapshot_download

        GatedRepoError, RepositoryNotFoundError = _hf_errors()
        if src.repo_type == "dataset":
            raise RuntimeError(f"{link} is a dataset link, not a model")
        what = "config + tokenizer" if light else "weights"
        ui.info(f"downloading model {what} for [hot]{src.ref}[/hot] (cached in HF_HOME)")
        try:
            folder = Path(
                snapshot_download(
                    src.ref, revision=src.revision, token=settings.hf_token,
                    allow_patterns=MODEL_LIGHT_PATTERNS if light else None, ignore_patterns=MODEL_IGNORE,
                )
            )
        except GatedRepoError as exc:
            raise RuntimeError(f"{src.ref} is gated: accept its licence on huggingface.co and set HF_TOKEN") from exc
        except RepositoryNotFoundError as exc:
            raise RuntimeError(f"model {src.ref} not found (typo? private? set HF_TOKEN)") from exc
        info = ModelInfo(link=link, base_model=src.ref, local_dir=folder, hf_repo=src.ref, revision=src.revision)
        _read_model_folder(folder, info)
        if info.param_count is None:
            try:
                st = HfApi(token=settings.hf_token).model_info(src.ref, revision=src.revision).safetensors
                info.param_count = int(st.total) if st and getattr(st, "total", None) else None
            except Exception:
                info.param_count = None
    else:
        if src.kind == "local":
            root = extract_if_archive(Path(src.ref))
        else:
            dest = settings.models_dir / slugify(src.ref)
            if src.kind == "remote":
                got = _fetch_remote(src.ref, dest)
            else:
                url = _gdrive_url(src.ref) if src.kind == "gdrive" else src.ref
                got = download_url(url, dest, token=settings.hf_token)
            root = extract_if_archive(got)
        folder = _find_model_dir(root)
        info = ModelInfo(link=link, base_model=str(folder), local_dir=folder)
        _read_model_folder(folder, info)
    if info.param_count is None:
        info.param_count = _estimate_params(info.config)
    label = f"{info.params_b}B params" if info.params_b else "size unknown"
    ui.ok(f"model ready: {info.base_model} · {info.model_type} · {label}{' · MoE' if info.is_moe else ''}")
    return info
