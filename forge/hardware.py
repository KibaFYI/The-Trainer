"""Detect GPUs / CPU / RAM / disk so configs fit the machine they run on."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path


@dataclass
class GPU:
    index: int
    name: str
    vram_gb: float
    free_gb: float
    compute_capability: str

    @property
    def cc(self) -> float:
        try:
            return float(self.compute_capability)
        except ValueError:
            return 0.0

    @property
    def arch(self) -> str:
        cc = self.cc
        if cc >= 12.0:
            return "blackwell-consumer (sm_120)"
        if cc >= 10.0:
            return "blackwell (sm_100)"
        if cc >= 9.0:
            return "hopper (sm_90)"
        if cc >= 8.9:
            return "ada (sm_89)"
        if cc >= 8.0:
            return "ampere (sm_80/86)"
        if cc >= 7.5:
            return "turing (sm_75)"
        if cc >= 7.0:
            return "volta (sm_70)"
        return "unknown"


@dataclass
class Hardware:
    gpus: list[GPU] = field(default_factory=list)
    nvlink: bool = False
    driver: str | None = None
    cuda: str | None = None
    cpu_count: int = 1
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    torch: str | None = None
    axolotl: str | None = None

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def min_vram_gb(self) -> float:
        return min((g.vram_gb for g in self.gpus), default=0.0)

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    @property
    def bf16(self) -> bool:
        return bool(self.gpus) and all(g.cc >= 8.0 for g in self.gpus)

    @property
    def min_cc(self) -> float:
        return min((g.cc for g in self.gpus), default=0.0)

    @property
    def gpu_label(self) -> str:
        if not self.gpus:
            return "no GPU detected"
        groups: dict[tuple[str, float], int] = {}
        for g in self.gpus:
            groups[(g.name, g.vram_gb)] = groups.get((g.name, g.vram_gb), 0) + 1
        return ", ".join(f"{n}x {name} ({vram:.0f} GB)" for (name, vram), n in groups.items())

    def to_prompt(self) -> dict:
        d = asdict(self)
        d["gpu_count"] = self.gpu_count
        d["min_vram_gb"] = round(self.min_vram_gb, 1)
        d["total_vram_gb"] = round(self.total_vram_gb, 1)
        d["bf16_supported"] = self.bf16
        d["gpu_arch"] = sorted({g.arch for g in self.gpus})
        return d


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _fake_gpus(spec: str) -> list[GPU]:
    """FORGE_FAKE_GPUS="2xH100:80:9.0" — for testing configs off-box."""
    m = re.match(r"(\d+)x([^:]+):(\d+(?:\.\d+)?):(\d+\.\d+)", spec.strip())
    if not m:
        return []
    n, name, vram, cc = int(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
    return [GPU(i, name, vram, vram, cc) for i in range(n)]


def _cpu_count() -> int:
    count = os.cpu_count() or 1
    try:
        count = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return count


def _ram_gb() -> float:
    total = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) / 1024 / 1024
                break
    except OSError:
        try:
            import psutil  # type: ignore

            total = psutil.virtual_memory().total / 1024**3
        except Exception:
            total = 0.0
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            total = min(total, int(limit) / 1024**3) if total else int(limit) / 1024**3
    except (OSError, ValueError):
        pass
    return round(total, 1)


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def detect(home: Path | None = None) -> Hardware:
    hw = Hardware()
    fake = os.environ.get("FORGE_FAKE_GPUS")
    if fake:
        hw.gpus = _fake_gpus(fake)
        hw.nvlink = os.environ.get("FORGE_FAKE_NVLINK", "0") == "1"
    elif shutil.which("nvidia-smi"):
        out = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                hw.gpus.append(
                    GPU(
                        index=int(parts[0]),
                        name=parts[1].replace("NVIDIA ", ""),
                        vram_gb=round(float(parts[2]) / 1024, 1),
                        free_gb=round(float(parts[3]) / 1024, 1),
                        compute_capability=parts[4],
                    )
                )
                hw.driver = parts[5]
            except ValueError:
                continue
        header = _run(["nvidia-smi"])
        m = re.search(r"CUDA Version:\s*([\d.]+)", header)
        hw.cuda = m.group(1) if m else os.environ.get("CUDA_VERSION")
        if len(hw.gpus) > 1:
            topo = _run(["nvidia-smi", "topo", "-m"])
            hw.nvlink = bool(re.search(r"\bNV\d+\b", topo))
    hw.cpu_count = _cpu_count()
    hw.ram_gb = _ram_gb()
    target = home if home and home.exists() else Path(os.environ.get("FORGE_HOME", "/"))
    try:
        hw.disk_free_gb = round(shutil.disk_usage(target if target.exists() else Path.cwd()).free / 1024**3, 1)
    except OSError:
        hw.disk_free_gb = 0.0
    hw.torch = _version("torch")
    hw.axolotl = _version("axolotl")
    return hw
