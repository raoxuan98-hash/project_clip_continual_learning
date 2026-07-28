import csv
import fcntl
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_GPU_LOCK_PATH = "/tmp/llm_lora_nf_gpu.lock"
PARENT_MANAGED_GPU_BATCH_ENV = "LLM_LORA_NF_PARENT_MANAGED_GPU_BATCH"


@dataclass(frozen=True)
class GPUStatus:
    index: int
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: int

    def is_idle(self, *, max_memory_used_mb: int, max_utilization_percent: int) -> bool:
        return (
            self.memory_used_mb <= max_memory_used_mb
            and self.utilization_percent <= max_utilization_percent
        )


@dataclass(frozen=True)
class AdmissionDecision:
    mode: str
    selected_gpu_indices: List[int]
    idle_gpu_indices: List[int]
    reason: str


def parse_nvidia_smi_csv(text: str) -> List[GPUStatus]:
    statuses: List[GPUStatus] = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(row) != 4:
            raise ValueError(f"Expected four nvidia-smi fields, got: {row}")
        statuses.append(
            GPUStatus(
                index=int(row[0].strip()),
                memory_total_mb=int(row[1].strip()),
                memory_used_mb=int(row[2].strip()),
                utilization_percent=int(row[3].strip()),
            )
        )
    return statuses


def query_gpu_status() -> List[GPUStatus]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return parse_nvidia_smi_csv(result.stdout)


def decide_admission(
    statuses: Iterable[GPUStatus],
    *,
    requested: int = 3,
    reserve_idle: int = 1,
    max_memory_used_mb: int = 512,
    max_utilization_percent: int = 5,
    eligible_gpu_indices: Optional[Iterable[int]] = None,
) -> AdmissionDecision:
    if requested < 0 or requested > 3:
        raise ValueError("requested must be between 0 and 3")
    if reserve_idle < 1:
        raise ValueError("At least one GPU must be reserved")
    idle = [
        status.index
        for status in statuses
        if status.is_idle(
            max_memory_used_mb=max_memory_used_mb,
            max_utilization_percent=max_utilization_percent,
        )
    ]
    eligible = (
        None
        if eligible_gpu_indices is None
        else [int(index) for index in eligible_gpu_indices]
    )
    if eligible is not None and len(eligible) != len(set(eligible)):
        raise ValueError("Eligible GPU indices contain duplicates")
    idle_set = set(idle)
    eligible_idle = (
        idle
        if eligible is None
        else [index for index in eligible if index in idle_set]
    )
    if requested == 0:
        return AdmissionDecision(
            mode="cpu_smoke_only",
            selected_gpu_indices=[],
            idle_gpu_indices=idle,
            reason=(
                "no GPUs requested; CPU smoke reserves every currently idle GPU"
            ),
        )
    allowed = max(
        0,
        min(
            requested,
            3,
            len(idle) - reserve_idle,
            len(eligible_idle),
        ),
    )
    selected = eligible_idle[:allowed]
    if selected:
        return AdmissionDecision(
            mode="gpu",
            selected_gpu_indices=selected,
            idle_gpu_indices=idle,
            reason=f"selected {len(selected)} GPU(s), reserved {len(idle) - len(selected)} idle",
        )
    return AdmissionDecision(
        mode="cpu_smoke_only",
        selected_gpu_indices=[],
        idle_gpu_indices=idle,
        reason="no GPU can be used while preserving at least one idle GPU",
    )


def _numeric_visible_gpu_indices(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    if not value.strip():
        return []
    tokens = [token.strip() for token in value.split(",")]
    if not all(token.isdigit() for token in tokens):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain physical numeric GPU indices"
        )
    indices = [int(token) for token in tokens]
    if len(indices) != len(set(indices)):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate GPU indices")
    return indices


def inspect_admission(requested: int = 3) -> AdmissionDecision:
    try:
        statuses = query_gpu_status()
    except (FileNotFoundError, RuntimeError) as error:
        return AdmissionDecision(
            mode="cpu_smoke_only",
            selected_gpu_indices=[],
            idle_gpu_indices=[],
            reason=f"GPU query unavailable: {error}",
        )
    return decide_admission(
        statuses,
        requested=requested,
        eligible_gpu_indices=_numeric_visible_gpu_indices(
            os.environ.get("CUDA_VISIBLE_DEVICES")
        ),
    )


def admitted_torch_device(
    selected_physical_index: int,
    *,
    cuda_visible_devices: Optional[str],
) -> str:
    """Resolve an admitted physical GPU without mutating CUDA visibility.

    Libraries imported before admission may already have initialized PyTorch
    CUDA. Changing ``CUDA_VISIBLE_DEVICES`` inside that process can invalidate
    its cached device map, so callers must resolve the existing map instead.
    """

    if selected_physical_index < 0:
        raise ValueError("Selected physical GPU index must be non-negative")
    if cuda_visible_devices is None:
        return f"cuda:{selected_physical_index}"
    visible = cuda_visible_devices.strip()
    if not visible:
        raise RuntimeError(
            "GPU admission selected a device but CUDA_VISIBLE_DEVICES is empty"
        )
    tokens = [token.strip() for token in visible.split(",")]
    if not all(token.isdigit() for token in tokens):
        raise RuntimeError(
            "TRACE runner requires numeric CUDA_VISIBLE_DEVICES entries"
        )
    physical_indices = [int(token) for token in tokens]
    if len(set(physical_indices)) != len(physical_indices):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate GPU indices")
    if selected_physical_index not in physical_indices:
        raise RuntimeError(
            "Admitted physical GPU is outside CUDA_VISIBLE_DEVICES: "
            f"{selected_physical_index} not in {physical_indices}"
        )
    return f"cuda:{physical_indices.index(selected_physical_index)}"


@contextmanager
def project_file_lock(path: str):
    lock_path = Path(path)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def project_gpu_lock(
    path: str = DEFAULT_GPU_LOCK_PATH,
):
    if os.environ.get(PARENT_MANAGED_GPU_BATCH_ENV) == "1":
        # A bounded-parallel matrix parent holds the same global lock for the
        # lifetime of the whole child batch and assigns each child one unique
        # physical CUDA_VISIBLE_DEVICES entry. Re-acquiring here would
        # deadlock/serialize those already protected children.
        yield
        return
    with project_file_lock(path):
        yield
