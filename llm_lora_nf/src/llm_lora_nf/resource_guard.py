import csv
import fcntl
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


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
    requested: int = 2,
    reserve_idle: int = 1,
    max_memory_used_mb: int = 512,
    max_utilization_percent: int = 5,
) -> AdmissionDecision:
    if requested < 0 or requested > 2:
        raise ValueError("requested must be between 0 and 2")
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
    if requested == 0:
        return AdmissionDecision(
            mode="cpu_smoke_only",
            selected_gpu_indices=[],
            idle_gpu_indices=idle,
            reason=(
                "no GPUs requested; CPU smoke reserves every currently idle GPU"
            ),
        )
    allowed = max(0, min(requested, 2, len(idle) - reserve_idle))
    selected = idle[:allowed]
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


def inspect_admission(requested: int = 2) -> AdmissionDecision:
    try:
        statuses = query_gpu_status()
    except (FileNotFoundError, RuntimeError) as error:
        return AdmissionDecision(
            mode="cpu_smoke_only",
            selected_gpu_indices=[],
            idle_gpu_indices=[],
            reason=f"GPU query unavailable: {error}",
        )
    return decide_admission(statuses, requested=requested)


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
    path: str = "/tmp/llm_lora_nf_gpu.lock",
):
    with project_file_lock(path):
        yield
