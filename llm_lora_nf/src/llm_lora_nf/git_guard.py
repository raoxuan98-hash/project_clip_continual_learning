from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, List


MAX_FILE_BYTES = 10 * 1024 * 1024
ALLOWED_PREFIXES = (
    "llm_lora_nf/src/",
    "llm_lora_nf/configs/",
    "llm_lora_nf/scripts/",
    "llm_lora_nf/tests/",
    "llm_lora_nf/evaluation/",
)
ALLOWED_PATHS = frozenset(
    {
        ".gitignore",
        "docs/llm-lora-nf/GOAL.md",
        "docs/llm-lora-nf/METHOD_SPEC.md",
        "docs/llm-lora-nf/MODEL_DATA_EVAL_MANIFEST.md",
        "docs/llm-lora-nf/README.md",
        "docs/llm-lora-nf/SERVER_RUNBOOK.md",
        "llm_lora_nf/README.md",
        "llm_lora_nf/pyproject.toml",
        "llm_lora_nf/requirements-code-evaluation.txt",
        "llm_lora_nf/requirements-evaluation.txt",
        "llm_lora_nf/requirements-server.txt",
    }
)
FORBIDDEN_PREFIXES = (
    "chat-history/",
    "chat-history-for-paper-writing/",
    "paper_writing/",
    "docs/llm-lora-nf/records/",
    "llm_lora_nf/data/",
    "llm_lora_nf/datasets/",
    "llm_lora_nf/checkpoints/",
    "llm_lora_nf/experiments/",
    "llm_lora_nf/artifacts/",
    "llm_lora_nf/calibration_cache/",
    "llm_lora_nf/request_cache/",
)
FORBIDDEN_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".gguf",
    ".onnx",
    ".ckpt",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
    ".arrow",
    ".parquet",
    ".tar",
    ".tar.gz",
    ".zip",
)


@dataclass(frozen=True)
class PayloadEntry:
    path: str
    size_bytes: int


def validate_payload(entries: Iterable[PayloadEntry]) -> List[str]:
    violations: List[str] = []
    for entry in entries:
        candidate = PurePosixPath(entry.path)
        normalized = candidate.as_posix()
        if candidate.is_absolute() or ".." in candidate.parts:
            violations.append(f"unsafe payload path: {normalized}")
            continue
        if (
            normalized not in ALLOWED_PATHS
            and not any(
                normalized.startswith(prefix)
                for prefix in ALLOWED_PREFIXES
            )
        ):
            violations.append(f"path outside allowed LLM code payload: {normalized}")
        if any(normalized.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            violations.append(f"forbidden path: {normalized}")
        if normalized.lower().endswith(FORBIDDEN_SUFFIXES):
            violations.append(f"forbidden artifact extension: {normalized}")
        if entry.size_bytes > MAX_FILE_BYTES:
            violations.append(
                f"file exceeds 10 MiB: {normalized} ({entry.size_bytes} bytes)"
            )
    return violations
