from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, List


MAX_FILE_BYTES = 10 * 1024 * 1024
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
)


@dataclass(frozen=True)
class PayloadEntry:
    path: str
    size_bytes: int


def validate_payload(entries: Iterable[PayloadEntry]) -> List[str]:
    violations: List[str] = []
    for entry in entries:
        normalized = PurePosixPath(entry.path).as_posix()
        if any(normalized.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            violations.append(f"forbidden path: {normalized}")
        if entry.size_bytes > MAX_FILE_BYTES:
            violations.append(
                f"file exceeds 10 MiB: {normalized} ({entry.size_bytes} bytes)"
            )
    return violations

