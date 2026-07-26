import subprocess
from pathlib import Path
from typing import Any, Dict


def git_source_state(
    repo_root: Path,
    *,
    pathspec: str = "llm_lora_nf",
) -> Dict[str, Any]:
    root = repo_root.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", pathspec],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "commit_sha": commit,
        "source_dirty": dirty,
    }


def validate_derivation_source(
    source: Dict[str, Any],
    *,
    expected_commit: str,
    formal: bool,
) -> None:
    if formal and (
        source.get("source_dirty")
        or source.get("commit_sha") != expected_commit
    ):
        raise ValueError(
            "Formal derived evidence requires the exact clean evaluation "
            f"commit {expected_commit}"
        )
