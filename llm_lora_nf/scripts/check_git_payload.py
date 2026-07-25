#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

from llm_lora_nf.git_guard import PayloadEntry, validate_payload


def staged_paths() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]


def main() -> None:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    entries = []
    for relative in staged_paths():
        path = root / relative
        if path.is_file():
            entries.append(PayloadEntry(relative, path.stat().st_size))
    violations = validate_payload(entries)
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        raise SystemExit(1)
    print(f"git payload check passed: {len(entries)} staged file(s)")


if __name__ == "__main__":
    main()

