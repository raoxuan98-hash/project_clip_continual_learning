#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.trace_length_audit import (
    summarize_trace_token_lengths,
    trace_example_token_lengths,
)
from llm_lora_nf.trace_protocol import TRACE_OFFICIAL_ORDER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "llm_lora_nf"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit_sha": commit, "source_dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit TRACE chat-template token lengths without training"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    source = _git_state(package_root.parent)
    if source["source_dirty"]:
        raise RuntimeError("TRACE length audit requires a clean source tree")
    model_path = Path(args.model_path).resolve()
    model_integrity = validate_directory_integrity(
        str(model_path),
        expected_kind="model_snapshot",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )

    data_root = Path(args.trace_data_root).resolve()
    by_task = {}
    all_train_records = []
    all_records = []
    file_records = []
    for task in TRACE_OFFICIAL_ORDER:
        by_task[task] = {}
        for split in ("train", "eval", "test"):
            path = data_root / task / f"{split}.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            records = [
                trace_example_token_lengths(
                    tokenizer,
                    prompt=str(row["prompt"]),
                    answer=str(row["answer"]),
                    enable_thinking=args.enable_thinking,
                )
                for row in rows
            ]
            by_task[task][split] = summarize_trace_token_lengths(records)
            all_records.extend(records)
            if split == "train":
                all_train_records.extend(records)
            file_records.append(
                {
                    "task": task,
                    "split": split,
                    "rows": len(rows),
                    "sha256": _sha256(path),
                }
            )

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite TRACE length audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "training_truncation_protocol": {
            "name": "trace_official_combined_left_v1",
            "max_prompt_length": 1024,
            "max_answer_length": 512,
            "combined_max_sequence_length": 1536,
            "side": "left",
            "response_only_loss": True,
        },
        "source": source,
        "model_path": str(model_path),
        "model_integrity_manifest_sha256": model_integrity["manifest_sha256"],
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_model_max_length": int(tokenizer.model_max_length),
        "enable_thinking": bool(args.enable_thinking),
        "data_root": str(data_root),
        "files": file_records,
        "train": summarize_trace_token_lengths(all_train_records),
        "all_splits": summarize_trace_token_lengths(all_records),
        "by_task": by_task,
    }
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "train": payload["train"],
                "all_splits": payload["all_splits"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
