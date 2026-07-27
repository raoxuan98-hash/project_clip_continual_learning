#!/usr/bin/env python3
"""Audit response supervision under the exact formal Track A formatter."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.data import IGNORE_INDEX, OfficialLoRANullMathDataset
from llm_lora_nf.dataset_io import load_metamath_examples, sha256_file
from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.model_io import assert_instruct_checkpoint
from llm_lora_nf.protocol_validation import validate_track_a_formal_config
from llm_lora_nf.supervision_audit import (
    summarize_integer_series,
    summarize_shuffled_zero_schedule,
)


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize formal MetaMathQA rows with the official LoRA-Null "
            "right-truncation formatter and audit response supervision"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metamath-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    package_root = Path(__file__).resolve().parents[1]
    source = _git_state(package_root.parent)
    if source["source_dirty"]:
        raise RuntimeError("Track A supervision audit requires a clean source tree")

    config = load_yaml_config(args.config)
    validate_track_a_formal_config(config)
    model_path = Path(args.model_path).resolve()
    model_integrity = validate_directory_integrity(
        str(model_path),
        expected_kind="model_snapshot",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    assert_instruct_checkpoint(
        str(config["model"]["id"]),
        tokenizer,
        checkpoint_type=str(config["model"]["checkpoint_type"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    data_path = Path(args.metamath_json).resolve()
    first_n = int(config["train"]["first_n"])
    examples = load_metamath_examples(str(data_path), first_n=first_n)
    dataset = OfficialLoRANullMathDataset(
        examples,
        tokenizer,
        max_length=int(config["train"]["max_sequence_length"]),
    )
    supervised_counts = []
    sequence_lengths = []
    for index in range(len(dataset)):
        row = dataset[index]
        supervised_counts.append(
            int(row["labels"].ne(IGNORE_INDEX).sum().item())
        )
        sequence_lengths.append(int(row["input_ids"].numel()))
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(f"audited_rows={index + 1}", flush=True)

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite Track A supervision audit: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_manifest = model_path / "local_snapshot_manifest.json"
    supervised_summary = summarize_integer_series(supervised_counts)
    shuffle_generator = torch.Generator().manual_seed(int(config["run"]["seed"]))
    shuffled_indices = [
        int(batch.item())
        for batch in DataLoader(
            range(first_n),
            batch_size=1,
            shuffle=True,
            generator=shuffle_generator,
        )
    ]
    schedule_summary = summarize_shuffled_zero_schedule(
        shuffled_indices,
        zero_source_indices=supervised_summary["zero_indices"],
        consumed_rows=int(config["train"]["consumed_examples_per_epoch"]),
        accumulation_steps=(
            int(config["train"]["global_batch_size"])
            // int(config["train"]["per_device_batch_size"])
        ),
    )
    payload = {
        "format_version": 2,
        "audit": "track_a_official_response_supervision",
        "source": source,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": canonical_config_hash(config),
        "protocol": {
            "prompt_format": config["train"]["prompt_format"],
            "truncation_strategy": config["train"]["truncation_strategy"],
            "max_sequence_length": int(
                config["train"]["max_sequence_length"]
            ),
            "fully_truncated_policy": (
                "consume_as_zero_gradient_micro_batch_preserve_schedule"
            ),
        },
        "model": {
            "requested_id": config["model"]["id"],
            "source_id": config["model"]["source_id"],
            "resolved_path": str(model_path),
            "tokenizer_class": tokenizer.__class__.__name__,
            "snapshot_manifest_sha256": _sha256(snapshot_manifest),
            "snapshot_integrity_manifest_sha256": model_integrity[
                "manifest_sha256"
            ],
        },
        "dataset": {
            "path": str(data_path),
            "resolved_path": str(data_path.resolve()),
            "size_bytes": data_path.stat().st_size,
            "sha256": sha256_file(str(data_path)),
            "first_n": first_n,
        },
        "supervised_tokens_per_row": supervised_summary,
        "sequence_tokens_per_row": summarize_integer_series(sequence_lengths),
        "seeded_training_schedule": {
            "seed": int(config["run"]["seed"]),
            "sampler": "torch_dataloader_random_sampler",
            **schedule_summary,
        },
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
                "supervised_tokens_per_row": payload[
                    "supervised_tokens_per_row"
                ],
                "seeded_training_schedule": payload[
                    "seeded_training_schedule"
                ],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
