#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

from llm_lora_nf.checkpoint import load_native_adapter
from llm_lora_nf.config_io import adapter_config_from_mapping
from llm_lora_nf.environment import (
    execution_environment_identity,
    runtime_environment,
)
from llm_lora_nf.inject import merge_native_adapters
from llm_lora_nf.integrity import (
    validate_directory_integrity,
    write_directory_integrity,
)
from llm_lora_nf.method_registry import (
    TRACK_B_METHODS,
    TRACK_B_NATIVE_METHODS,
)
from llm_lora_nf.model_io import load_instruct_model
from llm_lora_nf.track_a import (
    load_track_a_native_adapter,
    track_a_config_from_mapping,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _git_state() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "llm_lora_nf"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit_sha": commit, "source_dirty": dirty}


def _validation_batch(tokenizer, device: torch.device) -> Dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": "Reply with the single word: merge."}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=64,
    )
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def _logits(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    return model(**batch).logits.detach().float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a trained adapter into a standard offline HF checkpoint"
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--run-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--requested-id", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--checkpoint-type", default="instruct")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
    )
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="Permit exporting an ineligible CPU smoke checkpoint for chain testing.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    run_report_path = Path(args.run_report).resolve()
    output_dir = Path(args.output_dir).resolve()
    report = _load_json(run_report_path)
    eligible = bool(report.get("formal_result_eligible", False))
    if not eligible and not args.allow_cpu_smoke:
        raise ValueError(
            "CPU smoke checkpoints cannot be exported for formal evaluation; "
            "pass --allow-cpu-smoke only to validate the export chain"
        )
    if eligible and args.dtype != "float32":
        raise ValueError(
            "Formal primary evaluation requires an FP32 merged export; "
            "lower-precision merge belongs to a separately ineligible ablation"
        )
    source = _git_state()
    if eligible:
        if source["source_dirty"]:
            raise RuntimeError("Formal export requires a clean source tree")
        if source["commit_sha"] != report["metadata"]["commit_sha"]:
            raise RuntimeError(
                "Formal export must use the exact training commit"
            )
    method = str(report["metadata"]["method"])
    if method not in TRACK_B_METHODS:
        raise ValueError(f"Unsupported method in run report: {method}")
    protocol_track = str(report.get("protocol_track", "B"))
    if protocol_track not in {"A", "B"}:
        raise ValueError(f"Unsupported protocol track: {protocol_track}")
    native_checkpoint = (
        protocol_track == "A" and method == "lora_null"
    ) or (
        protocol_track == "B" and method in TRACK_B_NATIVE_METHODS
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    checkpoint_integrity = validate_directory_integrity(
        str(checkpoint_dir),
        expected_kind="adapter_checkpoint",
    )

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("cpu")
    environment_record = runtime_environment(
        device,
        validate_formal=eligible,
    )
    base_model_integrity = None
    if eligible:
        base_model_integrity = validate_directory_integrity(
            args.base_model_path,
            expected_kind="model_snapshot",
        )
    model, tokenizer, load_record = load_instruct_model(
        requested_id=args.requested_id,
        source_id=args.source_id,
        resolved_path=args.base_model_path,
        checkpoint_type=args.checkpoint_type,
        device=str(device),
        local_files_only=True,
        torch_dtype=dtype,
    )
    if eligible and (
        load_record.snapshot_manifest_sha256 is None
        or load_record.snapshot_integrity_sha256 is None
    ):
        raise RuntimeError(
            "Formal export requires hashed provenance and integrity manifests "
            "for the local model snapshot"
        )

    if native_checkpoint:
        adapter_manifest_path = checkpoint_dir / "adapter_manifest.json"
        adapter_manifest = _load_json(adapter_manifest_path)
        if protocol_track == "A":
            adapter_config = track_a_config_from_mapping(
                adapter_manifest["adapter_config"]
            )
        else:
            adapter_config = adapter_config_from_mapping(
                adapter_manifest["adapter_config"]
            )
        if adapter_config.method != method:
            raise ValueError(
                f"Run report method {method} does not match adapter "
                f"manifest method {adapter_config.method}"
            )
        if protocol_track == "A":
            load_track_a_native_adapter(
                model,
                str(checkpoint_dir),
                adapter_config=adapter_config,
            )
        else:
            load_native_adapter(
                model,
                str(checkpoint_dir),
                adapter_config=adapter_config,
            )
        checkpoint_manifest_sha256 = _sha256(adapter_manifest_path)
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(checkpoint_dir),
            is_trainable=False,
            local_files_only=True,
        )
        checkpoint_manifest_sha256 = _sha256(checkpoint_dir / "adapter_config.json")

    validation_batch = _validation_batch(tokenizer, device)
    unmerged_logits = _logits(model, validation_batch)
    if native_checkpoint:
        merge_native_adapters(model)
    else:
        model = model.merge_and_unload(safe_merge=True)
    merged_logits = _logits(model, validation_batch)
    tolerance = (
        {"atol": 2e-4, "rtol": 2e-4}
        if dtype == torch.float32
        else {"atol": 2e-2, "rtol": 2e-2}
    )
    torch.testing.assert_close(merged_logits, unmerged_logits, **tolerance)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=str(output_dir.parent),
        )
    )
    try:
        model.config.use_cache = True
        model.save_pretrained(
            str(temporary),
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(str(temporary))
        export_manifest = {
            "format_version": 1,
            "method": method,
            "protocol_track": protocol_track,
            "formal_result_eligible": eligible,
            "execution_mode": report["metadata"]["execution_mode"],
            "run_id": report["metadata"]["run_id"],
            "training_seed": int(report["metadata"]["seed"]),
            "training_model_id": str(report["metadata"]["model_id"]),
            "training_commit_sha": report["metadata"]["commit_sha"],
            "training_protocol_config_hash": report.get(
                "protocol_config_hash",
                report["config_hash"],
            ),
            "training_comparison_config_hash": report.get(
                "comparison_config_hash",
                report.get("protocol_config_hash", report["config_hash"]),
            ),
            "training_environment_identity": execution_environment_identity(
                report["environment"]
            ),
            "export_commit_sha": source["commit_sha"],
            "export_source_dirty": source["source_dirty"],
            "run_report_sha256": _sha256(run_report_path),
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "checkpoint_integrity_manifest_sha256": checkpoint_integrity[
                "manifest_sha256"
            ],
            "checkpoint_integrity_tree_sha256": checkpoint_integrity[
                "tree_sha256"
            ],
            "base_model": load_record.to_dict(),
            "base_model_integrity_manifest_sha256": (
                base_model_integrity["manifest_sha256"]
                if base_model_integrity is not None
                else None
            ),
            "environment": environment_record,
            "dtype": str(dtype),
            "merge_validation": tolerance,
            "checkpoint_retention": (
                "ephemeral_delete_after_qualified_evaluation"
                if eligible
                else "cpu_smoke_manual_cleanup"
            ),
        }
        (temporary / "merged_export_manifest.json").write_text(
            json.dumps(export_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        merged_integrity = write_directory_integrity(
            str(temporary),
            kind="merged_checkpoint",
        )
        if output_dir.exists():
            output_dir.rmdir()
        temporary.rename(output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    merged_integrity["manifest_path"] = str(
        output_dir / "artifact_integrity.json"
    )
    print(
        json.dumps(
            {
                "export": export_manifest,
                "integrity": merged_integrity,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
