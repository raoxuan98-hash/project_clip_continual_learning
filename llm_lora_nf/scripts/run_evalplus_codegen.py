#!/usr/bin/env python3
"""Generate one greedy EvalPlus solution per task without executing code."""

import argparse
import copy
import gc
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.environment import (
    code_evaluation_package_versions,
    runtime_environment,
    validate_code_evaluation_package_versions,
)
from llm_lora_nf.evalplus_evidence import (
    stream_jsonl,
    validate_evalplus_data_manifest,
)
from llm_lora_nf.evalplus_results import validate_evalplus_samples
from llm_lora_nf.integrity import (
    validate_directory_integrity,
    write_directory_integrity,
)
from llm_lora_nf.protocol_validation import (
    validate_formal_code_evaluation_config,
)
from llm_lora_nf.resource_guard import inspect_admission, project_gpu_lock


def _git_state(path: Path, *, scope: str = ".") -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", scope],
            cwd=str(path),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit_sha": commit, "source_dirty": dirty}


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _validate_model(
    model_path: Path,
    manifest_path: Path,
    *,
    role: str,
    source_commit: str,
) -> Dict[str, Any]:
    manifest = _load_json(manifest_path)
    if role == "merged":
        if manifest_path != model_path / "merged_export_manifest.json":
            raise ValueError("Merged manifest must be inside the checkpoint")
        integrity = validate_directory_integrity(
            str(model_path),
            expected_kind="merged_checkpoint",
        )
        if (
            not manifest.get("formal_result_eligible")
            or manifest.get("execution_mode") != "gpu_formal"
            or manifest.get("dtype") != "torch.float32"
            or manifest.get("export_source_dirty")
            or manifest.get("export_commit_sha") != source_commit
            or manifest.get("training_commit_sha") != source_commit
        ):
            raise ValueError("Unqualified merged checkpoint for code generation")
    else:
        if manifest_path != model_path / "local_snapshot_manifest.json":
            raise ValueError("Base manifest must be inside the checkpoint")
        integrity = validate_directory_integrity(
            str(model_path),
            expected_kind="model_snapshot",
        )
        if (
            Path(str(manifest.get("resolved_path", ""))).resolve() != model_path
            or not manifest.get("files")
        ):
            raise ValueError("Unqualified base snapshot for code generation")
    return {"manifest": manifest, "integrity": integrity}


def _run(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.config)
    if args.seed is not None:
        config["run"]["seed"] = args.seed
    validate_formal_code_evaluation_config(config)
    config_hash = canonical_config_hash(config)
    protocol = copy.deepcopy(config)
    protocol["run"].pop("seed", None)
    protocol_hash = canonical_config_hash(protocol)

    repo_root = Path(__file__).resolve().parents[2]
    source = _git_state(repo_root, scope="llm_lora_nf")
    evaluator_path = Path(args.evaluator_path).resolve()
    evaluator = _git_state(evaluator_path)
    if (
        source["source_dirty"]
        or evaluator["source_dirty"]
        or evaluator["commit_sha"] != config["evaluator"]["revision"]
    ):
        raise RuntimeError(
            "Formal EvalPlus generation requires clean, exact source revisions"
        )
    data = validate_evalplus_data_manifest(
        args.dataset_manifest,
        config=config,
    )
    model_path = Path(args.model_path).resolve()
    model = _validate_model(
        model_path,
        Path(args.model_manifest).resolve(),
        role=args.model_role,
        source_commit=source["commit_sha"],
    )

    admission = inspect_admission(requested=1)
    if admission.mode == "gpu":
        selected_gpu = admission.selected_gpu_indices[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        execution_mode = "gpu_formal"
        smoke_limit = None
        device = torch.device("cuda:0")
    elif args.cpu_smoke_fallback:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        execution_mode = "cpu_smoke_only"
        smoke_limit = args.limit_tasks or 1
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            "No admissible GPU is available; CPU fallback is chain-only"
        )
    if execution_mode == "gpu_formal" and args.limit_tasks is not None:
        raise ValueError("Formal EvalPlus generation forbids task limits")
    environment = runtime_environment(
        device,
        validate_formal=execution_mode == "gpu_formal",
    )
    evalplus_packages = code_evaluation_package_versions()
    validate_code_evaluation_package_versions(evalplus_packages)

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HUMANEVAL_OVERRIDE_PATH"] = data["datasets"]["humaneval"][
        "jsonl"
    ]["path"]
    os.environ["MBPP_OVERRIDE_PATH"] = data["datasets"]["mbpp"]["jsonl"]["path"]
    sys.path.insert(0, str(evaluator_path))
    from evalplus.codegen import codegen
    from evalplus.provider import make_model
    from evalplus.provider.base import DecoderBase

    default_max_new_tokens = inspect.signature(
        DecoderBase.__init__
    ).parameters["max_new_tokens"].default
    if default_max_new_tokens != config["generation"]["max_new_tokens"]:
        raise RuntimeError(
            "Pinned EvalPlus max_new_tokens default drifted: "
            f"{default_max_new_tokens}"
        )
    generation = config["generation"]
    started = time.monotonic()
    model_runner = make_model(
        model=str(model_path),
        backend=generation["backend"],
        dataset="humaneval",
        batch_size=generation["batch_size"],
        temperature=generation["temperature"],
        force_base_prompt=generation["force_base_prompt"],
        instruction_prefix=generation["instruction_prefix"],
        response_prefix=generation["response_prefix"],
        dtype=generation["dtype"],
        trust_remote_code=generation["trust_remote_code"],
        attn_implementation=generation["attn_implementation"],
    )
    if model_runner.is_direct_completion():
        raise RuntimeError(
            "Formal code evaluation requires the Instruct chat template"
        )
    sample_records = {}
    try:
        for dataset in ("humaneval", "mbpp"):
            sample_path = output_dir / f"{dataset}_samples.jsonl"
            id_range = None
            if smoke_limit is not None:
                task_ids = [
                    str(row["task_id"])
                    for row in stream_jsonl(
                        Path(data["datasets"][dataset]["jsonl"]["path"])
                    )
                ]
                first_id = int(task_ids[0].split("/")[-1])
                id_range = (first_id, first_id + smoke_limit)
            codegen(
                target_path=str(sample_path),
                model=model_runner,
                dataset=dataset,
                greedy=True,
                n_samples=1,
                id_range=id_range,
                version="default",
                resume=False,
            )
            sample_records[dataset] = validate_evalplus_samples(
                sample_path,
                dataset_path=Path(
                    data["datasets"][dataset]["jsonl"]["path"]
                ),
                allow_subset=smoke_limit is not None,
            )
    finally:
        del model_runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "format_version": 1,
        "status": "generated",
        "formal_result_eligible": execution_mode == "gpu_formal",
        "execution_mode": execution_mode,
        "source": source,
        "evaluator": {
            **evaluator,
            "repository": config["evaluator"]["repository"],
            "version": config["evaluator"]["version"],
        },
        "config_hash": config_hash,
        "protocol_config_hash": protocol_hash,
        "seed": int(config["run"]["seed"]),
        "model_role": args.model_role,
        "model_path": str(model_path),
        "model_manifest": model["manifest"],
        "model_integrity": model["integrity"],
        "dataset_evidence": data,
        "samples": sample_records,
        "generation": generation,
        "environment": environment,
        "code_evaluation_packages": evalplus_packages,
        "admission": {
            "mode": admission.mode,
            "selected_gpu_indices": admission.selected_gpu_indices,
            "idle_gpu_indices": admission.idle_gpu_indices,
            "reason": admission.reason,
        },
        "wall_seconds": time.monotonic() - started,
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "code_generation_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    integrity = write_directory_integrity(
        str(output_dir),
        kind="code_generation_output",
    )
    print(
        json.dumps(
            {"report": report, "integrity": integrity},
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resource-guarded official EvalPlus greedy code generation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluator-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-role", choices=["base", "merged"], required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu-smoke-fallback", action="store_true")
    parser.add_argument("--limit-tasks", type=int)
    args = parser.parse_args()
    with project_gpu_lock():
        _run(args)


if __name__ == "__main__":
    main()
