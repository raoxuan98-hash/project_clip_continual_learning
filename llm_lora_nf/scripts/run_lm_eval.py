#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.dataset_evidence import validate_eval_data_manifest
from llm_lora_nf.environment import (
    runtime_environment,
    software_environment_identity,
)
from llm_lora_nf.evaluator_tasks import materialize_pinned_tasks
from llm_lora_nf.integrity import (
    INTEGRITY_FILENAME,
    validate_directory_integrity,
    write_directory_integrity,
)
from llm_lora_nf.protocol_validation import validate_formal_evaluation_config
from llm_lora_nf.resource_guard import inspect_admission, project_gpu_lock


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(arguments: List[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_evaluator(path: Path, expected_revision: str) -> None:
    actual = _git_output(["rev-parse", "HEAD"], path)
    if actual != expected_revision:
        raise ValueError(
            f"Evaluator checkout is {actual}, expected {expected_revision}"
        )
    if _git_output(["status", "--porcelain"], path):
        raise ValueError("Evaluator checkout must be clean")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _source_state(repo_root: Path) -> Dict[str, Any]:
    return {
        "commit_sha": _git_output(["rev-parse", "HEAD"], repo_root),
        "source_dirty": bool(
            _git_output(
                ["status", "--porcelain", "--", "llm_lora_nf"],
                repo_root,
            )
        ),
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualified, resource-guarded offline lm-eval runner"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluator-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-role", choices=["base", "merged"], required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--request-cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--cpu-smoke-fallback", action="store_true")
    parser.add_argument(
        "--tasks",
        help="Comma-separated task override; accepted only for CPU smoke.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(str(config_path))
    if args.seed is not None:
        config["run"]["seed"] = args.seed
    if args.run_name is not None:
        config["run"]["name"] = args.run_name
    config_hash = canonical_config_hash(config)
    protocol_config = copy.deepcopy(config)
    protocol_config["run"].pop("seed", None)
    protocol_config["run"].pop("name", None)
    protocol_config_hash = canonical_config_hash(protocol_config)
    evaluator_path = Path(args.evaluator_path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    source = _source_state(repo_root)
    expected_revision = str(config["evaluator"]["revision"])
    _validate_evaluator(evaluator_path, expected_revision)
    hf_home = Path(args.hf_home).resolve()
    dataset_evidence = validate_eval_data_manifest(
        args.dataset_manifest,
        config=config,
        hf_home=str(hf_home),
    )

    model_manifest_path = Path(args.model_manifest).resolve()
    model_manifest = _load_json(model_manifest_path)
    model_path = Path(args.model_path).resolve()
    if args.model_role == "merged":
        expected_manifest_path = model_path / "merged_export_manifest.json"
        if model_manifest_path != expected_manifest_path:
            raise ValueError(
                "Merged models require the manifest inside the evaluated "
                f"checkpoint: {expected_manifest_path}"
            )
        model_formal_eligible = bool(
            model_manifest.get("formal_result_eligible", False)
        )
        model_integrity_record = validate_directory_integrity(
            str(model_path),
            expected_kind="merged_checkpoint",
        )
        if model_formal_eligible:
            if model_manifest.get("export_source_dirty"):
                raise ValueError("Merged export was created from dirty source")
            if model_manifest.get("export_commit_sha") != source["commit_sha"]:
                raise ValueError(
                    "Formal evaluation must use the exact export commit"
                )
            if (
                model_manifest.get("format_version") != 1
                or model_manifest.get("execution_mode") != "gpu_formal"
                or model_manifest.get("training_commit_sha")
                != model_manifest.get("export_commit_sha")
                or model_manifest.get("dtype") != "torch.float32"
                or model_manifest.get("checkpoint_retention")
                != "ephemeral_delete_after_qualified_evaluation"
            ):
                raise ValueError(
                    "Formal primary evaluation requires the qualified FP32 "
                    "ephemeral merged export"
                )
    else:
        expected_manifest_path = model_path / "local_snapshot_manifest.json"
        if model_manifest_path != expected_manifest_path:
            raise ValueError(
                "Base models require the local snapshot manifest inside the "
                f"evaluated checkpoint: {expected_manifest_path}"
            )
        expected_source = model_manifest.get("model_id")
        if not expected_source:
            raise ValueError("Base model snapshot manifest lacks model_id")
        resolved_snapshot_path = model_manifest.get("resolved_path")
        if (
            resolved_snapshot_path is None
            or Path(resolved_snapshot_path).resolve() != model_path
        ):
            raise ValueError(
                "Base model snapshot manifest resolved_path does not match "
                f"the evaluated checkpoint: {model_path}"
            )
        if not model_manifest.get("files"):
            raise ValueError("Base model snapshot manifest contains no files")
        model_integrity_record = validate_directory_integrity(
            str(model_path),
            expected_kind="model_snapshot",
        )
        model_formal_eligible = True

    configured_tasks = [
        *config["tasks"]["downstream"],
        *config["tasks"]["retention"],
    ]
    task_override = (
        [task.strip() for task in args.tasks.split(",") if task.strip()]
        if args.tasks
        else None
    )
    requested_mode = str(config["run"]["execution_mode"])
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    tokenizer_path = (
        Path(model_manifest["base_model"]["resolved_path"]).resolve()
        if args.model_role == "merged"
        else model_path
    )
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"Missing shared base tokenizer snapshot: {tokenizer_path}"
        )
    request_cache_root = Path(args.request_cache_root).resolve()
    for protected_path, label in (
        (output_dir, "evaluation output"),
        (model_path, "evaluated model"),
        (tokenizer_path, "shared base tokenizer"),
        (evaluator_path, "evaluator checkout"),
    ):
        if _paths_overlap(request_cache_root, protected_path):
            raise ValueError(
                "The dedicated request-cache root must not overlap the "
                f"{label}: {request_cache_root}"
            )
    evaluator_seeds = {
        key: int(config["evaluator"]["seeds"][key])
        for key in ("random", "numpy", "torch", "fewshot")
    }
    with project_gpu_lock():
        admission = inspect_admission(requested=1)
        if requested_mode == "gpu_formal" and admission.mode == "gpu":
            validate_formal_evaluation_config(config)
            if source["source_dirty"]:
                raise RuntimeError("Formal evaluation requires a clean source tree")
            if not model_formal_eligible:
                raise RuntimeError("Merged CPU smoke checkpoints are not formal results")
            if task_override is not None or args.limit is not None:
                raise ValueError("Formal evaluation forbids task overrides and limits")
            selected_gpu = admission.selected_gpu_indices[0]
            execution_mode = "gpu_formal"
            tasks = configured_tasks
            limit = None
            device = "cuda:0"
            dtype = str(config["runtime"]["dtype"])
            visible_devices = str(selected_gpu)
        elif args.cpu_smoke_fallback:
            execution_mode = "cpu_smoke_only"
            tasks = task_override or configured_tasks[:1]
            limit = args.limit or 1
            device = "cpu"
            dtype = "float32"
            visible_devices = ""
            runtime_batch_size = "1"
            runtime_max_batch_size = "1"
        else:
            raise RuntimeError(
                "No admissible GPU is available; use --cpu-smoke-fallback only "
                "for a non-result chain check"
            )
        if execution_mode == "gpu_formal":
            runtime_batch_size = str(config["runtime"]["batch_size"])
            runtime_max_batch_size = str(config["runtime"]["max_batch_size"])
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        environment_record = runtime_environment(
            torch.device(device),
            validate_formal=execution_mode == "gpu_formal",
        )
        environment_identity = software_environment_identity(
            environment_record
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        task_definitions = materialize_pinned_tasks(
            str(evaluator_path),
            str(output_dir / "task_definitions"),
            config=config,
        )
        request_cache_identity = {
            "format_version": 2,
            "execution_mode": execution_mode,
            "tasks": tasks,
            "limit": limit,
            "evaluator_revision": expected_revision,
            "protocol_config_hash": protocol_config_hash,
            "dataset_manifest_sha256": dataset_evidence["manifest_sha256"],
            "task_definition_tree_sha256": task_definitions["tree_sha256"],
            "base_snapshot_integrity_sha256": (
                model_manifest["base_model"]["snapshot_integrity_sha256"]
                if args.model_role == "merged"
                else model_integrity_record["manifest_sha256"]
            ),
            "tokenizer_path": str(tokenizer_path),
            "evaluator_seeds": evaluator_seeds,
            "software_environment": environment_identity,
            "cache_requests_enabled": execution_mode == "gpu_formal",
        }
        request_cache_key = hashlib.sha256(
            json.dumps(
                request_cache_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        request_cache_path = request_cache_root / request_cache_key
        request_cache_manifest = request_cache_path / INTEGRITY_FILENAME
        request_cache_identity_path = (
            request_cache_path / "request_cache_identity.json"
        )
        prior_request_cache_integrity = None
        if request_cache_path.exists() and any(request_cache_path.iterdir()):
            if not request_cache_manifest.is_file():
                raise ValueError(
                    "Unsealed shared request cache requires audit: "
                    f"{request_cache_path}"
                )
            prior_request_cache_integrity = validate_directory_integrity(
                str(request_cache_path),
                expected_kind="lm_eval_request_cache",
            )
            if _load_json(request_cache_identity_path) != request_cache_identity:
                raise ValueError(
                    "Shared request-cache identity marker does not match its "
                    f"content-addressed key: {request_cache_path}"
                )
        else:
            request_cache_path.mkdir(parents=True, exist_ok=True)
            request_cache_identity_path.write_text(
                json.dumps(
                    request_cache_identity,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        model_args = [
            f"pretrained={model_path}",
            f"dtype={dtype}",
            f"tokenizer={tokenizer_path}",
            "trust_remote_code=False",
            "enable_thinking="
            + str(bool(config["prompt"].get("enable_thinking", False))),
        ]
        if execution_mode == "cpu_smoke_only":
            model_args.append("max_length=512")
        command = [
            sys.executable,
            "-m",
            "lm_eval",
            "run",
            "--model",
            "hf",
            "--model_args",
            ",".join(model_args),
            "--tasks",
            ",".join(tasks),
            "--include_path",
            str(output_dir / "task_definitions"),
            "--batch_size",
            runtime_batch_size,
            "--max_batch_size",
            runtime_max_batch_size,
            "--device",
            device,
            "--output_path",
            str(output_dir / "lm_eval"),
            "--seed",
            ",".join(
                [
                    str(evaluator_seeds["random"]),
                    str(evaluator_seeds["numpy"]),
                    str(evaluator_seeds["torch"]),
                    str(evaluator_seeds["fewshot"]),
                ]
            ),
            "--show_config",
        ]
        if bool(config["prompt"]["apply_chat_template"]):
            command.append("--apply_chat_template")
        if bool(config["prompt"].get("fewshot_as_multiturn", False)):
            command.extend(["--fewshot_as_multiturn", "true"])
        if bool(config["runtime"].get("log_samples", True)):
            command.append("--log_samples")
        if (
            execution_mode == "gpu_formal"
            and bool(config["runtime"].get("cache_requests", True))
        ):
            command.extend(["--cache_requests", "true"])
        if limit is not None:
            command.extend(["--limit", str(limit)])

        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": visible_devices,
                "HF_HOME": str(Path(args.hf_home).resolve()),
                "HF_DATASETS_CACHE": str(
                    (hf_home / "datasets").resolve()
                ),
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "LMEVAL_LOG_LEVEL": "INFO",
                "LM_HARNESS_CACHE_PATH": str(request_cache_path),
                "PYTHONPATH": os.pathsep.join(
                    [
                        str(evaluator_path),
                        environment.get("PYTHONPATH", ""),
                    ]
                ),
            }
        )
        completed = subprocess.run(
            command,
            cwd=str(evaluator_path),
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            request_cache_integrity = None
        elif prior_request_cache_integrity is None:
            request_cache_integrity = write_directory_integrity(
                str(request_cache_path),
                kind="lm_eval_request_cache",
            )
        else:
            request_cache_integrity = validate_directory_integrity(
                str(request_cache_path),
                expected_kind="lm_eval_request_cache",
            )

    run_manifest = {
        "format_version": 2,
        "execution_mode": execution_mode,
        "formal_result_eligible": execution_mode == "gpu_formal",
        "run_name": config["run"]["name"],
        "seed": int(config["run"]["seed"]),
        "tasks": tasks,
        "limit": limit,
        "model_role": args.model_role,
        "model_path": str(model_path),
        "model_manifest_path": str(model_manifest_path),
        "model_manifest_sha256": _sha256(model_manifest_path),
        "model_manifest": model_manifest,
        "model_integrity": model_integrity_record,
        "model_checkpoint_retention": (
            "ephemeral_delete_after_qualified_evaluation"
            if args.model_role == "merged"
            else "shared_persistent_snapshot_per_revision"
        ),
        "evaluator_revision": expected_revision,
        "evaluator_seeds": evaluator_seeds,
        "dataset_evidence": dataset_evidence,
        "task_definitions": task_definitions,
        "request_cache": {
            "identity": request_cache_identity,
            "key": request_cache_key,
            "path": str(request_cache_path),
            "integrity": request_cache_integrity,
            "retention": (
                "shared_content_addressed_per_base_model_protocol"
            ),
        },
        "config_hash": config_hash,
        "protocol_config_hash": protocol_config_hash,
        "source": source,
        "environment": environment_record,
        "admission": {
            "mode": admission.mode,
            "selected_gpu_indices": admission.selected_gpu_indices,
            "idle_gpu_indices": admission.idle_gpu_indices,
            "reason": admission.reason,
        },
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "command": command,
    }
    (output_dir / "evaluation_run.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_integrity = write_directory_integrity(
        str(output_dir),
        kind="evaluation_output",
    )
    print(
        json.dumps(
            {
                "run": run_manifest,
                "integrity": output_integrity,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
