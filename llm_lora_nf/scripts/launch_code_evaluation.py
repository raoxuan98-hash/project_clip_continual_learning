#!/usr/bin/env python3
"""Resumable low-storage export, EvalPlus generation, and sandbox execution."""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from llm_lora_nf.artifact_retention import validate_artifact_retention
from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.evalplus_evidence import validate_evalplus_data_manifest
from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.protocol_validation import (
    validate_formal_code_evaluation_config,
)
from llm_lora_nf.resource_guard import project_file_lock
try:
    from scripts.launch_math_evaluation import (
        _assert_single_merged_slot,
        _completed_export,
        _delete_ephemeral_merged,
        _validate_training_run,
    )
except ModuleNotFoundError:
    from launch_math_evaluation import (
        _assert_single_merged_slot,
        _completed_export,
        _delete_ephemeral_merged,
        _validate_training_run,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_dirty(repo_root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "llm_lora_nf"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _completed_codegen(path: Path, *, expected_role: str) -> bool:
    report_path = path / "code_generation_run.json"
    integrity_path = path / "artifact_integrity.json"
    if not report_path.is_file() or not integrity_path.is_file():
        return False
    validate_directory_integrity(
        str(path),
        expected_kind="code_generation_output",
    )
    report = _load_json(report_path)
    return bool(
        report.get("format_version") == 1
        and report.get("status") == "generated"
        and report.get("formal_result_eligible")
        and report.get("execution_mode") == "gpu_formal"
        and report.get("model_role") == expected_role
        and report.get("source", {}).get("commit_sha")
        and not report.get("source", {}).get("source_dirty")
    )


def _completed_execution(path: Path) -> bool:
    report_path = path / "code_execution_run.json"
    integrity_path = path / "artifact_integrity.json"
    if not report_path.is_file() or not integrity_path.is_file():
        return False
    validate_directory_integrity(
        str(path),
        expected_kind="code_execution_output",
    )
    report = _load_json(report_path)
    metrics = report.get("metrics", {})
    return bool(
        report.get("format_version") == 1
        and report.get("status") == "passed"
        and report.get("formal_result_eligible")
        and report.get("execution_mode") == "cpu_sandbox_formal"
        and set(metrics)
        == {
            "humaneval_pass_at_1",
            "humaneval_plus_pass_at_1",
            "mbpp_pass_at_1",
            "mbpp_plus_pass_at_1",
        }
        and report.get("source", {}).get("commit_sha")
        and not report.get("source", {}).get("source_dirty")
    )


def _validate_codegen_identity(
    path: Path,
    *,
    expected_role: str,
    expected_commit: str,
    expected_protocol_hash: str,
    expected_data: Dict[str, Any],
    expected_model_integrity_sha: str,
    expected_model_manifest: Dict[str, Any],
    expected_seed: int,
) -> Dict[str, Any]:
    if not _completed_codegen(path, expected_role=expected_role):
        raise ValueError(f"Unqualified EvalPlus generation: {path}")
    report = _load_json(path / "code_generation_run.json")
    if (
        report.get("source", {}).get("commit_sha") != expected_commit
        or report.get("protocol_config_hash") != expected_protocol_hash
        or report.get("seed") != expected_seed
        or report.get("dataset_evidence") != expected_data
        or report.get("model_integrity", {}).get("manifest_sha256")
        != expected_model_integrity_sha
        or report.get("model_manifest") != expected_model_manifest
    ):
        raise ValueError(f"EvalPlus generation identity mismatch: {path}")
    return report


def _validate_execution_identity(
    path: Path,
    *,
    expected_commit: str,
    expected_protocol_hash: str,
    expected_data: Dict[str, Any],
    expected_generation: Dict[str, Any],
    generation_dir: Path,
    expected_seed: int,
) -> Dict[str, Any]:
    if not _completed_execution(path):
        raise ValueError(f"Unqualified EvalPlus execution: {path}")
    report = _load_json(path / "code_execution_run.json")
    generation_integrity = validate_directory_integrity(
        str(generation_dir),
        expected_kind="code_generation_output",
    )
    if (
        report.get("source", {}).get("commit_sha") != expected_commit
        or report.get("protocol_config_hash") != expected_protocol_hash
        or report.get("seed") != expected_seed
        or Path(str(report.get("generation_dir", ""))).resolve()
        != generation_dir.resolve()
        or report.get("dataset_evidence") != expected_data
        or report.get("generation_report") != expected_generation
        or report.get("generation_integrity", {}).get("manifest_sha256")
        != generation_integrity["manifest_sha256"]
        or report.get("generation_integrity", {}).get("tree_sha256")
        != generation_integrity["tree_sha256"]
        or report.get("sandbox", {}).get("network") != "disabled"
        or not report.get("sandbox", {}).get("unshare_all")
    ):
        raise ValueError(f"EvalPlus execution identity mismatch: {path}")
    return report


def _validate_adapted_codegen(
    generation_dir: Path,
    *,
    run_report: Dict[str, Any],
    run_report_path: Path,
    method: str,
    source_commit: str,
    protocol_hash: str,
    data: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    unvalidated = _load_json(
        generation_dir / "code_generation_run.json"
    )
    model_manifest = unvalidated["model_manifest"]
    if (
        model_manifest.get("run_id") != run_report["metadata"]["run_id"]
        or model_manifest.get("method") != method
        or model_manifest.get("training_seed") != seed
        or model_manifest.get("training_seed")
        != int(run_report["metadata"]["seed"])
        or model_manifest.get("training_model_id")
        != str(run_report["metadata"]["model_id"])
        or model_manifest.get("training_commit_sha") != source_commit
        or model_manifest.get("run_report_sha256")
        != _sha256(run_report_path)
        or model_manifest.get("checkpoint_integrity_manifest_sha256")
        != run_report["checkpoint_integrity"]["manifest_sha256"]
        or model_manifest.get("checkpoint_integrity_tree_sha256")
        != run_report["checkpoint_integrity"]["tree_sha256"]
    ):
        raise ValueError(
            f"Generation/training mismatch: {generation_dir}"
        )
    return _validate_codegen_identity(
        generation_dir,
        expected_role="merged",
        expected_commit=source_commit,
        expected_protocol_hash=protocol_hash,
        expected_data=data,
        expected_model_integrity_sha=unvalidated["model_integrity"][
            "manifest_sha256"
        ],
        expected_model_manifest=model_manifest,
        expected_seed=seed,
    )


def _main_locked() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a completed code SFT matrix with one ephemeral FP32 "
            "merged checkpoint and sandboxed EvalPlus"
        )
    )
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--training-output-root", required=True)
    parser.add_argument("--merged-output-root", required=True)
    parser.add_argument("--generation-output-root", required=True)
    parser.add_argument("--execution-output-root", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--evaluator-path", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    args = parser.parse_args()

    matrix_path = Path(args.matrix).resolve()
    matrix = _load_yaml(matrix_path)
    validate_artifact_retention(matrix.get("artifact_retention", {}))
    if args.stage not in matrix["stages"]:
        raise ValueError(f"Unknown stage: {args.stage}")
    stage = matrix["stages"][args.stage]
    if stage.get("runner", "track_b") != "track_b":
        raise ValueError("Code evaluation supports Track B stages only")

    evaluation_config_path = Path(args.evaluation_config).resolve()
    evaluation_config = load_yaml_config(str(evaluation_config_path))
    validate_formal_code_evaluation_config(evaluation_config)
    protocol = copy.deepcopy(evaluation_config)
    protocol["run"].pop("seed", None)
    protocol_hash = canonical_config_hash(protocol)
    data = validate_evalplus_data_manifest(
        args.dataset_manifest,
        config=evaluation_config,
    )

    repo_root = Path(__file__).resolve().parents[2]
    source_commit = _git_commit(repo_root)
    if _source_dirty(repo_root):
        raise RuntimeError("Formal code evaluation requires a clean source tree")
    scripts = Path(__file__).resolve().parent
    evaluator_path = Path(args.evaluator_path).resolve()
    training_root = Path(args.training_output_root).resolve() / args.stage
    merged_root = Path(args.merged_output_root).resolve() / args.stage
    generation_root = Path(args.generation_output_root).resolve() / args.stage
    execution_root = Path(args.execution_output_root).resolve() / args.stage
    shared_base_root = (
        Path(args.execution_output_root).resolve() / "shared_base"
    )
    for root in (
        merged_root,
        generation_root,
        execution_root,
        shared_base_root,
    ):
        root.mkdir(parents=True, exist_ok=True)
    manifest_path = execution_root / "code_evaluation_launcher_manifest.json"
    jobs = []

    def save_manifest() -> None:
        manifest_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "stage": args.stage,
                    "matrix": str(matrix_path),
                    "source_commit": source_commit,
                    "protocol_config_hash": protocol_hash,
                    "dataset_evidence": data,
                    "jobs": jobs,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    for model_key in stage["models"]:
        model_spec = matrix["models"][model_key]
        model_config_path = (
            matrix_path.parent / model_spec["config"]
        ).resolve()
        model_config = load_yaml_config(str(model_config_path))
        model_path = Path(model_spec["model_path"]).resolve()
        model_manifest_path = model_path / "local_snapshot_manifest.json"
        model_manifest = _load_json(model_manifest_path)
        model_integrity = validate_directory_integrity(
            str(model_path),
            expected_kind="model_snapshot",
        )
        if model_manifest.get("model_id") != model_config["model"]["source_id"]:
            raise ValueError(
                f"Model config/snapshot source mismatch: {model_key}"
            )

        training_runs = {}
        for method in stage["methods"]:
            for seed in stage["seeds"]:
                relative = Path(model_key) / method / f"seed_{seed}"
                training_runs[(method, int(seed))] = _validate_training_run(
                    training_root / relative,
                    expected_method=method,
                    expected_model_id=str(model_config["model"]["id"]),
                    expected_seed=int(seed),
                    expected_commit=source_commit,
                )

        base_root = (
            shared_base_root
            / model_integrity["manifest_sha256"]
            / protocol_hash
            / data["manifest_sha256"]
            / source_commit
        )
        base_generation = base_root / "generation"
        base_execution = base_root / "execution"
        if not _completed_codegen(base_generation, expected_role="base"):
            if base_generation.exists() and any(base_generation.iterdir()):
                raise FileExistsError(
                    f"Incomplete base generation requires audit: {base_generation}"
                )
            _run(
                [
                    sys.executable,
                    str(scripts / "run_evalplus_codegen.py"),
                    "--config",
                    str(evaluation_config_path),
                    "--evaluator-path",
                    str(evaluator_path),
                    "--model-path",
                    str(model_path),
                    "--model-role",
                    "base",
                    "--model-manifest",
                    str(model_manifest_path),
                    "--dataset-manifest",
                    str(Path(args.dataset_manifest).resolve()),
                    "--output-dir",
                    str(base_generation),
                    "--seed",
                    str(stage["seeds"][0]),
                ]
            )
        base_generation_report = _validate_codegen_identity(
            base_generation,
            expected_role="base",
            expected_commit=source_commit,
            expected_protocol_hash=protocol_hash,
            expected_data=data,
            expected_model_integrity_sha=model_integrity["manifest_sha256"],
            expected_model_manifest=model_manifest,
            expected_seed=int(stage["seeds"][0]),
        )
        if not _completed_execution(base_execution):
            if base_execution.exists() and any(base_execution.iterdir()):
                raise FileExistsError(
                    f"Incomplete base execution requires audit: {base_execution}"
                )
            _run(
                [
                    sys.executable,
                    str(scripts / "run_evalplus_sandbox.py"),
                    "--config",
                    str(evaluation_config_path),
                    "--evaluator-path",
                    str(evaluator_path),
                    "--dataset-manifest",
                    str(Path(args.dataset_manifest).resolve()),
                    "--generation-dir",
                    str(base_generation),
                    "--output-dir",
                    str(base_execution),
                    "--seed",
                    str(stage["seeds"][0]),
                ]
            )
        base_execution_report = _validate_execution_identity(
            base_execution,
            expected_commit=source_commit,
            expected_protocol_hash=protocol_hash,
            expected_data=data,
            expected_generation=base_generation_report,
            generation_dir=base_generation,
            expected_seed=int(stage["seeds"][0]),
        )
        jobs.append(
            {
                "model_key": model_key,
                "role": "shared_base",
                "generation_dir": str(base_generation),
                "execution_dir": str(base_execution),
                "metrics": base_execution_report["metrics"],
                "status": "completed",
            }
        )
        save_manifest()

        for method in stage["methods"]:
            for seed in stage["seeds"]:
                relative = Path(model_key) / method / f"seed_{seed}"
                training_dir = training_root / relative
                run_report = training_runs[(method, int(seed))]
                run_report_path = training_dir / "run_report.json"
                merged_dir = merged_root / relative
                generation_dir = generation_root / relative
                execution_dir = execution_root / relative
                job = {
                    "model_key": model_key,
                    "method": method,
                    "seed": seed,
                    "training_dir": str(training_dir),
                    "merged_dir": str(merged_dir),
                    "generation_dir": str(generation_dir),
                    "execution_dir": str(execution_dir),
                    "status": "running",
                }
                jobs.append(job)
                save_manifest()
                _assert_single_merged_slot(
                    merged_root,
                    allowed_path=merged_dir,
                )

                if not _completed_execution(execution_dir):
                    if not _completed_codegen(
                        generation_dir,
                        expected_role="merged",
                    ):
                        if not _completed_export(merged_dir):
                            if merged_dir.exists() and any(merged_dir.iterdir()):
                                raise FileExistsError(
                                    "Incomplete merged output requires audit: "
                                    f"{merged_dir}"
                                )
                            _run(
                                [
                                    sys.executable,
                                    str(scripts / "export_merged.py"),
                                    "--base-model-path",
                                    str(model_path),
                                    "--checkpoint-dir",
                                    str(training_dir / "checkpoint"),
                                    "--run-report",
                                    str(run_report_path),
                                    "--output-dir",
                                    str(merged_dir),
                                    "--requested-id",
                                    str(run_report["model"]["requested_id"]),
                                    "--source-id",
                                    str(run_report["model"]["source_id"]),
                                    "--dtype",
                                    "float32",
                                ]
                            )
                        if generation_dir.exists() and any(
                            generation_dir.iterdir()
                        ):
                            raise FileExistsError(
                                "Incomplete generation requires audit: "
                                f"{generation_dir}"
                            )
                        _run(
                            [
                                sys.executable,
                                str(scripts / "run_evalplus_codegen.py"),
                                "--config",
                                str(evaluation_config_path),
                                "--evaluator-path",
                                str(evaluator_path),
                                "--model-path",
                                str(merged_dir),
                                "--model-role",
                                "merged",
                                "--model-manifest",
                                str(
                                    merged_dir
                                    / "merged_export_manifest.json"
                                ),
                                "--dataset-manifest",
                                str(Path(args.dataset_manifest).resolve()),
                                "--output-dir",
                                str(generation_dir),
                                "--seed",
                                str(seed),
                            ]
                        )
                    generation_report = _validate_adapted_codegen(
                        generation_dir,
                        run_report=run_report,
                        run_report_path=run_report_path,
                        method=method,
                        source_commit=source_commit,
                        protocol_hash=protocol_hash,
                        data=data,
                        seed=int(seed),
                    )
                    if execution_dir.exists() and any(
                        execution_dir.iterdir()
                    ):
                        raise FileExistsError(
                            "Incomplete execution requires audit: "
                            f"{execution_dir}"
                        )
                    _run(
                        [
                            sys.executable,
                            str(scripts / "run_evalplus_sandbox.py"),
                            "--config",
                            str(evaluation_config_path),
                            "--evaluator-path",
                            str(evaluator_path),
                            "--dataset-manifest",
                            str(Path(args.dataset_manifest).resolve()),
                            "--generation-dir",
                            str(generation_dir),
                            "--output-dir",
                            str(execution_dir),
                            "--seed",
                            str(seed),
                        ]
                    )
                generation_report = _validate_adapted_codegen(
                    generation_dir,
                    run_report=run_report,
                    run_report_path=run_report_path,
                    method=method,
                    source_commit=source_commit,
                    protocol_hash=protocol_hash,
                    data=data,
                    seed=int(seed),
                )
                execution_report = _validate_execution_identity(
                    execution_dir,
                    expected_commit=source_commit,
                    expected_protocol_hash=protocol_hash,
                    expected_data=data,
                    expected_generation=generation_report,
                    generation_dir=generation_dir,
                    expected_seed=int(seed),
                )
                job["status"] = "completed"
                job["metrics"] = execution_report["metrics"]
                job["execution_report_sha256"] = _sha256(
                    execution_dir / "code_execution_run.json"
                )
                _delete_ephemeral_merged(
                    merged_dir,
                    merged_root=merged_root,
                )
                job["merged_checkpoint_retention"] = (
                    "deleted_after_integrity_sealed_sandbox_evaluation"
                )
                save_manifest()

    save_manifest()
    print(manifest_path.read_text(encoding="utf-8"), end="")


def main() -> None:
    with project_file_lock("/tmp/llm_lora_nf_evaluation_pipeline.lock"):
        _main_locked()


if __name__ == "__main__":
    main()
