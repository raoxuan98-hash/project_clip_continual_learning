#!/usr/bin/env python3
"""Resumable export, lm-eval, and per-run aggregation for a matrix stage."""

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from llm_lora_nf.artifact_retention import validate_artifact_retention
from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.dataset_evidence import validate_eval_data_manifest
from llm_lora_nf.environment import (
    execution_environment_identity,
    software_environment_identity,
)
from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.protocol_validation import validate_formal_evaluation_config
from llm_lora_nf.resource_guard import project_file_lock


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
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


def _parse_model_paths(values) -> Dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--base-run must be MODEL_KEY=RUN_DIR")
        model_key, path = value.split("=", 1)
        if not model_key or model_key in parsed:
            raise ValueError(f"Invalid or duplicate base-run model key: {model_key}")
        parsed[model_key] = Path(path).resolve()
    return parsed


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_single_merged_slot(
    merged_root: Path,
    *,
    allowed_path: Path,
) -> None:
    root = merged_root.resolve()
    allowed = allowed_path.resolve()
    if not _is_within(allowed, root) or allowed == root:
        raise ValueError("Allowed merged checkpoint must be below its root")
    foreign_files = []
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file() and not _is_within(path.resolve(), allowed):
                foreign_files.append(path.resolve())
                if len(foreign_files) == 3:
                    break
    if foreign_files:
        raise RuntimeError(
            "The one-checkpoint merged slot is occupied outside the current "
            f"target; audit these files first: {foreign_files}"
        )


def _completed_export(path: Path) -> bool:
    manifest_path = path / "merged_export_manifest.json"
    integrity_path = path / "artifact_integrity.json"
    if not manifest_path.exists() or not integrity_path.exists():
        return False
    validate_directory_integrity(
        str(path),
        expected_kind="merged_checkpoint",
    )
    return bool(_load_json(manifest_path).get("formal_result_eligible"))


def _completed_evaluation(
    path: Path,
    *,
    expected_role: str = "merged",
) -> bool:
    manifest_path = path / "evaluation_run.json"
    integrity_path = path / "artifact_integrity.json"
    if not manifest_path.exists() or not integrity_path.exists():
        return False
    validate_directory_integrity(
        str(path),
        expected_kind="evaluation_output",
    )
    manifest = _load_json(manifest_path)
    return bool(
        manifest.get("format_version") == 2
        and manifest.get("formal_result_eligible")
        and manifest.get("status") == "passed"
        and manifest.get("returncode") == 0
        and manifest.get("execution_mode") == "gpu_formal"
        and manifest.get("limit") is None
        and manifest.get("model_role") == expected_role
        and bool(manifest.get("source", {}).get("commit_sha"))
        and not manifest.get("source", {}).get("source_dirty")
    )


def _validate_training_run(
    training_dir: Path,
    *,
    expected_method: str,
    expected_model_id: str,
    expected_seed: int,
    expected_commit: str,
) -> Dict[str, Any]:
    run_report_path = training_dir / "run_report.json"
    run_report = _load_json(run_report_path)
    metadata = run_report.get("metadata", {})
    if (
        not run_report.get("formal_result_eligible")
        or run_report.get("status") != "trained"
        or run_report.get("source_dirty")
        or metadata.get("execution_mode") != "gpu_formal"
        or metadata.get("method") != expected_method
        or metadata.get("model_id") != expected_model_id
        or metadata.get("seed") != expected_seed
        or metadata.get("commit_sha") != expected_commit
    ):
        raise ValueError(f"Unqualified or mismatched training run: {training_dir}")
    validate_artifact_retention(run_report.get("artifact_retention", {}))
    checkpoint_integrity = validate_directory_integrity(
        str(training_dir / "checkpoint"),
        expected_kind="adapter_checkpoint",
    )
    recorded_integrity = run_report.get("checkpoint_integrity", {})
    if (
        checkpoint_integrity["manifest_sha256"]
        != recorded_integrity.get("manifest_sha256")
        or checkpoint_integrity["tree_sha256"]
        != recorded_integrity.get("tree_sha256")
    ):
        raise ValueError(
            f"Training report/checkpoint identity mismatch: {training_dir}"
        )
    return run_report


def _validate_base_evaluation_identity(
    path: Path,
    *,
    model_path: Path,
    model_manifest: Dict[str, Any],
    model_integrity: Dict[str, Any],
    dataset_evidence: Dict[str, Any],
    evaluation_config: Dict[str, Any],
    protocol_config_hash: str,
    source_commit: str,
) -> Dict[str, Any]:
    if not _completed_evaluation(path, expected_role="base"):
        raise ValueError(f"Unqualified base evaluation: {path}")
    evaluation = _load_json(path / "evaluation_run.json")
    expected_seeds = {
        key: int(evaluation_config["evaluator"]["seeds"][key])
        for key in ("random", "numpy", "torch", "fewshot")
    }
    embedded_model = evaluation.get("model_manifest")
    request_cache = evaluation.get("request_cache", {})
    request_identity = request_cache.get("identity", {})
    if (
        embedded_model != model_manifest
        or evaluation.get("model_path") != str(model_path)
        or evaluation.get("model_manifest_sha256")
        != _sha256(model_path / "local_snapshot_manifest.json")
        or evaluation.get("model_integrity", {}).get("manifest_sha256")
        != model_integrity.get("manifest_sha256")
        or evaluation.get("model_integrity", {}).get("tree_sha256")
        != model_integrity.get("tree_sha256")
        or evaluation.get("dataset_evidence") != dataset_evidence
        or evaluation.get("evaluator_revision")
        != evaluation_config["evaluator"]["revision"]
        or evaluation.get("evaluator_seeds") != expected_seeds
        or evaluation.get("protocol_config_hash") != protocol_config_hash
        or evaluation.get("source", {}).get("commit_sha") != source_commit
        or evaluation.get("source", {}).get("source_dirty")
        or request_identity.get("base_snapshot_integrity_sha256")
        != model_integrity.get("manifest_sha256")
        or request_identity.get("evaluator_seeds") != expected_seeds
        or request_identity.get("software_environment")
        != software_environment_identity(evaluation.get("environment", {}))
        or not evaluation.get("task_definitions", {}).get("tree_sha256")
        or not request_cache.get("integrity", {}).get("tree_sha256")
    ):
        raise ValueError(
            f"Base evaluation/model/protocol identity mismatch: {path}"
        )
    return evaluation


def _validate_export_identity(
    path: Path,
    *,
    run_report_path: Path,
    run_report: Dict[str, Any],
    base_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    if not _completed_export(path):
        raise ValueError(f"Unqualified merged export: {path}")
    export = _load_json(path / "merged_export_manifest.json")
    metadata = run_report["metadata"]
    base_model = export.get("base_model", {})
    if (
        export.get("format_version") != 1
        or not export.get("formal_result_eligible")
        or export.get("execution_mode") != "gpu_formal"
        or export.get("method") != metadata["method"]
        or export.get("run_id") != metadata["run_id"]
        or export.get("training_commit_sha") != metadata["commit_sha"]
        or export.get("export_commit_sha") != metadata["commit_sha"]
        or export.get("export_source_dirty")
        or export.get("dtype") != "torch.float32"
        or export.get("checkpoint_retention")
        != "ephemeral_delete_after_qualified_evaluation"
        or export.get("run_report_sha256") != _sha256(run_report_path)
        or export.get("training_environment_identity")
        != execution_environment_identity(run_report["environment"])
        or base_model.get("snapshot_integrity_sha256")
        != base_manifest.get("model_integrity", {}).get("manifest_sha256")
        or base_model.get("snapshot_manifest_sha256")
        != base_manifest.get("model_manifest_sha256")
    ):
        raise ValueError(
            f"Merged export does not match its training/base evidence: {path}"
        )
    return export


def _validate_evaluation_identity(
    path: Path,
    *,
    run_report_path: Path,
    run_report: Dict[str, Any],
    base_manifest: Dict[str, Any],
) -> None:
    if not _completed_evaluation(path, expected_role="merged"):
        raise ValueError(f"Unqualified formal evaluation: {path}")
    evaluation = _load_json(path / "evaluation_run.json")
    export = evaluation.get("model_manifest")
    if not isinstance(export, dict):
        raise ValueError(f"Evaluation lacks embedded export manifest: {path}")
    metadata = run_report["metadata"]
    base_model = export.get("base_model", {})
    if (
        export.get("format_version") != 1
        or not export.get("formal_result_eligible")
        or export.get("execution_mode") != "gpu_formal"
        or export.get("method") != metadata["method"]
        or export.get("run_id") != metadata["run_id"]
        or export.get("training_commit_sha") != metadata["commit_sha"]
        or export.get("export_commit_sha") != metadata["commit_sha"]
        or export.get("export_source_dirty")
        or export.get("dtype") != "torch.float32"
        or export.get("checkpoint_retention")
        != "ephemeral_delete_after_qualified_evaluation"
        or export.get("run_report_sha256") != _sha256(run_report_path)
        or export.get("training_environment_identity")
        != execution_environment_identity(run_report["environment"])
        or base_model.get("snapshot_integrity_sha256")
        != base_manifest.get("model_integrity", {}).get("manifest_sha256")
        or base_model.get("snapshot_manifest_sha256")
        != base_manifest.get("model_manifest_sha256")
        or evaluation.get("dataset_evidence")
        != base_manifest.get("dataset_evidence")
        or evaluation.get("task_definitions", {}).get("tree_sha256")
        != base_manifest.get("task_definitions", {}).get("tree_sha256")
        or evaluation.get("request_cache", {}).get("key")
        != base_manifest.get("request_cache", {}).get("key")
        or evaluation.get("request_cache", {})
        .get("integrity", {})
        .get("tree_sha256")
        != base_manifest.get("request_cache", {})
        .get("integrity", {})
        .get("tree_sha256")
        or evaluation.get("source", {}).get("commit_sha")
        != metadata["commit_sha"]
        or evaluation.get("source", {}).get("source_dirty")
    ):
        raise ValueError(
            f"Evaluation does not match its training/base evidence: {path}"
        )


def _validate_summary_identity(
    path: Path,
    *,
    evaluation_dir: Path,
    base_run: Path,
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Aggregation did not create its declared summary: {path}"
        )
    summary = _load_json(path)
    evaluation_manifest_path = evaluation_dir / "evaluation_run.json"
    base_manifest_path = base_run / "evaluation_run.json"
    evaluation = _load_json(evaluation_manifest_path)
    base_evaluation = _load_json(base_manifest_path)
    evaluation_integrity = validate_directory_integrity(
        str(evaluation_dir),
        expected_kind="evaluation_output",
    )
    base_integrity = validate_directory_integrity(
        str(base_run),
        expected_kind="evaluation_output",
    )
    comparison = summary.get("base_comparison")
    model_identity = summary.get("model_identity")
    base_identity = (
        comparison.get("base_model_identity")
        if isinstance(comparison, dict)
        else None
    )
    if (
        summary.get("format_version") != 2
        or not summary.get("formal_result_eligible")
        or summary.get("run_manifest") != str(evaluation_manifest_path)
        or summary.get("run_manifest_sha256")
        != _sha256(evaluation_manifest_path)
        or summary.get("evaluation_output_integrity")
        != evaluation_integrity
        or not isinstance(comparison, dict)
        or comparison.get("base_run_manifest") != str(base_manifest_path)
        or comparison.get("base_run_manifest_sha256")
        != _sha256(base_manifest_path)
        or comparison.get("base_evaluation_output_integrity")
        != base_integrity
        or not isinstance(model_identity, dict)
        or model_identity.get("evaluated_checkpoint_integrity_sha256")
        != evaluation.get("model_integrity", {}).get("manifest_sha256")
        or model_identity.get("base_snapshot_integrity_sha256")
        != base_evaluation.get("model_integrity", {}).get("manifest_sha256")
        or not isinstance(base_identity, dict)
        or base_identity.get("evaluated_checkpoint_integrity_sha256")
        != base_evaluation.get("model_integrity", {}).get("manifest_sha256")
        or summary.get("dataset_evidence")
        != evaluation.get("dataset_evidence")
        or summary.get("task_definition_tree_sha256")
        != evaluation.get("task_definitions", {}).get("tree_sha256")
        or summary.get("evaluator_seeds")
        != evaluation.get("evaluator_seeds")
        or summary.get("request_cache_key")
        != evaluation.get("request_cache", {}).get("key")
        or summary.get("request_cache_tree_sha256")
        != evaluation.get("request_cache", {})
        .get("integrity", {})
        .get("tree_sha256")
        or not isinstance(summary.get("aggregation_source"), dict)
        or summary.get("aggregation_source")
        != evaluation.get("source")
        or not isinstance(
            summary.get("aggregation_software_environment"),
            dict,
        )
        or summary.get("aggregation_software_environment")
        != evaluation.get("request_cache", {})
        .get("identity", {})
        .get("software_environment")
        or base_evaluation.get("source") != evaluation.get("source")
        or base_evaluation.get("evaluator_revision")
        != evaluation.get("evaluator_revision")
        or base_evaluation.get("evaluator_seeds")
        != evaluation.get("evaluator_seeds")
        or base_evaluation.get("tasks") != evaluation.get("tasks")
        or base_evaluation.get("dataset_evidence")
        != evaluation.get("dataset_evidence")
        or base_evaluation.get("protocol_config_hash")
        != evaluation.get("protocol_config_hash")
        or base_evaluation.get("task_definitions", {}).get("tree_sha256")
        != evaluation.get("task_definitions", {}).get("tree_sha256")
        or base_evaluation.get("request_cache", {}).get("key")
        != evaluation.get("request_cache", {}).get("key")
        or base_evaluation.get("request_cache", {})
        .get("integrity", {})
        .get("tree_sha256")
        != evaluation.get("request_cache", {})
        .get("integrity", {})
        .get("tree_sha256")
    ):
        raise ValueError(
            "Aggregation summary does not bind the exact qualified "
            f"evaluation/base evidence: {path}"
        )
    return summary


def _run(command) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _delete_ephemeral_merged(path: Path, *, merged_root: Path) -> None:
    target = path.resolve()
    root = merged_root.resolve()
    if target == root:
        raise ValueError("Refusing to delete the merged-checkpoint root")
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Ephemeral merged checkpoint escapes its output root: {target}"
        ) from error
    if not target.exists():
        return
    if not _completed_export(target):
        raise ValueError(
            f"Refusing to delete an unqualified merged checkpoint: {target}"
        )
    shutil.rmtree(target)


def _main_locked() -> None:
    parser = argparse.ArgumentParser(
        description="Export and evaluate every completed formal matrix checkpoint"
    )
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--training-output-root", required=True)
    parser.add_argument("--merged-output-root", required=True)
    parser.add_argument("--evaluation-output-root", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--evaluator-path", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--request-cache-root", required=True)
    parser.add_argument(
        "--base-run",
        action="append",
        default=[],
        help=(
            "Optional MODEL_KEY=qualified base evaluation override; otherwise "
            "a content-addressed shared base run is created or reused."
        ),
    )
    args = parser.parse_args()

    matrix_path = Path(args.matrix).resolve()
    matrix = _load_yaml(matrix_path)
    validate_artifact_retention(matrix.get("artifact_retention", {}))
    evaluation_config = load_yaml_config(
        str(Path(args.evaluation_config).resolve())
    )
    validate_formal_evaluation_config(evaluation_config)
    protocol_config = copy.deepcopy(evaluation_config)
    protocol_config["run"].pop("seed", None)
    protocol_config["run"].pop("name", None)
    protocol_config_hash = canonical_config_hash(protocol_config)
    dataset_evidence = validate_eval_data_manifest(
        args.dataset_manifest,
        config=evaluation_config,
        hf_home=str(Path(args.hf_home).resolve()),
    )
    if args.stage not in matrix["stages"]:
        raise ValueError(f"Unknown stage: {args.stage}")
    stage = matrix["stages"][args.stage]
    base_runs = _parse_model_paths(args.base_run)
    training_root = Path(args.training_output_root).resolve() / args.stage
    merged_root = Path(args.merged_output_root).resolve() / args.stage
    all_evaluation_root = Path(args.evaluation_output_root).resolve()
    evaluation_root = all_evaluation_root / args.stage
    shared_base_root = all_evaluation_root / "shared_base"
    scripts = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]
    source_commit = _git_commit(repo_root)
    if _source_dirty(repo_root):
        raise RuntimeError("Formal evaluation launch requires a clean source tree")
    merged_root.mkdir(parents=True, exist_ok=True)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    shared_base_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    manifest_path = evaluation_root / "evaluation_launcher_manifest.json"

    def save_manifest() -> None:
        manifest_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "stage": args.stage,
                    "matrix": str(matrix_path),
                    "source_commit": source_commit,
                    "protocol_config_hash": protocol_config_hash,
                    "dataset_evidence": dataset_evidence,
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
        model_config = load_yaml_config(
            str((matrix_path.parent / model_spec["config"]).resolve())
        )
        model_path = Path(model_spec["model_path"]).resolve()
        model_manifest_path = model_path / "local_snapshot_manifest.json"
        model_manifest = _load_json(model_manifest_path)
        model_integrity = validate_directory_integrity(
            str(model_path),
            expected_kind="model_snapshot",
        )
        if model_manifest.get("model_id") != model_config["model"]["source_id"]:
            raise ValueError(
                f"Model config/snapshot source identity mismatch: {model_key}"
            )
        validated_training_runs = {}
        for method in stage["methods"]:
            for seed in stage["seeds"]:
                relative = Path(model_key) / method / f"seed_{seed}"
                training_dir = training_root / relative
                validated_training_runs[(method, int(seed))] = (
                    _validate_training_run(
                        training_dir,
                        expected_method=method,
                        expected_model_id=str(model_config["model"]["id"]),
                        expected_seed=int(seed),
                        expected_commit=source_commit,
                    )
                )
        base_run = base_runs.get(model_key)
        if base_run is None:
            base_run = (
                shared_base_root
                / model_integrity["manifest_sha256"]
                / protocol_config_hash
                / dataset_evidence["manifest_sha256"]
                / source_commit
            )
            if not _completed_evaluation(base_run, expected_role="base"):
                if base_run.exists() and any(base_run.iterdir()):
                    raise FileExistsError(
                        f"Incomplete shared base evaluation requires audit: {base_run}"
                    )
                _run(
                    [
                        sys.executable,
                        str(scripts / "run_lm_eval.py"),
                        "--config",
                        str(Path(args.evaluation_config).resolve()),
                        "--evaluator-path",
                        str(Path(args.evaluator_path).resolve()),
                        "--model-path",
                        str(model_path),
                        "--model-role",
                        "base",
                        "--model-manifest",
                        str(model_manifest_path),
                        "--hf-home",
                        str(Path(args.hf_home).resolve()),
                        "--dataset-manifest",
                        str(Path(args.dataset_manifest).resolve()),
                        "--request-cache-root",
                        str(Path(args.request_cache_root).resolve()),
                        "--output-dir",
                        str(base_run),
                        "--seed",
                        str(stage["seeds"][0]),
                        "--run-name",
                        f"shared_base_{model_key}",
                    ]
                )
        base_manifest = _validate_base_evaluation_identity(
            base_run,
            model_path=model_path,
            model_manifest=model_manifest,
            model_integrity=model_integrity,
            dataset_evidence=dataset_evidence,
            evaluation_config=evaluation_config,
            protocol_config_hash=protocol_config_hash,
            source_commit=source_commit,
        )
        jobs.append(
            {
                "model_key": model_key,
                "role": "shared_base",
                "evaluation_dir": str(base_run),
                "model_snapshot_integrity_sha256": model_integrity[
                    "manifest_sha256"
                ],
                "protocol_config_hash": protocol_config_hash,
                "dataset_manifest_sha256": dataset_evidence["manifest_sha256"],
                "source_commit": source_commit,
                "status": "completed",
            }
        )
        save_manifest()
        for method in stage["methods"]:
            for seed in stage["seeds"]:
                relative = Path(model_key) / method / f"seed_{seed}"
                training_dir = training_root / relative
                run_report_path = training_dir / "run_report.json"
                run_report = validated_training_runs[(method, int(seed))]
                merged_dir = merged_root / relative
                evaluation_dir = evaluation_root / relative
                summary_path = (
                    evaluation_root
                    / "summaries"
                    / relative
                    / "evaluation_summary.json"
                )
                job = {
                    "model_key": model_key,
                    "method": method,
                    "seed": seed,
                    "training_dir": str(training_dir),
                    "merged_dir": str(merged_dir),
                    "evaluation_dir": str(evaluation_dir),
                    "status": "running",
                }
                jobs.append(job)
                save_manifest()

                _assert_single_merged_slot(
                    merged_root,
                    allowed_path=merged_dir,
                )
                evaluation_complete = _completed_evaluation(
                    evaluation_dir,
                    expected_role="merged",
                )
                if not evaluation_complete:
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
                    _validate_export_identity(
                        merged_dir,
                        run_report_path=run_report_path,
                        run_report=run_report,
                        base_manifest=base_manifest,
                    )
                    if evaluation_dir.exists() and any(evaluation_dir.iterdir()):
                        raise FileExistsError(
                            "Incomplete evaluation output requires audit: "
                            f"{evaluation_dir}"
                        )
                    _run(
                        [
                            sys.executable,
                            str(scripts / "run_lm_eval.py"),
                            "--config",
                            str(Path(args.evaluation_config).resolve()),
                            "--evaluator-path",
                            str(Path(args.evaluator_path).resolve()),
                            "--model-path",
                            str(merged_dir),
                            "--model-role",
                            "merged",
                            "--model-manifest",
                            str(merged_dir / "merged_export_manifest.json"),
                            "--hf-home",
                            str(Path(args.hf_home).resolve()),
                            "--dataset-manifest",
                            str(Path(args.dataset_manifest).resolve()),
                            "--request-cache-root",
                            str(Path(args.request_cache_root).resolve()),
                            "--output-dir",
                            str(evaluation_dir),
                            "--seed",
                            str(seed),
                            "--run-name",
                            f"{args.stage}_{model_key}_{method}_seed{seed}_eval",
                        ]
                    )
                _validate_evaluation_identity(
                    evaluation_dir,
                    run_report_path=run_report_path,
                    run_report=run_report,
                    base_manifest=base_manifest,
                )

                _run(
                    [
                        sys.executable,
                        str(scripts / "aggregate_lm_eval.py"),
                        "--config",
                        str(Path(args.evaluation_config).resolve()),
                        "--run-dir",
                        str(evaluation_dir),
                        "--base-run-dir",
                        str(base_run),
                        "--output",
                        str(summary_path),
                    ]
                )
                _validate_summary_identity(
                    summary_path,
                    evaluation_dir=evaluation_dir,
                    base_run=base_run,
                )
                job["status"] = "completed"
                job["summary_path"] = str(summary_path)
                job["summary_sha256"] = _sha256(summary_path)
                _delete_ephemeral_merged(
                    merged_dir,
                    merged_root=merged_root,
                )
                job["merged_checkpoint_retention"] = (
                    "deleted_after_integrity_sealed_evaluation_and_aggregation"
                )
                save_manifest()

    save_manifest()
    print(manifest_path.read_text(encoding="utf-8"), end="")


def main() -> None:
    with project_file_lock("/tmp/llm_lora_nf_evaluation_pipeline.lock"):
        _main_locked()


if __name__ == "__main__":
    main()
