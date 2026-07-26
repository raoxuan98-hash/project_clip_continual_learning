#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from llm_lora_nf.artifact_retention import validate_artifact_retention
from llm_lora_nf.config_io import load_yaml_config
from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.resource_guard import project_file_lock


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


def _load(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _completed_formal_run(
    output_dir: Path,
    *,
    expected_method: str,
    expected_model_id: str,
    expected_seed: int,
    expected_commit: str,
) -> bool:
    report_path = output_dir / "run_report.json"
    if not report_path.exists():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = report.get("metadata", {})
    completed = bool(
        report.get("status") == "trained"
        and report.get("formal_result_eligible", False)
        and not report.get("source_dirty")
        and metadata.get("execution_mode") == "gpu_formal"
        and metadata.get("method") == expected_method
        and metadata.get("model_id") == expected_model_id
        and metadata.get("seed") == expected_seed
        and metadata.get("commit_sha") == expected_commit
    )
    if completed:
        validate_artifact_retention(report.get("artifact_retention", {}))
        checkpoint_integrity = validate_directory_integrity(
            str(output_dir / "checkpoint"),
            expected_kind="adapter_checkpoint",
        )
        recorded_integrity = report.get("checkpoint_integrity", {})
        if (
            checkpoint_integrity["manifest_sha256"]
            != recorded_integrity.get("manifest_sha256")
            or checkpoint_integrity["tree_sha256"]
            != recorded_integrity.get("tree_sha256")
        ):
            raise ValueError(
                f"Run report/checkpoint identity mismatch: {output_dir}"
            )
    return completed


def _main_locked() -> None:
    parser = argparse.ArgumentParser(
        description="Serial, resumable launcher for a locked SFT matrix"
    )
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    matrix_path = Path(args.matrix).resolve()
    matrix = _load(matrix_path)
    validate_artifact_retention(matrix.get("artifact_retention", {}))
    if args.stage not in matrix["stages"]:
        raise ValueError(f"Unknown stage: {args.stage}")
    stage = matrix["stages"][args.stage]
    runner = str(stage.get("runner", "track_b"))
    if runner not in {"track_a", "track_b"}:
        raise ValueError(f"Unsupported matrix runner: {runner}")
    output_root = Path(args.output_root).resolve() / args.stage
    script_root = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]
    source_commit = _git_commit(repo_root)
    if _source_dirty(repo_root):
        raise RuntimeError("Formal matrix launch requires a clean source tree")
    output_root.mkdir(parents=True, exist_ok=True)
    config_root = matrix_path.parent
    jobs = []

    for model_key in stage["models"]:
        model_spec = matrix["models"][model_key]
        model_config = load_yaml_config(
            str((config_root / model_spec["config"]).resolve())
        )
        for method in stage["methods"]:
            for seed in stage["seeds"]:
                output_dir = output_root / model_key / method / f"seed_{seed}"
                run_name = f"{args.stage}_{model_key}_{method}_seed{seed}"
                if _completed_formal_run(
                    output_dir,
                    expected_method=method,
                    expected_model_id=str(model_config["model"]["id"]),
                    expected_seed=int(seed),
                    expected_commit=source_commit,
                ):
                    jobs.append(
                        {
                            "run_name": run_name,
                            "status": "skipped_completed",
                            "output_dir": str(output_dir),
                        }
                    )
                    continue
                if output_dir.exists() and any(output_dir.iterdir()):
                    raise FileExistsError(
                        f"Incomplete/non-formal output requires manual audit: {output_dir}"
                    )
                command = [
                    sys.executable,
                    str(
                        script_root
                        / (
                            "run_track_a_sft.py"
                            if runner == "track_a"
                            else "run_sft.py"
                        )
                    ),
                    "--config",
                    str((config_root / model_spec["config"]).resolve()),
                    "--model-path",
                    str(Path(model_spec["model_path"]).resolve()),
                ]
                if runner == "track_a":
                    command.extend(
                        [
                            "--metamath-json",
                            str(
                                Path(
                                    matrix["data"]["metamath_json"]
                                ).resolve()
                            ),
                        ]
                    )
                else:
                    train_json = matrix["data"].get(
                        "train_json",
                        matrix["data"].get("metamath_json"),
                    )
                    if train_json is None:
                        raise ValueError(
                            "Track B matrix data requires train_json"
                        )
                    command.extend(
                        [
                            "--train-json",
                            str(Path(train_json).resolve()),
                        ]
                    )
                command.extend(
                    [
                        "--nq-parquet",
                        str(Path(matrix["data"]["nq_parquet"]).resolve()),
                        "--output-dir",
                        str(output_dir),
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                        "--run-name",
                        run_name,
                    ]
                )
                if runner == "track_b":
                    cache_root = matrix["data"].get("calibration_cache_root")
                    if cache_root:
                        command.extend(
                            [
                                "--calibration-cache-dir",
                                str(Path(cache_root).resolve()),
                            ]
                        )
                record = {
                    "run_name": run_name,
                    "status": "running",
                    "output_dir": str(output_dir),
                    "command": command,
                }
                jobs.append(record)
                manifest = {
                    "format_version": 1,
                    "stage": args.stage,
                    "runner": runner,
                    "matrix": str(matrix_path),
                    "source_commit": source_commit,
                    "jobs": jobs,
                }
                (output_root / "launcher_manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                completed = subprocess.run(command, check=False)
                record["returncode"] = completed.returncode
                record["status"] = (
                    "completed" if completed.returncode == 0 else "failed"
                )
                if completed.returncode == 0 and not _completed_formal_run(
                    output_dir,
                    expected_method=method,
                    expected_model_id=str(model_config["model"]["id"]),
                    expected_seed=int(seed),
                    expected_commit=source_commit,
                ):
                    record["status"] = "failed_artifact_validation"
                    record["returncode"] = 1
                    completed = subprocess.CompletedProcess(
                        completed.args,
                        returncode=1,
                    )
                (output_root / "launcher_manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if completed.returncode != 0:
                    raise SystemExit(completed.returncode)

    manifest = {
        "format_version": 1,
        "stage": args.stage,
        "runner": runner,
        "matrix": str(matrix_path),
        "source_commit": source_commit,
        "jobs": jobs,
    }
    (output_root / "launcher_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    with project_file_lock("/tmp/llm_lora_nf_training_pipeline.lock"):
        _main_locked()


if __name__ == "__main__":
    main()
