#!/usr/bin/env python3
"""Paired sample-level bootstrap for two qualified lm-eval runs."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.environment import (
    runtime_environment,
    software_environment_identity,
)
from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.provenance import (
    git_source_state,
    validate_derivation_source,
)
from llm_lora_nf.protocol_validation import validate_formal_evaluation_config


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise ValueError(f"Sample log is empty: {path}")
    return rows


def _sample_file(run_dir: Path, task: str) -> Path:
    matches = list(run_dir.rglob(f"samples_{task}_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one sample log for {task} in {run_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _metric_and_filter(requested: str) -> Tuple[str, str]:
    if "," not in requested:
        return requested, "none"
    metric, filter_name = requested.split(",", 1)
    return metric, filter_name


def _paired_values(
    reference_path: Path,
    comparison_path: Path,
    *,
    metric: str,
    filter_name: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    def selected(path: Path) -> Dict[int, Dict[str, Any]]:
        rows = [
            row
            for row in _load_jsonl(path)
            if str(row["filter"]) == filter_name
        ]
        by_doc = {int(row["doc_id"]): row for row in rows}
        if len(by_doc) != len(rows):
            raise ValueError(f"Duplicate doc_id/filter rows in {path}")
        if not by_doc:
            raise ValueError(
                f"No filter {filter_name!r} samples found in {path}"
            )
        return by_doc

    reference = selected(reference_path)
    comparison = selected(comparison_path)
    if set(reference) != set(comparison):
        raise ValueError("Paired sample logs have different document IDs")
    audit = []
    reference_values = []
    comparison_values = []
    for doc_id in sorted(reference):
        ref = reference[doc_id]
        cmp = comparison[doc_id]
        for hash_key in ("doc_hash", "prompt_hash", "target_hash"):
            if ref[hash_key] != cmp[hash_key]:
                raise ValueError(
                    f"Sample {doc_id} has mismatched {hash_key}"
                )
        if metric not in ref or metric not in cmp:
            raise KeyError(
                f"Metric {metric!r} missing for document {doc_id}"
            )
        reference_values.append(float(ref[metric]))
        comparison_values.append(float(cmp[metric]))
        audit.append(
            {
                "doc_id": doc_id,
                "doc_hash": ref["doc_hash"],
                "prompt_hash": ref["prompt_hash"],
                "target_hash": ref["target_hash"],
            }
        )
    return (
        np.asarray(reference_values, dtype=np.float64),
        np.asarray(comparison_values, dtype=np.float64),
        audit,
    )


def _bootstrap_means(
    reference: np.ndarray,
    comparison: np.ndarray,
    *,
    samples: int,
    seed: int,
    chunk_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    if reference.shape != comparison.shape or reference.ndim != 1:
        raise ValueError("Paired samples must be same-shape one-dimensional arrays")
    if reference.size == 0 or samples <= 0:
        raise ValueError("Paired bootstrap requires data and positive samples")
    generator = np.random.default_rng(seed)
    reference_means = np.empty(samples, dtype=np.float64)
    comparison_means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = generator.integers(
            0,
            reference.size,
            size=(stop - start, reference.size),
        )
        reference_means[start:stop] = reference[indices].mean(axis=1)
        comparison_means[start:stop] = comparison[indices].mean(axis=1)
    return reference_means, comparison_means


def _effect(
    reference_observed: float,
    comparison_observed: float,
    reference_bootstrap: np.ndarray,
    comparison_bootstrap: np.ndarray,
) -> Dict[str, Any]:
    differences = reference_bootstrap - comparison_bootstrap
    return {
        "reference_score": float(reference_observed),
        "comparison_score": float(comparison_observed),
        "difference_definition": "reference_minus_comparison",
        "observed_difference": float(reference_observed - comparison_observed),
        "paired_bootstrap_mean_difference": float(differences.mean()),
        "paired_bootstrap_95ci": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "probability_reference_greater": float(np.mean(differences > 0.0)),
    }


def _validated_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "evaluation_run.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("format_version") != 2
        or not manifest.get("formal_result_eligible")
        or manifest.get("status") != "passed"
        or manifest.get("returncode") != 0
        or manifest.get("execution_mode") != "gpu_formal"
        or manifest.get("limit") is not None
        or not manifest.get("source", {}).get("commit_sha")
        or manifest.get("source", {}).get("source_dirty")
    ):
        raise ValueError(f"Sample bootstrap requires a complete formal run: {run_dir}")
    output_integrity = validate_directory_integrity(
        str(run_dir),
        expected_kind="evaluation_output",
    )
    manifest["_validated_output_integrity"] = output_integrity
    return manifest


def _method_identity(manifest: Dict[str, Any]) -> Dict[str, str]:
    if (
        manifest.get("model_role") != "merged"
        or manifest.get("model_integrity", {}).get("kind")
        != "merged_checkpoint"
    ):
        raise ValueError("Paired bootstrap requires merged adapter evaluations")
    model_manifest = manifest.get("model_manifest")
    if not isinstance(model_manifest, dict):
        raise ValueError(
            "Formal paired bootstrap requires an embedded model manifest"
        )
    if "method" not in model_manifest:
        raise ValueError("Paired bootstrap requires merged adapter models")
    if (
        not model_manifest.get("formal_result_eligible")
        or model_manifest.get("execution_mode") != "gpu_formal"
        or model_manifest.get("export_source_dirty")
        or model_manifest.get("export_commit_sha")
        != manifest.get("source", {}).get("commit_sha")
        or model_manifest.get("training_commit_sha")
        != model_manifest.get("export_commit_sha")
        or model_manifest.get("dtype") != "torch.float32"
        or model_manifest.get("checkpoint_retention")
        != "ephemeral_delete_after_qualified_evaluation"
    ):
        raise ValueError("Paired bootstrap requires a qualified merged export")
    base_snapshot_integrity = model_manifest["base_model"].get(
        "snapshot_integrity_sha256"
    )
    if (
        not base_snapshot_integrity
        or model_manifest.get("base_model_integrity_manifest_sha256")
        != base_snapshot_integrity
    ):
        raise ValueError(
            "Paired bootstrap requires a validated base snapshot identity"
        )
    return {
        "method": str(model_manifest["method"]),
        "protocol_track": str(model_manifest.get("protocol_track", "B")),
        "model_id": str(model_manifest["base_model"]["requested_id"]),
        "training_commit": str(model_manifest["training_commit_sha"]),
        "training_comparison_config_hash": str(
            model_manifest["training_comparison_config_hash"]
        ),
        "training_environment_identity": model_manifest.get(
            "training_environment_identity"
        ),
        "base_snapshot_integrity_sha256": str(base_snapshot_integrity),
        "evaluated_checkpoint_integrity_sha256": str(
            manifest["model_integrity"]["manifest_sha256"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired document bootstrap for exact lm-eval sample logs"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-run-dir", required=True)
    parser.add_argument("--comparison-run-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    validate_formal_evaluation_config(config)
    protocol_config = copy.deepcopy(config)
    protocol_config["run"].pop("seed", None)
    protocol_config["run"].pop("name", None)
    expected_protocol_hash = canonical_config_hash(protocol_config)
    reference_dir = Path(args.reference_run_dir).resolve()
    comparison_dir = Path(args.comparison_run_dir).resolve()
    output = Path(args.output).resolve()
    for run_dir in (reference_dir, comparison_dir):
        try:
            output.relative_to(run_dir)
        except ValueError:
            continue
        raise ValueError(
            "Bootstrap output must be outside integrity-protected evaluation "
            "directories"
        )
    reference_manifest = _validated_manifest(reference_dir)
    comparison_manifest = _validated_manifest(comparison_dir)
    configured_tasks = [
        *config["tasks"]["downstream"],
        *config["tasks"]["retention"],
    ]
    for manifest in (reference_manifest, comparison_manifest):
        if manifest.get("protocol_config_hash") != expected_protocol_hash:
            raise ValueError(
                "Paired run does not match the supplied locked protocol config"
            )
        if manifest.get("tasks") != configured_tasks:
            raise ValueError(
                "Paired bootstrap requires every locked task in protocol order"
            )
    for key in (
        "seed",
        "tasks",
        "evaluator_revision",
        "evaluator_seeds",
        "protocol_config_hash",
        "dataset_evidence",
        "source",
    ):
        if reference_manifest.get(key) != comparison_manifest.get(key):
            raise ValueError(f"Paired runs disagree on {key}")
    if (
        reference_manifest.get("task_definitions", {}).get("tree_sha256")
        != comparison_manifest.get("task_definitions", {}).get("tree_sha256")
    ):
        raise ValueError("Paired runs use different evaluator task definitions")
    if (
        reference_manifest.get("request_cache", {}).get("key")
        != comparison_manifest.get("request_cache", {}).get("key")
        or reference_manifest.get("request_cache", {})
        .get("integrity", {})
        .get("tree_sha256")
        != comparison_manifest.get("request_cache", {})
        .get("integrity", {})
        .get("tree_sha256")
    ):
        raise ValueError("Paired runs use different request-cache identities")
    reference_identity = _method_identity(reference_manifest)
    comparison_identity = _method_identity(comparison_manifest)
    for key in (
        "protocol_track",
        "model_id",
        "base_snapshot_integrity_sha256",
    ):
        if reference_identity[key] != comparison_identity[key]:
            raise ValueError(f"Paired model manifests disagree on {key}")
    if (
        reference_identity["training_commit"]
        != comparison_identity["training_commit"]
    ):
        raise ValueError("Paired methods must share one training commit")
    if (
        reference_identity["training_comparison_config_hash"]
        != comparison_identity["training_comparison_config_hash"]
    ):
        raise ValueError(
            "Paired methods must share one normalized training protocol"
        )
    if (
        not isinstance(
            reference_identity["training_environment_identity"],
            dict,
        )
        or reference_identity["training_environment_identity"]
        != comparison_identity["training_environment_identity"]
    ):
        raise ValueError(
            "Paired methods must share one training environment"
        )

    aggregation_source = git_source_state(
        Path(__file__).resolve().parents[2]
    )
    validate_derivation_source(
        aggregation_source,
        expected_commit=str(
            reference_manifest["source"]["commit_sha"]
        ),
        formal=True,
    )
    aggregation_environment = software_environment_identity(
        runtime_environment(
            torch.device("cpu"),
            validate_formal=True,
        )
    )
    expected_environment = reference_manifest["request_cache"][
        "identity"
    ]["software_environment"]
    if (
        expected_environment
        != comparison_manifest["request_cache"]["identity"][
            "software_environment"
        ]
        or aggregation_environment != expected_environment
    ):
        raise ValueError(
            "Paired bootstrap software environment differs from evaluation"
        )

    tasks = configured_tasks
    task_bootstrap: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    task_reports = {}
    audit_hash = hashlib.sha256()
    for task_index, task in enumerate(tasks):
        metric, filter_name = _metric_and_filter(config["metrics"][task])
        reference_values, comparison_values, audit = _paired_values(
            _sample_file(reference_dir, task),
            _sample_file(comparison_dir, task),
            metric=metric,
            filter_name=filter_name,
        )
        audit_hash.update(
            json.dumps(audit, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        reference_bootstrap, comparison_bootstrap = _bootstrap_means(
            reference_values,
            comparison_values,
            samples=args.bootstrap_samples,
            seed=20260725 + task_index,
        )
        task_bootstrap[task] = (reference_bootstrap, comparison_bootstrap)
        differences = reference_values - comparison_values
        task_reports[task] = {
            "metric": metric,
            "filter": filter_name,
            "documents": int(reference_values.size),
            "paired_sample_counts": {
                "reference_wins": int(np.sum(differences > 0.0)),
                "comparison_wins": int(np.sum(differences < 0.0)),
                "ties": int(np.sum(differences == 0.0)),
            },
            "effect": _effect(
                float(reference_values.mean()),
                float(comparison_values.mean()),
                reference_bootstrap,
                comparison_bootstrap,
            ),
        }

    aggregate_reports = {}
    category_bootstrap = {}
    for category in ("downstream", "retention"):
        category_tasks = config["tasks"][category]
        reference_matrix = np.stack(
            [task_bootstrap[task][0] for task in category_tasks]
        )
        comparison_matrix = np.stack(
            [task_bootstrap[task][1] for task in category_tasks]
        )
        reference_bootstrap = reference_matrix.mean(axis=0)
        comparison_bootstrap = comparison_matrix.mean(axis=0)
        reference_observed = np.mean(
            [task_reports[task]["effect"]["reference_score"] for task in category_tasks]
        )
        comparison_observed = np.mean(
            [
                task_reports[task]["effect"]["comparison_score"]
                for task in category_tasks
            ]
        )
        category_bootstrap[category] = (
            reference_bootstrap,
            comparison_bootstrap,
        )
        aggregate_reports[f"{category}_average"] = _effect(
            float(reference_observed),
            float(comparison_observed),
            reference_bootstrap,
            comparison_bootstrap,
        )

    reference_geometric = np.sqrt(
        np.maximum(category_bootstrap["downstream"][0], 0.0)
        * np.maximum(category_bootstrap["retention"][0], 0.0)
    )
    comparison_geometric = np.sqrt(
        np.maximum(category_bootstrap["downstream"][1], 0.0)
        * np.maximum(category_bootstrap["retention"][1], 0.0)
    )
    aggregate_reports["skill_retention_geometric_mean"] = _effect(
        float(
            np.sqrt(
                max(
                    aggregate_reports["downstream_average"]["reference_score"],
                    0.0,
                )
                * max(
                    aggregate_reports["retention_average"]["reference_score"],
                    0.0,
                )
            )
        ),
        float(
            np.sqrt(
                max(
                    aggregate_reports["downstream_average"]["comparison_score"],
                    0.0,
                )
                * max(
                    aggregate_reports["retention_average"]["comparison_score"],
                    0.0,
                )
            )
        ),
        reference_geometric,
        comparison_geometric,
    )

    payload = {
        "format_version": 2,
        "formal_result_eligible": True,
        "bootstrap_scope": "paired document/sample level",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": int(reference_manifest["seed"]),
        "evaluator_revision": reference_manifest["evaluator_revision"],
        "model_id": reference_identity["model_id"],
        "protocol_track": reference_identity["protocol_track"],
        "reference": reference_identity,
        "comparison": comparison_identity,
        "reference_evaluation_output_integrity": reference_manifest[
            "_validated_output_integrity"
        ],
        "comparison_evaluation_output_integrity": comparison_manifest[
            "_validated_output_integrity"
        ],
        "paired_sample_identity_sha256": audit_hash.hexdigest(),
        "aggregation_source": aggregation_source,
        "aggregation_software_environment": aggregation_environment,
        "tasks": task_reports,
        "aggregates": aggregate_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
