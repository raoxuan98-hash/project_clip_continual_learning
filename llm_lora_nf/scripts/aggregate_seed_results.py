#!/usr/bin/env python3
"""Aggregate formal lm-eval summaries across seeds and methods."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch

from llm_lora_nf.environment import (
    runtime_environment,
    software_environment_identity,
)
from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.provenance import (
    git_source_state,
    validate_derivation_source,
)
from llm_lora_nf.stats import paired_differences, summarize_values


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_close(actual: Any, expected: Any, *, label: str) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Summary-derived value mismatch for {label}")


def _validate_result_scores(
    result_path: Path,
    *,
    run_dir: Path,
    expected_sha256: str,
    scores: Mapping[str, Any],
) -> None:
    if not _is_within(result_path, run_dir):
        raise ValueError("lm-eval result path escapes its protected run directory")
    if _sha256(result_path) != expected_sha256:
        raise ValueError(f"lm-eval result checksum mismatch: {result_path}")
    result = _load_json(result_path)
    raw_results = result.get("results")
    if not isinstance(raw_results, dict):
        raise ValueError(f"Invalid lm-eval result payload: {result_path}")
    for task, score_record in scores.items():
        metric = str(score_record["metric"])
        if task not in raw_results or metric not in raw_results[task]:
            raise ValueError(
                f"Summary metric {task}/{metric} is absent from raw lm-eval output"
            )
        _assert_close(
            score_record["score"],
            raw_results[task][metric],
            label=f"{task}/{metric}",
        )


def _validate_summary_derivations(summary: Mapping[str, Any]) -> None:
    scores = summary["scores"]
    groups = summary.get("task_groups")
    if not isinstance(groups, dict):
        raise ValueError("Summary lacks task-group membership")
    grouped_tasks = [
        *groups.get("downstream", []),
        *groups.get("retention", []),
    ]
    if (
        len(grouped_tasks) != len(set(grouped_tasks))
        or set(grouped_tasks) != set(scores)
    ):
        raise ValueError("Summary task groups do not exactly cover its scores")
    averages = {}
    for category in ("downstream", "retention"):
        tasks = list(groups[category])
        if not tasks:
            averages[category] = None
            continue
        average = sum(float(scores[task]["score"]) for task in tasks) / len(tasks)
        _assert_close(
            summary[f"{category}_average"],
            average,
            label=f"{category}_average",
        )
        averages[category] = average
    if averages["downstream"] is not None and averages["retention"] is not None:
        expected_geometric = math.sqrt(
            max(0.0, averages["downstream"])
            * max(0.0, averages["retention"])
        )
        _assert_close(
            summary["skill_retention_geometric_mean"],
            expected_geometric,
            label="skill_retention_geometric_mean",
        )


def _validate_summary_evidence(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if summary.get("format_version") != 2:
        raise ValueError(f"Unsupported evaluation summary version: {summary_path}")
    run_manifest_path = Path(summary["run_manifest"]).resolve()
    if _sha256(run_manifest_path) != summary.get("run_manifest_sha256"):
        raise ValueError(f"Run-manifest checksum mismatch: {summary_path}")
    run_dir = run_manifest_path.parent
    output_integrity = validate_directory_integrity(
        str(run_dir),
        expected_kind="evaluation_output",
    )
    recorded_integrity = summary.get("evaluation_output_integrity", {})
    if (
        output_integrity["manifest_sha256"]
        != recorded_integrity.get("manifest_sha256")
        or output_integrity["tree_sha256"]
        != recorded_integrity.get("tree_sha256")
    ):
        raise ValueError(f"Evaluation-output identity mismatch: {summary_path}")
    run_manifest = _load_json(run_manifest_path)
    if (
        run_manifest.get("format_version") != 2
        or run_manifest.get("status") != "passed"
        or run_manifest.get("returncode") != 0
    ):
        raise ValueError(f"Evaluation run did not pass: {summary_path}")
    if summary.get("dataset_evidence") != run_manifest.get(
        "dataset_evidence"
    ):
        raise ValueError(f"Evaluation-data evidence mismatch: {summary_path}")
    if summary.get("task_definition_tree_sha256") != run_manifest.get(
        "task_definitions",
        {},
    ).get("tree_sha256"):
        raise ValueError(f"Evaluator-task evidence mismatch: {summary_path}")
    if summary.get("evaluator_seeds") != run_manifest.get("evaluator_seeds"):
        raise ValueError(f"Evaluator-seed evidence mismatch: {summary_path}")
    if summary.get("request_cache_key") != run_manifest.get(
        "request_cache",
        {},
    ).get("key"):
        raise ValueError(f"Request-cache identity mismatch: {summary_path}")
    if summary.get("request_cache_tree_sha256") != run_manifest.get(
        "request_cache",
        {},
    ).get("integrity", {}).get("tree_sha256"):
        raise ValueError(f"Request-cache content mismatch: {summary_path}")
    if (
        not isinstance(summary.get("aggregation_source"), dict)
        or summary.get("aggregation_source") != run_manifest.get("source")
    ):
        raise ValueError(f"Aggregation-source mismatch: {summary_path}")
    if (
        not isinstance(
            summary.get("aggregation_software_environment"),
            dict,
        )
        or summary.get("aggregation_software_environment")
        != run_manifest.get("request_cache", {})
        .get("identity", {})
        .get("software_environment")
    ):
        raise ValueError(
            f"Aggregation-environment mismatch: {summary_path}"
        )
    _validate_result_scores(
        Path(summary["lm_eval_result"]).resolve(),
        run_dir=run_dir,
        expected_sha256=str(summary["lm_eval_result_sha256"]),
        scores=summary["scores"],
    )
    _validate_summary_derivations(summary)
    model_identity = summary.get("model_identity")
    if not isinstance(model_identity, dict):
        raise ValueError(f"Summary lacks model identity: {summary_path}")
    if (
        model_identity.get("evaluated_checkpoint_integrity_sha256")
        != run_manifest.get("model_integrity", {}).get("manifest_sha256")
    ):
        raise ValueError(f"Evaluated checkpoint identity mismatch: {summary_path}")
    base = summary.get("base_comparison")
    if base is not None:
        base_manifest_path = Path(base["base_run_manifest"]).resolve()
        if _sha256(base_manifest_path) != base.get(
            "base_run_manifest_sha256"
        ):
            raise ValueError(f"Base run-manifest checksum mismatch: {summary_path}")
        base_dir = base_manifest_path.parent
        base_output_integrity = validate_directory_integrity(
            str(base_dir),
            expected_kind="evaluation_output",
        )
        recorded_base_integrity = base.get(
            "base_evaluation_output_integrity",
            {},
        )
        if (
            base_output_integrity["manifest_sha256"]
            != recorded_base_integrity.get("manifest_sha256")
            or base_output_integrity["tree_sha256"]
            != recorded_base_integrity.get("tree_sha256")
        ):
            raise ValueError(f"Base evaluation identity mismatch: {summary_path}")
        base_manifest = _load_json(base_manifest_path)
        if (
            base_manifest.get("format_version") != 2
            or base_manifest.get("status") != "passed"
            or base_manifest.get("returncode") != 0
            or base_manifest.get("model_role") != "base"
        ):
            raise ValueError(f"Unqualified base evaluation: {summary_path}")
        for key in (
            "evaluator_revision",
            "evaluator_seeds",
            "tasks",
            "dataset_evidence",
            "protocol_config_hash",
            "source",
        ):
            if base_manifest.get(key) != run_manifest.get(key):
                raise ValueError(
                    "Base/adapted evaluation evidence disagrees on "
                    f"{key}: {summary_path}"
                )
        if (
            base_manifest.get("task_definitions", {}).get("tree_sha256")
            != run_manifest.get("task_definitions", {}).get("tree_sha256")
        ):
            raise ValueError(
                "Base/adapted evaluator task definitions disagree: "
                f"{summary_path}"
            )
        if (
            base_manifest.get("request_cache", {}).get("key")
            != run_manifest.get("request_cache", {}).get("key")
            or base_manifest.get("request_cache", {})
            .get("integrity", {})
            .get("tree_sha256")
            != run_manifest.get("request_cache", {})
            .get("integrity", {})
            .get("tree_sha256")
        ):
            raise ValueError(
                "Base/adapted request-cache evidence disagrees: "
                f"{summary_path}"
            )
        if summary.get("formal_result_eligible"):
            if (
                not base_manifest.get("formal_result_eligible")
                or base_manifest.get("execution_mode") != "gpu_formal"
                or base_manifest.get("limit") is not None
                or not base_manifest.get("source", {}).get("commit_sha")
                or base_manifest.get("source", {}).get("source_dirty")
            ):
                raise ValueError(
                    "Formal summary references a non-formal or unqualified "
                    f"base run: {summary_path}"
                )
        _validate_result_scores(
            Path(base["base_lm_eval_result"]).resolve(),
            run_dir=base_dir,
            expected_sha256=str(base["base_lm_eval_result_sha256"]),
            scores=base["base_scores"],
        )
        if set(base["base_scores"]) != set(summary["scores"]):
            raise ValueError(
                f"Base/adapted task sets do not match: {summary_path}"
            )
        if (
            base.get("base_model_identity", {}).get(
                "base_snapshot_integrity_sha256"
            )
            != model_identity.get("base_snapshot_integrity_sha256")
        ):
            raise ValueError(f"Base snapshot identity mismatch: {summary_path}")
        groups = summary["task_groups"]
        downstream_tasks = list(groups["downstream"])
        if downstream_tasks:
            base_downstream = sum(
                float(base["base_scores"][task]["score"])
                for task in downstream_tasks
            ) / len(downstream_tasks)
            _assert_close(
                base["downstream_gain"],
                float(summary["downstream_average"]) - base_downstream,
                label="base/downstream_gain",
            )
        retention_ratios = {}
        for task in groups["retention"]:
            denominator = float(base["base_scores"][task]["score"])
            retention_ratios[task] = (
                float(summary["scores"][task]["score"]) / denominator
                if denominator > 0.0
                else None
            )
        recorded_ratios = base["retention_ratios"]
        if set(recorded_ratios) != set(retention_ratios):
            raise ValueError(
                f"Base retention-ratio task mismatch: {summary_path}"
            )
        for task, expected_ratio in retention_ratios.items():
            recorded_ratio = recorded_ratios[task]
            if expected_ratio is None:
                if recorded_ratio is not None:
                    raise ValueError(
                        "Base retention-ratio derivation mismatch for "
                        f"{task}: {summary_path}"
                    )
            else:
                _assert_close(
                    recorded_ratio,
                    expected_ratio,
                    label=f"base/retention_ratio/{task}",
                )
        defined = [
            value for value in retention_ratios.values() if value is not None
        ]
        expected_ratio_average = (
            sum(defined) / len(defined) if defined else None
        )
        if expected_ratio_average is None:
            if base["retention_ratio_average"] is not None:
                raise ValueError(
                    f"Unexpected base retention average: {summary_path}"
                )
        else:
            _assert_close(
                base["retention_ratio_average"],
                expected_ratio_average,
                label="base/retention_ratio_average",
            )
    return run_manifest, model_identity


def _resolve_record(summary_path: Path, *, allow_cpu_smoke: bool) -> Dict[str, Any]:
    summary = _load_json(summary_path)
    if not summary.get("formal_result_eligible") and not allow_cpu_smoke:
        raise ValueError(f"Non-formal summary cannot be aggregated: {summary_path}")
    run_manifest, model_identity = _validate_summary_evidence(
        summary,
        summary_path=summary_path,
    )
    run_manifest_path = Path(summary["run_manifest"]).resolve()
    if not run_manifest.get("formal_result_eligible") and not allow_cpu_smoke:
        raise ValueError(f"Non-formal evaluation run: {run_manifest_path}")
    if run_manifest.get("status") != "passed":
        raise ValueError(f"Evaluation run did not pass: {run_manifest_path}")
    if run_manifest.get("formal_result_eligible") and (
        run_manifest.get("execution_mode") != "gpu_formal"
        or run_manifest.get("limit") is not None
        or not run_manifest.get("source", {}).get("commit_sha")
        or run_manifest.get("source", {}).get("source_dirty")
    ):
        raise ValueError(f"Unqualified formal evaluation: {run_manifest_path}")
    model_manifest = run_manifest.get("model_manifest")
    if not isinstance(model_manifest, dict):
        if run_manifest.get("formal_result_eligible"):
            raise ValueError(
                "Formal run lacks its embedded model manifest: "
                f"{run_manifest_path}"
            )
        model_manifest = _load_json(
            Path(run_manifest["model_manifest_path"]).resolve()
        )
    if run_manifest.get("model_role") == "merged":
        if run_manifest.get("formal_result_eligible") and (
            not model_manifest.get("formal_result_eligible")
            or model_manifest.get("execution_mode") != "gpu_formal"
            or model_manifest.get("export_source_dirty")
            or model_manifest.get("export_commit_sha")
            != run_manifest.get("source", {}).get("commit_sha")
            or model_manifest.get("training_commit_sha")
            != model_manifest.get("export_commit_sha")
            or model_manifest.get("dtype") != "torch.float32"
            or model_manifest.get("checkpoint_retention")
            != "ephemeral_delete_after_qualified_evaluation"
        ):
            raise ValueError(
                f"Unqualified merged export evidence: {summary_path}"
            )
        method = str(model_manifest["method"])
        protocol_track = str(model_manifest.get("protocol_track", "B"))
        model_id = str(model_manifest["base_model"]["requested_id"])
        training_commit = str(model_manifest["training_commit_sha"])
        training_config_hash = str(
            model_manifest["training_protocol_config_hash"]
        )
        training_comparison_config_hash = str(
            model_manifest["training_comparison_config_hash"]
        )
        training_environment_identity = model_manifest.get(
            "training_environment_identity"
        )
        if not isinstance(training_environment_identity, dict):
            raise ValueError(
                f"Merged export lacks training environment: {summary_path}"
            )
        base_snapshot_integrity_sha256 = str(
            model_manifest["base_model"]["snapshot_integrity_sha256"]
        )
    elif run_manifest.get("model_role") == "base":
        method = "base"
        protocol_track = "base"
        model_id = str(model_manifest["model_id"])
        training_commit = None
        training_config_hash = None
        training_comparison_config_hash = None
        training_environment_identity = None
        base_snapshot_integrity_sha256 = str(
            run_manifest["model_integrity"]["manifest_sha256"]
        )
    else:
        raise ValueError(
            f"Unsupported model role in evaluation: {run_manifest_path}"
        )
    if (
        base_snapshot_integrity_sha256
        != model_identity.get("base_snapshot_integrity_sha256")
    ):
        raise ValueError(
            f"Summary base-snapshot identity mismatch: {summary_path}"
        )
    task_scores = {
        task: float(payload["score"])
        for task, payload in summary["scores"].items()
    }
    metrics = {
        **{f"task/{task}": score for task, score in task_scores.items()},
        "aggregate/downstream_average": summary["downstream_average"],
        "aggregate/retention_average": summary["retention_average"],
        "aggregate/skill_retention_geometric_mean": summary[
            "skill_retention_geometric_mean"
        ],
    }
    if summary.get("base_comparison"):
        metrics["base/downstream_gain"] = summary["base_comparison"][
            "downstream_gain"
        ]
        metrics["base/retention_ratio_average"] = summary["base_comparison"][
            "retention_ratio_average"
        ]
    if any(value is None for value in metrics.values()):
        raise ValueError(f"Summary contains undefined aggregate metrics: {summary_path}")
    return {
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "run_manifest_path": str(run_manifest_path),
        "protected_run_dirs": [
            str(run_manifest_path.parent),
            *(
                [
                    str(
                        Path(
                            summary["base_comparison"]["base_run_manifest"]
                        ).resolve().parent
                    )
                ]
                if summary.get("base_comparison")
                else []
            ),
        ],
        "method": method,
        "protocol_track": protocol_track,
        "model_id": model_id,
        "base_snapshot_integrity_sha256": base_snapshot_integrity_sha256,
        "seed": int(run_manifest["seed"]),
        "training_commit": training_commit,
        "training_config_hash": training_config_hash,
        "training_comparison_config_hash": training_comparison_config_hash,
        "training_environment_identity": training_environment_identity,
        "evaluator_revision": str(run_manifest["evaluator_revision"]),
        "evaluator_seeds": dict(run_manifest["evaluator_seeds"]),
        "dataset_manifest_sha256": str(
            run_manifest["dataset_evidence"]["manifest_sha256"]
        ),
        "task_definition_tree_sha256": str(
            run_manifest["task_definitions"]["tree_sha256"]
        ),
        "request_cache_key": str(run_manifest["request_cache"]["key"]),
        "request_cache_tree_sha256": str(
            run_manifest["request_cache"]["integrity"]["tree_sha256"]
        ),
        "evaluation_source": dict(run_manifest["source"]),
        "evaluation_software_environment": dict(
            run_manifest["request_cache"]["identity"][
                "software_environment"
            ]
        ),
        "config_hash": str(
            run_manifest.get(
                "protocol_config_hash",
                run_manifest["config_hash"],
            )
        ),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "formal_result_eligible": bool(
            summary["formal_result_eligible"]
            and run_manifest["formal_result_eligible"]
        ),
    }


def _aggregate_group(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_seeds: List[int],
    bootstrap_samples: int,
) -> Dict[str, Any]:
    rows = list(records)
    seed_map = {int(row["seed"]): row for row in rows}
    if len(seed_map) != len(rows):
        raise ValueError("Duplicate seed found in an aggregation group")
    if set(seed_map) != set(expected_seeds):
        raise ValueError(
            f"Expected seeds {expected_seeds}, found {sorted(seed_map)}"
        )
    evaluator_revisions = {row["evaluator_revision"] for row in rows}
    evaluator_seed_sets = {
        json.dumps(row["evaluator_seeds"], sort_keys=True)
        for row in rows
    }
    dataset_manifest_hashes = {
        row["dataset_manifest_sha256"] for row in rows
    }
    task_definition_hashes = {
        row["task_definition_tree_sha256"] for row in rows
    }
    request_cache_keys = {row["request_cache_key"] for row in rows}
    request_cache_trees = {
        row["request_cache_tree_sha256"] for row in rows
    }
    config_hashes = {row["config_hash"] for row in rows}
    metric_sets = {tuple(sorted(row["metrics"])) for row in rows}
    base_snapshot_integrities = {
        row["base_snapshot_integrity_sha256"] for row in rows
    }
    training_commits = {
        row["training_commit"]
        for row in rows
        if row["training_commit"] is not None
    }
    training_config_hashes = {
        row["training_config_hash"]
        for row in rows
        if row["training_config_hash"] is not None
    }
    training_comparison_config_hashes = {
        row["training_comparison_config_hash"]
        for row in rows
        if row["training_comparison_config_hash"] is not None
    }
    training_environment_identities = {
        json.dumps(row["training_environment_identity"], sort_keys=True)
        for row in rows
        if row["training_environment_identity"] is not None
    }
    if len(evaluator_revisions) != 1:
        raise ValueError("Mixed evaluator revisions in one aggregation group")
    if len(evaluator_seed_sets) != 1:
        raise ValueError("Mixed evaluator seeds in one aggregation group")
    if len(dataset_manifest_hashes) != 1:
        raise ValueError("Mixed evaluation datasets in one aggregation group")
    if len(task_definition_hashes) != 1:
        raise ValueError(
            "Mixed evaluator task definitions in one aggregation group"
        )
    if len(request_cache_keys) != 1:
        raise ValueError("Mixed request-cache identities in one aggregation group")
    if len(request_cache_trees) != 1:
        raise ValueError("Mixed request-cache contents in one aggregation group")
    if len(config_hashes) != 1:
        raise ValueError("Mixed evaluation config hashes in one aggregation group")
    if len(metric_sets) != 1:
        raise ValueError("Mixed metric sets in one aggregation group")
    if len(base_snapshot_integrities) != 1:
        raise ValueError("Mixed base-model snapshots in one aggregation group")
    if rows[0]["method"] != "base":
        if len(training_commits) != 1:
            raise ValueError("Mixed training commits in one aggregation group")
        if len(training_config_hashes) != 1:
            raise ValueError(
                "Mixed training protocol configs in one aggregation group"
            )
        if len(training_comparison_config_hashes) != 1:
            raise ValueError(
                "Mixed comparison configs in one aggregation group"
            )
        if len(training_environment_identities) != 1:
            raise ValueError(
                "Mixed training environments in one aggregation group"
            )
    metric_names = sorted(rows[0]["metrics"])
    return {
        "model_id": rows[0]["model_id"],
        "protocol_track": rows[0]["protocol_track"],
        "method": rows[0]["method"],
        "base_snapshot_integrity_sha256": next(
            iter(base_snapshot_integrities)
        ),
        "seeds": sorted(seed_map),
        "training_commits": sorted(training_commits),
        "training_protocol_config_hashes": sorted(training_config_hashes),
        "training_comparison_config_hashes": sorted(
            training_comparison_config_hashes
        ),
        "training_environment_identities": [
            json.loads(value)
            for value in sorted(training_environment_identities)
        ],
        "evaluator_revision": next(iter(evaluator_revisions)),
        "evaluator_seeds": json.loads(next(iter(evaluator_seed_sets))),
        "dataset_manifest_sha256": next(iter(dataset_manifest_hashes)),
        "task_definition_tree_sha256": next(iter(task_definition_hashes)),
        "request_cache_key": next(iter(request_cache_keys)),
        "request_cache_tree_sha256": next(iter(request_cache_trees)),
        "evaluation_config_hash": next(iter(config_hashes)),
        "formal_result_eligible": all(
            row["formal_result_eligible"] for row in rows
        ),
        "metrics": {
            metric: summarize_values(
                [seed_map[seed]["metrics"][metric] for seed in sorted(seed_map)],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=20260725,
            )
            for metric in metric_names
        },
        "source_summaries": [
            seed_map[seed]["summary_path"] for seed in sorted(seed_map)
        ],
        "source_summary_sha256": {
            seed_map[seed]["summary_path"]: seed_map[seed]["summary_sha256"]
            for seed in sorted(seed_map)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate exact-seed formal results and paired method deltas"
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Path to aggregate_lm_eval.py output; repeat for every run.",
    )
    parser.add_argument("--expected-seeds", default="42,43,44")
    parser.add_argument("--reference-method", default="lora_nf")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected_seeds = [
        int(value.strip())
        for value in args.expected_seeds.split(",")
        if value.strip()
    ]
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected-seeds must be a non-empty unique list")
    records = [
        _resolve_record(
            Path(path).resolve(),
            allow_cpu_smoke=args.allow_cpu_smoke,
        )
        for path in args.summary
    ]
    formal_inputs = all(
        record["formal_result_eligible"] for record in records
    )
    evaluation_commits = {
        record["evaluation_source"]["commit_sha"] for record in records
    }
    if formal_inputs and len(evaluation_commits) != 1:
        raise ValueError(
            "Formal seed aggregation cannot mix evaluation source commits"
        )
    aggregation_source = git_source_state(
        Path(__file__).resolve().parents[2]
    )
    validate_derivation_source(
        aggregation_source,
        expected_commit=(
            next(iter(evaluation_commits))
            if len(evaluation_commits) == 1
            else aggregation_source["commit_sha"]
        ),
        formal=formal_inputs,
    )
    aggregation_environment = software_environment_identity(
        runtime_environment(
            torch.device("cpu"),
            validate_formal=formal_inputs,
        )
    )
    evaluation_environments = {
        json.dumps(
            record["evaluation_software_environment"],
            sort_keys=True,
        )
        for record in records
    }
    if (
        formal_inputs
        and (
            len(evaluation_environments) != 1
            or aggregation_environment
            != json.loads(next(iter(evaluation_environments)))
        )
    ):
        raise ValueError(
            "Formal seed aggregation software environment differs from "
            "evaluation"
        )
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["model_id"],
            record["protocol_track"],
            record["method"],
        )
        grouped[key].append(record)
    aggregates = {
        key: _aggregate_group(
            rows,
            expected_seeds=expected_seeds,
            bootstrap_samples=args.bootstrap_samples,
        )
        for key, rows in grouped.items()
    }

    pairwise = []
    model_tracks = sorted({(key[0], key[1]) for key in aggregates})
    for model_id, protocol_track in model_tracks:
        reference_key = (model_id, protocol_track, args.reference_method)
        if reference_key not in aggregates:
            continue
        reference_rows = {
            int(row["seed"]): row
            for row in grouped[reference_key]
        }
        for key in sorted(aggregates):
            if key[:2] != (model_id, protocol_track) or key == reference_key:
                continue
            if (
                aggregates[key]["training_commits"]
                != aggregates[reference_key]["training_commits"]
            ):
                raise ValueError(
                    "Paired method comparison requires the same training commit"
                )
            if (
                aggregates[key]["training_comparison_config_hashes"]
                != aggregates[reference_key][
                    "training_comparison_config_hashes"
                ]
            ):
                raise ValueError(
                    "Paired method comparison requires the same normalized "
                    "training protocol"
                )
            if (
                aggregates[key]["training_environment_identities"]
                != aggregates[reference_key][
                    "training_environment_identities"
                ]
            ):
                raise ValueError(
                    "Paired method comparison requires the same training "
                    "environment"
                )
            if (
                aggregates[key]["base_snapshot_integrity_sha256"]
                != aggregates[reference_key][
                    "base_snapshot_integrity_sha256"
                ]
            ):
                raise ValueError(
                    "Paired method comparison requires the exact same base "
                    "model snapshot"
                )
            if (
                aggregates[key]["evaluator_revision"]
                != aggregates[reference_key]["evaluator_revision"]
                or aggregates[key]["evaluator_seeds"]
                != aggregates[reference_key]["evaluator_seeds"]
                or aggregates[key]["evaluation_config_hash"]
                != aggregates[reference_key]["evaluation_config_hash"]
                or aggregates[key]["dataset_manifest_sha256"]
                != aggregates[reference_key]["dataset_manifest_sha256"]
                or aggregates[key]["task_definition_tree_sha256"]
                != aggregates[reference_key]["task_definition_tree_sha256"]
                or aggregates[key]["request_cache_key"]
                != aggregates[reference_key]["request_cache_key"]
                or aggregates[key]["request_cache_tree_sha256"]
                != aggregates[reference_key]["request_cache_tree_sha256"]
            ):
                raise ValueError(
                    "Paired method comparison requires one evaluation protocol"
                )
            comparison_rows = {
                int(row["seed"]): row
                for row in grouped[key]
            }
            reference_metrics = set(
                reference_rows[expected_seeds[0]]["metrics"]
            )
            comparison_metrics = set(
                comparison_rows[expected_seeds[0]]["metrics"]
            )
            if reference_metrics != comparison_metrics:
                raise ValueError(
                    "Paired method comparison requires identical metric sets"
                )
            common_metrics = sorted(reference_metrics)
            pairwise.append(
                {
                    "model_id": model_id,
                    "protocol_track": protocol_track,
                    "reference_method": args.reference_method,
                    "comparison_method": key[2],
                    "difference_definition": "reference_minus_comparison",
                    "metrics": {
                        metric: summarize_values(
                            paired_differences(
                                {
                                    seed: reference_rows[seed]["metrics"][metric]
                                    for seed in expected_seeds
                                },
                                {
                                    seed: comparison_rows[seed]["metrics"][metric]
                                    for seed in expected_seeds
                                },
                            ),
                            bootstrap_samples=args.bootstrap_samples,
                            bootstrap_seed=20260725,
                        )
                        for metric in common_metrics
                    },
                }
            )

    output = Path(args.output).resolve()
    source_summary_paths = {
        Path(record["summary_path"]).resolve() for record in records
    }
    if output in source_summary_paths:
        raise ValueError(
            "Seed aggregation output cannot overwrite a source summary"
        )
    protected_run_dirs = {
        Path(path).resolve()
        for record in records
        for path in record["protected_run_dirs"]
    }
    for run_dir in protected_run_dirs:
        if _is_within(output, run_dir):
            raise ValueError(
                "Seed aggregation output must be outside every "
                "integrity-protected evaluation directory"
            )

    payload = {
        "format_version": 2,
        "formal_result_eligible": all(
            aggregate["formal_result_eligible"]
            for aggregate in aggregates.values()
        ),
        "expected_seeds": expected_seeds,
        "bootstrap_scope": (
            "seed-level mean uncertainty; sample-level paired bootstrap "
            "must be reported separately when task logs support it"
        ),
        "aggregation_source": aggregation_source,
        "aggregation_software_environment": aggregation_environment,
        "groups": [
            aggregates[key]
            for key in sorted(aggregates)
        ],
        "paired_comparisons": pairwise,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
