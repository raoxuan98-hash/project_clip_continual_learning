#!/usr/bin/env python3
"""Aggregate sealed EvalPlus runs across exact training seeds."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch

from llm_lora_nf.environment import (
    code_evaluation_package_versions,
    runtime_environment,
    software_environment_identity,
    validate_code_evaluation_package_versions,
)
from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.provenance import (
    git_source_state,
    validate_derivation_source,
)
from llm_lora_nf.stats import paired_differences, summarize_values


METRICS = (
    "humaneval_pass_at_1",
    "humaneval_plus_pass_at_1",
    "mbpp_pass_at_1",
    "mbpp_plus_pass_at_1",
)


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


def _stable_identity(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_execution_run(
    run_dir: Path,
    *,
    allow_cpu_smoke: bool,
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    integrity = validate_directory_integrity(
        str(run_dir),
        expected_kind="code_execution_output",
    )
    report_path = run_dir / "code_execution_run.json"
    report = _load_json(report_path)
    formal = bool(report.get("formal_result_eligible"))
    if (
        report.get("format_version") != 1
        or report.get("status") != "passed"
        or (formal and report.get("execution_mode") != "cpu_sandbox_formal")
        or (not formal and not allow_cpu_smoke)
    ):
        raise ValueError(f"Unqualified EvalPlus execution: {run_dir}")
    if (
        report.get("sandbox", {}).get("network") != "disabled"
        or not report.get("sandbox", {}).get("unshare_all")
    ):
        raise ValueError(f"EvalPlus sandbox evidence is incomplete: {run_dir}")

    generation_dir = Path(str(report.get("generation_dir", ""))).resolve()
    generation_integrity = validate_directory_integrity(
        str(generation_dir),
        expected_kind="code_generation_output",
    )
    if (
        report.get("generation_integrity", {}).get("manifest_sha256")
        != generation_integrity["manifest_sha256"]
        or report.get("generation_integrity", {}).get("tree_sha256")
        != generation_integrity["tree_sha256"]
    ):
        raise ValueError(f"Generation integrity mismatch: {run_dir}")
    generation = _load_json(generation_dir / "code_generation_run.json")
    if report.get("generation_report") != generation:
        raise ValueError(f"Embedded generation report mismatch: {run_dir}")
    if (
        generation.get("format_version") != 1
        or generation.get("status") != "generated"
        or bool(generation.get("formal_result_eligible")) != formal
        or report.get("seed") != generation.get("seed")
        or report.get("source") != generation.get("source")
        or report.get("evaluator") != generation.get("evaluator")
        or report.get("protocol_config_hash")
        != generation.get("protocol_config_hash")
        or report.get("dataset_evidence")
        != generation.get("dataset_evidence")
        or report.get("code_evaluation_packages")
        != generation.get("code_evaluation_packages")
    ):
        raise ValueError(f"Generation/execution identity mismatch: {run_dir}")
    for dataset in ("humaneval", "mbpp"):
        if _sha256(run_dir / f"{dataset}_samples.jsonl") != generation[
            "samples"
        ][dataset]["sha256"]:
            raise ValueError(f"Copied {dataset} samples changed: {run_dir}")

    metrics = report.get("metrics", {})
    if formal and set(metrics) != set(METRICS):
        raise ValueError(f"Incomplete formal EvalPlus metrics: {run_dir}")
    generation_software = software_environment_identity(
        generation["environment"]
    )
    execution_software = software_environment_identity(report["environment"])
    if formal and generation_software != execution_software:
        raise ValueError(
            f"Generation/execution software environments differ: {run_dir}"
        )
    model_manifest = generation["model_manifest"]
    role = str(generation["model_role"])
    base_integrity = (
        generation["model_integrity"]["manifest_sha256"]
        if role == "base"
        else model_manifest["base_model_integrity_manifest_sha256"]
    )
    record = {
        "run_dir": str(run_dir),
        "run_report_sha256": _sha256(report_path),
        "generation_dir": str(generation_dir),
        "generation_integrity": generation_integrity,
        "execution_integrity": integrity,
        "formal_result_eligible": formal,
        "source": report["source"],
        "role": role,
        "seed": int(report["seed"]),
        "base_snapshot_integrity_sha256": str(base_integrity),
        "evaluation_protocol_config_hash": report[
            "protocol_config_hash"
        ],
        "dataset_evidence": report["dataset_evidence"],
        "evaluator": report["evaluator"],
        "code_evaluation_packages": report["code_evaluation_packages"],
        "evaluation_software_environment": execution_software,
        "metrics": {metric: float(metrics[metric]) for metric in metrics},
    }
    if role == "base":
        record.update(
            {
                "model_id": str(
                    model_manifest.get("model_id")
                    or model_manifest.get("requested_id")
                    or model_manifest.get("resolved_path")
                ),
                "method": "base",
                "protocol_track": "B",
            }
        )
        return record
    if role != "merged":
        raise ValueError(f"Unknown code-generation model role: {role}")
    if (
        not model_manifest.get("formal_result_eligible")
        or model_manifest.get("execution_mode") != "gpu_formal"
        or model_manifest.get("dtype") != "torch.float32"
        or model_manifest.get("checkpoint_retention")
        != "ephemeral_delete_after_qualified_evaluation"
        or model_manifest.get("training_seed") != record["seed"]
        or model_manifest.get("training_commit_sha")
        != report["source"]["commit_sha"]
    ):
        raise ValueError(f"Unqualified merged-model evidence: {run_dir}")
    record.update(
        {
            "model_id": str(model_manifest["training_model_id"]),
            "method": str(model_manifest["method"]),
            "protocol_track": str(model_manifest["protocol_track"]),
            "run_id": str(model_manifest["run_id"]),
            "training_commit_sha": str(
                model_manifest["training_commit_sha"]
            ),
            "training_protocol_config_hash": str(
                model_manifest["training_protocol_config_hash"]
            ),
            "training_comparison_config_hash": str(
                model_manifest["training_comparison_config_hash"]
            ),
            "training_environment_identity": model_manifest[
                "training_environment_identity"
            ],
            "training_checkpoint_integrity_manifest_sha256": str(
                model_manifest["checkpoint_integrity_manifest_sha256"]
            ),
            "training_checkpoint_integrity_tree_sha256": str(
                model_manifest["checkpoint_integrity_tree_sha256"]
            ),
        }
    )
    return record


def _common_value(
    rows: List[Dict[str, Any]],
    field: str,
    *,
    label: str,
) -> Any:
    values = {_stable_identity(row[field]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"Mixed {label} in one EvalPlus aggregation group")
    return rows[0][field]


def _aggregate_group(
    rows: List[Dict[str, Any]],
    *,
    expected_seeds: List[int],
    bootstrap_samples: int,
    base_metrics: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    seed_map = {int(row["seed"]): row for row in rows}
    if len(seed_map) != len(rows):
        raise ValueError("Duplicate seed in one EvalPlus aggregation group")
    if sorted(seed_map) != sorted(expected_seeds):
        raise ValueError(
            "EvalPlus aggregation requires exact seeds "
            f"{sorted(expected_seeds)}, got {sorted(seed_map)}"
        )
    ordered = [seed_map[seed] for seed in sorted(seed_map)]
    for field, label in (
        ("base_snapshot_integrity_sha256", "base-model snapshots"),
        ("training_commit_sha", "training commits"),
        ("training_protocol_config_hash", "training protocol configs"),
        ("training_comparison_config_hash", "comparison configs"),
        ("training_environment_identity", "training environments"),
        ("evaluation_protocol_config_hash", "evaluation protocols"),
        ("dataset_evidence", "evaluation datasets"),
        ("evaluator", "evaluator revisions"),
        ("code_evaluation_packages", "code-evaluation packages"),
        ("evaluation_software_environment", "evaluation environments"),
        ("source", "evaluation source states"),
    ):
        _common_value(ordered, field, label=label)
    metric_sets = {tuple(sorted(row["metrics"])) for row in ordered}
    if metric_sets != {tuple(sorted(METRICS))}:
        raise ValueError("Formal EvalPlus metric sets are inconsistent")
    group = {
        "model_id": ordered[0]["model_id"],
        "protocol_track": ordered[0]["protocol_track"],
        "method": ordered[0]["method"],
        "seeds": sorted(seed_map),
        "base_snapshot_integrity_sha256": ordered[0][
            "base_snapshot_integrity_sha256"
        ],
        "training_commit_sha": ordered[0]["training_commit_sha"],
        "training_protocol_config_hash": ordered[0][
            "training_protocol_config_hash"
        ],
        "training_comparison_config_hash": ordered[0][
            "training_comparison_config_hash"
        ],
        "training_environment_identity": ordered[0][
            "training_environment_identity"
        ],
        "evaluation_protocol_config_hash": ordered[0][
            "evaluation_protocol_config_hash"
        ],
        "dataset_evidence": ordered[0]["dataset_evidence"],
        "evaluator": ordered[0]["evaluator"],
        "code_evaluation_packages": ordered[0][
            "code_evaluation_packages"
        ],
        "evaluation_software_environment": ordered[0][
            "evaluation_software_environment"
        ],
        "formal_result_eligible": all(
            row["formal_result_eligible"] for row in ordered
        ),
        "metrics": {
            metric: summarize_values(
                [seed_map[seed]["metrics"][metric] for seed in sorted(seed_map)],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=20260725,
            )
            for metric in METRICS
        },
        "source_runs": [
            {
                "seed": seed,
                "execution_dir": seed_map[seed]["run_dir"],
                "execution_report_sha256": seed_map[seed][
                    "run_report_sha256"
                ],
                "generation_dir": seed_map[seed]["generation_dir"],
            }
            for seed in sorted(seed_map)
        ],
    }
    if base_metrics is not None:
        if set(base_metrics) != set(METRICS):
            raise ValueError("Shared-base EvalPlus metric set is incomplete")
        group["shared_base_metrics"] = {
            metric: float(base_metrics[metric]) for metric in METRICS
        }
        group["adapted_minus_base"] = {
            metric: summarize_values(
                [
                    seed_map[seed]["metrics"][metric]
                    - float(base_metrics[metric])
                    for seed in sorted(seed_map)
                ],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=20260725,
            )
            for metric in METRICS
        }
    return group


def _validate_paired_compatibility(
    reference: Dict[str, Any],
    comparison: Dict[str, Any],
) -> None:
    for field, label in (
        ("base_snapshot_integrity_sha256", "base snapshot"),
        ("training_commit_sha", "training commit"),
        ("training_comparison_config_hash", "normalized training protocol"),
        ("training_environment_identity", "training environment"),
        ("evaluation_protocol_config_hash", "evaluation protocol"),
        ("dataset_evidence", "evaluation dataset"),
        ("evaluator", "evaluator revision"),
        ("code_evaluation_packages", "code-evaluation packages"),
        ("evaluation_software_environment", "evaluation environment"),
    ):
        if reference[field] != comparison[field]:
            raise ValueError(
                f"Paired EvalPlus comparison requires one {label}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate exact-seed sealed EvalPlus runs and paired method deltas"
        )
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--base-run-dir", action="append", default=[])
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
    rows = [
        _validate_execution_run(
            Path(path),
            allow_cpu_smoke=args.allow_cpu_smoke,
        )
        for path in args.run_dir
    ]
    if any(row["role"] != "merged" for row in rows):
        raise ValueError("--run-dir accepts adapted merged-model runs only")
    base_rows = [
        _validate_execution_run(
            Path(path),
            allow_cpu_smoke=args.allow_cpu_smoke,
        )
        for path in args.base_run_dir
    ]
    if any(row["role"] != "base" for row in base_rows):
        raise ValueError("--base-run-dir accepts shared base-model runs only")

    formal = all(
        row["formal_result_eligible"] for row in [*rows, *base_rows]
    )
    commits = {row["source"]["commit_sha"] for row in [*rows, *base_rows]}
    if formal and len(commits) != 1:
        raise ValueError("Formal EvalPlus aggregation cannot mix source commits")
    aggregation_source = git_source_state(
        Path(__file__).resolve().parents[2]
    )
    validate_derivation_source(
        aggregation_source,
        expected_commit=(
            next(iter(commits))
            if len(commits) == 1
            else aggregation_source["commit_sha"]
        ),
        formal=formal,
    )
    aggregation_environment = software_environment_identity(
        runtime_environment(
            torch.device("cpu"),
            validate_formal=formal,
        )
    )
    package_versions = code_evaluation_package_versions()
    validate_code_evaluation_package_versions(package_versions)
    if formal:
        evaluation_environments = {
            _stable_identity(row["evaluation_software_environment"])
            for row in [*rows, *base_rows]
        }
        if (
            evaluation_environments
            != {_stable_identity(aggregation_environment)}
            or {
                _stable_identity(row["code_evaluation_packages"])
                for row in [*rows, *base_rows]
            }
            != {_stable_identity(package_versions)}
        ):
            raise ValueError(
                "Formal aggregation environment differs from EvalPlus runs"
            )

    bases: Dict[str, Dict[str, Any]] = {}
    for row in base_rows:
        key = row["base_snapshot_integrity_sha256"]
        if key in bases:
            raise ValueError("Duplicate shared-base EvalPlus run")
        bases[key] = row
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["method"])].append(row)
    aggregates = {}
    for key, group_rows in grouped.items():
        base_key = group_rows[0]["base_snapshot_integrity_sha256"]
        base = bases.get(base_key)
        if base is not None:
            for field in (
                "evaluation_protocol_config_hash",
                "dataset_evidence",
                "evaluator",
                "code_evaluation_packages",
                "evaluation_software_environment",
            ):
                if base[field] != group_rows[0][field]:
                    raise ValueError(
                        "Shared-base and adapted EvalPlus protocols differ"
                    )
        aggregates[key] = _aggregate_group(
            group_rows,
            expected_seeds=expected_seeds,
            bootstrap_samples=args.bootstrap_samples,
            base_metrics=base["metrics"] if base is not None else None,
        )

    pairwise = []
    for model_id in sorted({key[0] for key in aggregates}):
        reference_key = (model_id, args.reference_method)
        if reference_key not in aggregates:
            continue
        reference = aggregates[reference_key]
        reference_rows = {
            int(row["seed"]): row for row in grouped[reference_key]
        }
        for key in sorted(aggregates):
            if key[0] != model_id or key == reference_key:
                continue
            comparison = aggregates[key]
            _validate_paired_compatibility(reference, comparison)
            comparison_rows = {
                int(row["seed"]): row for row in grouped[key]
            }
            pairwise.append(
                {
                    "model_id": model_id,
                    "reference_method": args.reference_method,
                    "comparison_method": key[1],
                    "difference_definition": "reference_minus_comparison",
                    "metrics": {
                        metric: summarize_values(
                            paired_differences(
                                {
                                    seed: reference_rows[seed]["metrics"][
                                        metric
                                    ]
                                    for seed in expected_seeds
                                },
                                {
                                    seed: comparison_rows[seed]["metrics"][
                                        metric
                                    ]
                                    for seed in expected_seeds
                                },
                            ),
                            bootstrap_samples=args.bootstrap_samples,
                            bootstrap_seed=20260725,
                        )
                        for metric in METRICS
                    },
                }
            )

    output = Path(args.output).resolve()
    protected = {
        Path(row[field]).resolve()
        for row in [*rows, *base_rows]
        for field in ("run_dir", "generation_dir")
    }
    if any(_is_within(output, root) for root in protected):
        raise ValueError(
            "Aggregation output must be outside protected EvalPlus directories"
        )
    payload = {
        "format_version": 1,
        "formal_result_eligible": all(
            group["formal_result_eligible"]
            for group in aggregates.values()
        ),
        "expected_seeds": expected_seeds,
        "bootstrap_scope": (
            "seed-level mean uncertainty; task-level paired uncertainty is "
            "reported separately when task logs support it"
        ),
        "aggregation_source": aggregation_source,
        "aggregation_software_environment": aggregation_environment,
        "code_evaluation_packages": package_versions,
        "shared_base_runs": base_rows,
        "groups": [aggregates[key] for key in sorted(aggregates)],
        "paired_comparisons": pairwise,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
