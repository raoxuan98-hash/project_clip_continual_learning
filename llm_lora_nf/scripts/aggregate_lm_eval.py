#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.evaluator_tasks import PINNED_TASK_SOURCES
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


def _validated_run(
    run_dir: Path,
    *,
    allow_cpu_smoke: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    manifest_path = run_dir / "evaluation_run.json"
    run_manifest = _load_json(manifest_path)
    output_integrity = validate_directory_integrity(
        str(run_dir),
        expected_kind="evaluation_output",
    )
    if (
        run_manifest.get("format_version") != 2
        or run_manifest.get("status") != "passed"
        or run_manifest.get("returncode") != 0
    ):
        raise ValueError(f"Evaluation run did not pass: {run_dir}")
    formal = bool(run_manifest.get("formal_result_eligible"))
    if not formal and not allow_cpu_smoke:
        raise ValueError("CPU smoke outputs cannot enter formal aggregation")
    if formal:
        if (
            run_manifest.get("execution_mode") != "gpu_formal"
            or run_manifest.get("limit") is not None
            or not run_manifest.get("source", {}).get("commit_sha")
            or run_manifest.get("source", {}).get("source_dirty")
        ):
            raise ValueError(
                f"Formal evaluation evidence is not qualified: {run_dir}"
            )
    model_integrity = run_manifest.get("model_integrity")
    if not isinstance(model_integrity, dict) or not model_integrity.get(
        "manifest_sha256"
    ):
        raise ValueError(f"Run lacks evaluated-model integrity: {run_dir}")
    model_role = run_manifest.get("model_role")
    if model_role not in {"base", "merged"}:
        raise ValueError(f"Unsupported evaluated model role: {model_role!r}")
    expected_model_kind = (
        "merged_checkpoint" if model_role == "merged" else "model_snapshot"
    )
    if model_integrity.get("kind") != expected_model_kind:
        raise ValueError(f"Evaluated-model integrity kind mismatch: {run_dir}")
    dataset_evidence = run_manifest.get("dataset_evidence")
    if (
        not isinstance(dataset_evidence, dict)
        or not dataset_evidence.get("manifest_sha256")
        or int(dataset_evidence.get("verified_cache_files", 0)) <= 0
    ):
        raise ValueError(f"Run lacks verified evaluation-data evidence: {run_dir}")
    task_definitions = run_manifest.get("task_definitions")
    if (
        not isinstance(task_definitions, dict)
        or not task_definitions.get("tree_sha256")
        or Path(task_definitions.get("root", "")).resolve()
        != (run_dir / "task_definitions").resolve()
    ):
        raise ValueError(f"Run lacks pinned evaluator task evidence: {run_dir}")
    request_cache = run_manifest.get("request_cache")
    request_cache_identity = (
        request_cache.get("identity")
        if isinstance(request_cache, dict)
        else None
    )
    if (
        not isinstance(request_cache, dict)
        or not isinstance(request_cache_identity, dict)
        or request_cache_identity.get("format_version") != 2
        or not request_cache.get("key")
        or request_cache.get("retention")
        != "shared_content_addressed_per_base_model_protocol"
        or not request_cache.get("integrity", {}).get("tree_sha256")
        or hashlib.sha256(
            json.dumps(
                request_cache_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        != request_cache.get("key")
        or Path(request_cache.get("path", "")).name
        != request_cache.get("key")
        or request_cache_identity.get("evaluator_seeds")
        != run_manifest.get("evaluator_seeds")
        or request_cache_identity.get("software_environment")
        != software_environment_identity(
            run_manifest.get("environment", {})
        )
        or bool(request_cache_identity.get("cache_requests_enabled"))
        != formal
    ):
        raise ValueError(f"Run lacks shared request-cache identity: {run_dir}")
    return run_manifest, output_integrity


def _model_identity(
    run_manifest: Dict[str, Any],
    model_manifest: Dict[str, Any],
) -> Dict[str, str]:
    role = str(run_manifest.get("model_role"))
    evaluated_integrity = str(
        run_manifest["model_integrity"]["manifest_sha256"]
    )
    if role == "merged":
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
                "Formal merged evaluation lacks a qualified FP32 export"
            )
        base_model = model_manifest.get("base_model", {})
        base_integrity = base_model.get("snapshot_integrity_sha256")
        provenance = base_model.get("snapshot_manifest_sha256")
        export_base_integrity = model_manifest.get(
            "base_model_integrity_manifest_sha256"
        )
        if (
            not base_integrity
            or not provenance
            or export_base_integrity != base_integrity
        ):
            raise ValueError(
                "Merged export does not bind one validated base snapshot"
            )
        return {
            "role": role,
            "model_id": str(base_model["requested_id"]),
            "source_id": str(base_model["source_id"]),
            "base_snapshot_integrity_sha256": str(base_integrity),
            "base_snapshot_manifest_sha256": str(provenance),
            "evaluated_checkpoint_integrity_sha256": evaluated_integrity,
        }
    if role == "base":
        provenance = run_manifest.get("model_manifest_sha256")
        if not provenance:
            raise ValueError("Base evaluation lacks snapshot provenance hash")
        return {
            "role": role,
            "model_id": str(model_manifest["model_id"]),
            "source_id": str(model_manifest["model_id"]),
            "base_snapshot_integrity_sha256": evaluated_integrity,
            "base_snapshot_manifest_sha256": str(provenance),
            "evaluated_checkpoint_integrity_sha256": evaluated_integrity,
        }
    raise ValueError(f"Unsupported evaluated model role: {role!r}")


def _find_result(run_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    candidates = []
    for path in run_dir.rglob("*.json"):
        if path.name == "evaluation_run.json" or "samples_" in path.name:
            continue
        try:
            payload = _load_json(path)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload.get("results"), dict):
            candidates.append((path, payload))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one lm-eval result JSON in {run_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _metric(task_payload: Dict[str, Any], requested: str) -> Tuple[str, float]:
    if requested in task_payload:
        return requested, float(task_payload[requested])
    prefix = requested.split(",", 1)[0]
    matches = [
        key
        for key, value in task_payload.items()
        if key.split(",", 1)[0] == prefix
        and not key.endswith("_stderr")
        and isinstance(value, (int, float))
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Metric {requested!r} unavailable; candidates are {sorted(matches)}"
        )
    key = matches[0]
    return key, float(task_payload[key])


def _extract_scores(
    result: Dict[str, Any],
    metrics: Dict[str, str],
    tasks: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    scores = {}
    for task in tasks:
        if task not in result["results"]:
            raise KeyError(f"Task {task} missing from lm-eval results")
        metric_key, score = _metric(result["results"][task], metrics[task])
        if "," in metric_key:
            metric_name, filter_name = metric_key.split(",", 1)
            stderr_key = f"{metric_name}_stderr,{filter_name}"
        else:
            stderr_key = metric_key + "_stderr"
        scores[task] = {
            "metric": metric_key,
            "score": score,
            "stderr": result["results"][task].get(stderr_key),
        }
    return scores


def _validate_result_task_configs(
    result: Dict[str, Any],
    config: Dict[str, Any],
    tasks: Iterable[str],
) -> None:
    result_configs = result.get("configs")
    if not isinstance(result_configs, dict):
        raise ValueError("lm-eval result lacks resolved task configurations")
    for task in tasks:
        task_config = result_configs.get(task)
        if not isinstance(task_config, dict):
            raise ValueError(f"lm-eval result lacks task config for {task}")
        dataset_id = PINNED_TASK_SOURCES[task]["dataset_id"]
        specification = config["datasets"][dataset_id]
        if (
            task_config.get("dataset_path") != dataset_id
            or task_config.get("dataset_name") != specification.get("config")
            or task_config.get("dataset_kwargs", {}).get("revision")
            != str(specification["revision"])
        ):
            raise ValueError(
                f"lm-eval task {task} did not use the locked dataset revision"
            )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("Cannot average an empty collection")
    return sum(materialized) / len(materialized)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate qualified lm-eval math/knowledge results"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-run-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    if _is_within(output, run_dir):
        raise ValueError(
            "Aggregation output must be outside the integrity-protected "
            "evaluation directory"
        )
    run_manifest, run_output_integrity = _validated_run(
        run_dir,
        allow_cpu_smoke=args.allow_cpu_smoke,
    )
    if run_manifest["formal_result_eligible"]:
        validate_formal_evaluation_config(config)
    protocol_config = copy.deepcopy(config)
    protocol_config["run"].pop("seed", None)
    protocol_config["run"].pop("name", None)
    expected_protocol_hash = canonical_config_hash(protocol_config)
    if run_manifest.get("protocol_config_hash") != expected_protocol_hash:
        raise ValueError(
            "Evaluation run does not match the supplied locked protocol config"
        )
    model_manifest = run_manifest.get("model_manifest")
    if not isinstance(model_manifest, dict):
        if run_manifest.get("formal_result_eligible"):
            raise ValueError(
                "Formal evaluation must embed its model manifest so ephemeral "
                "merged checkpoints can be deleted"
            )
        model_manifest = _load_json(
            Path(run_manifest["model_manifest_path"]).resolve()
        )
    model_identity = _model_identity(run_manifest, model_manifest)
    result_path, result = _find_result(run_dir)
    configured_tasks = [
        *config["tasks"]["downstream"],
        *config["tasks"]["retention"],
    ]
    run_tasks = list(run_manifest["tasks"])
    if len(run_tasks) != len(set(run_tasks)):
        raise ValueError("Run manifest contains duplicate tasks")
    if not set(run_tasks).issubset(configured_tasks):
        raise ValueError("Run manifest contains tasks outside the locked protocol")
    if (
        run_manifest["formal_result_eligible"]
        and run_tasks != configured_tasks
    ):
        raise ValueError(
            "Formal aggregation requires every locked task in protocol order"
        )
    _validate_result_task_configs(result, config, run_tasks)
    scores = _extract_scores(result, config["metrics"], run_tasks)
    downstream_tasks = [
        task for task in config["tasks"]["downstream"] if task in run_tasks
    ]
    retention_tasks = [
        task for task in config["tasks"]["retention"] if task in run_tasks
    ]
    downstream_average = (
        _mean(scores[task]["score"] for task in downstream_tasks)
        if downstream_tasks
        else None
    )
    retention_average = (
        _mean(scores[task]["score"] for task in retention_tasks)
        if retention_tasks
        else None
    )

    base_comparison = None
    all_inputs_formal = bool(run_manifest["formal_result_eligible"])
    if args.base_run_dir:
        if model_identity["role"] != "merged":
            raise ValueError(
                "A base comparison is only defined for an adapted merged model"
            )
        base_dir = Path(args.base_run_dir).resolve()
        if _is_within(output, base_dir):
            raise ValueError(
                "Aggregation output must be outside the base evaluation directory"
            )
        base_manifest, base_output_integrity = _validated_run(
            base_dir,
            allow_cpu_smoke=args.allow_cpu_smoke,
        )
        all_inputs_formal = bool(
            all_inputs_formal
            and base_manifest["formal_result_eligible"]
        )
        if base_manifest.get("model_role") != "base":
            raise ValueError("Base comparison requires model_role=base")
        for key in (
            "evaluator_revision",
            "evaluator_seeds",
            "tasks",
            "dataset_evidence",
            "source",
        ):
            if base_manifest.get(key) != run_manifest.get(key):
                raise ValueError(f"Base and adapted runs disagree on {key}")
        run_protocol_hash = run_manifest.get("protocol_config_hash")
        base_protocol_hash = base_manifest.get("protocol_config_hash")
        if run_protocol_hash != base_protocol_hash:
            raise ValueError(
                "Base and adapted runs use different evaluation protocols"
            )
        if (
            run_manifest["task_definitions"]["tree_sha256"]
            != base_manifest["task_definitions"]["tree_sha256"]
        ):
            raise ValueError(
                "Base and adapted runs use different evaluator task definitions"
            )
        if (
            run_manifest["request_cache"]["key"]
            != base_manifest["request_cache"]["key"]
            or run_manifest["request_cache"]["integrity"]["tree_sha256"]
            != base_manifest["request_cache"]["integrity"]["tree_sha256"]
        ):
            raise ValueError(
                "Base and adapted runs use different request-cache identities"
            )
        base_model_manifest = base_manifest.get("model_manifest")
        if not isinstance(base_model_manifest, dict):
            if base_manifest.get("formal_result_eligible"):
                raise ValueError(
                    "Formal base evaluation must embed its model manifest"
                )
            base_model_manifest = _load_json(
                Path(base_manifest["model_manifest_path"]).resolve()
            )
        base_identity = _model_identity(base_manifest, base_model_manifest)
        if model_identity["source_id"] != base_identity["source_id"]:
            raise ValueError(
                "Base and adapted evaluations use different model snapshots"
            )
        if (
            model_identity["base_snapshot_integrity_sha256"]
            != base_identity["base_snapshot_integrity_sha256"]
            or model_identity["base_snapshot_manifest_sha256"]
            != base_identity["base_snapshot_manifest_sha256"]
        ):
            raise ValueError(
                "Base and adapted evaluations use different snapshot contents"
            )
        base_result_path, base_result = _find_result(base_dir)
        _validate_result_task_configs(base_result, config, run_tasks)
        base_scores = _extract_scores(base_result, config["metrics"], run_tasks)
        retention_ratios = {
            task: (
                scores[task]["score"] / base_scores[task]["score"]
                if base_scores[task]["score"] > 0
                else None
            )
            for task in retention_tasks
        }
        defined_ratios = [
            value for value in retention_ratios.values() if value is not None
        ]
        base_comparison = {
            "base_run_manifest": str(base_dir / "evaluation_run.json"),
            "base_run_manifest_sha256": _sha256(
                base_dir / "evaluation_run.json"
            ),
            "base_evaluation_output_integrity": base_output_integrity,
            "base_lm_eval_result": str(base_result_path),
            "base_lm_eval_result_sha256": _sha256(base_result_path),
            "base_model_identity": base_identity,
            "base_scores": base_scores,
            "downstream_gain": (
                downstream_average
                - _mean(base_scores[task]["score"] for task in downstream_tasks)
                if downstream_average is not None
                else None
            ),
            "retention_ratios": retention_ratios,
            "retention_ratio_average": (
                _mean(defined_ratios) if defined_ratios else None
            ),
        }

    aggregation_source = git_source_state(
        Path(__file__).resolve().parents[2]
    )
    validate_derivation_source(
        aggregation_source,
        expected_commit=str(run_manifest["source"]["commit_sha"]),
        formal=all_inputs_formal,
    )
    aggregation_environment = software_environment_identity(
        runtime_environment(
            torch.device("cpu"),
            validate_formal=all_inputs_formal,
        )
    )
    if (
        all_inputs_formal
        and aggregation_environment
        != run_manifest["request_cache"]["identity"][
            "software_environment"
        ]
    ):
        raise ValueError(
            "Formal aggregation software environment differs from evaluation"
        )

    summary = {
        "format_version": 2,
        "formal_result_eligible": all_inputs_formal,
        "run_manifest": str(run_dir / "evaluation_run.json"),
        "run_manifest_sha256": _sha256(run_dir / "evaluation_run.json"),
        "evaluation_output_integrity": run_output_integrity,
        "lm_eval_result": str(result_path),
        "lm_eval_result_sha256": _sha256(result_path),
        "model_identity": model_identity,
        "dataset_evidence": run_manifest["dataset_evidence"],
        "task_definition_tree_sha256": run_manifest[
            "task_definitions"
        ]["tree_sha256"],
        "evaluator_seeds": run_manifest["evaluator_seeds"],
        "request_cache_key": run_manifest["request_cache"]["key"],
        "request_cache_tree_sha256": run_manifest[
            "request_cache"
        ]["integrity"]["tree_sha256"],
        "aggregation_source": aggregation_source,
        "aggregation_software_environment": aggregation_environment,
        "task_groups": {
            "downstream": downstream_tasks,
            "retention": retention_tasks,
        },
        "scores": scores,
        "downstream_average": downstream_average,
        "retention_average": retention_average,
        "skill_retention_geometric_mean": (
            math.sqrt(
                max(0.0, downstream_average) * max(0.0, retention_average)
            )
            if downstream_average is not None and retention_average is not None
            else None
        ),
        "base_comparison": base_comparison,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
