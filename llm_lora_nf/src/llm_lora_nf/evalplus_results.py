import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .evalplus_evidence import EVALPLUS_DATASETS, stream_jsonl


def dataset_task_ids(dataset_path: Path) -> List[str]:
    return [str(row["task_id"]) for row in stream_jsonl(dataset_path)]


def validate_evalplus_samples(
    samples_path: Path,
    *,
    dataset_path: Path,
    allow_subset: bool = False,
) -> Dict[str, Any]:
    expected = dataset_task_ids(dataset_path)
    rows = list(stream_jsonl(samples_path))
    task_ids = [str(row.get("task_id", "")) for row in rows]
    if any(not isinstance(row.get("solution"), str) for row in rows):
        raise ValueError(
            f"EvalPlus samples contain a non-string solution: {samples_path}"
        )
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"EvalPlus samples contain duplicate task IDs: {samples_path}")
    if allow_subset:
        if not task_ids or task_ids != expected[: len(task_ids)]:
            raise ValueError(
                f"EvalPlus smoke samples are not an ordered prefix: {samples_path}"
            )
    elif task_ids != expected:
        raise ValueError(
            f"EvalPlus samples do not cover every task exactly: {samples_path}"
        )
    encoded_ids = json.dumps(task_ids, separators=(",", ":")).encode("utf-8")
    return {
        "path": str(samples_path.resolve()),
        "rows": len(rows),
        "sha256": _sha256(samples_path),
        "task_ids_sha256": hashlib.sha256(encoded_ids).hexdigest(),
        "empty_solution_count": sum(
            not row["solution"].strip() for row in rows
        ),
    }


def validate_evalplus_result(
    result_path: Path,
    *,
    dataset: str,
    dataset_path: Path,
    expected_samples: int = 1,
) -> Dict[str, Any]:
    if dataset not in EVALPLUS_DATASETS:
        raise ValueError(f"Unknown EvalPlus dataset: {dataset}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("eval"), dict):
        raise ValueError(f"Malformed EvalPlus result: {result_path}")
    expected_ids = dataset_task_ids(dataset_path)
    if payload.get("hash") != EVALPLUS_DATASETS[dataset]["jsonl_md5"]:
        raise ValueError(f"EvalPlus result dataset hash mismatch: {result_path}")
    if set(payload["eval"]) != set(expected_ids):
        raise ValueError(f"EvalPlus result task coverage mismatch: {result_path}")

    allowed_statuses = {"pass", "fail", "timeout", None}
    base_passes = 0
    plus_passes = 0
    base_statuses: Counter = Counter()
    plus_statuses: Counter = Counter()
    for task_id in expected_ids:
        task_results = payload["eval"][task_id]
        if not isinstance(task_results, list) or len(task_results) != expected_samples:
            raise ValueError(
                f"EvalPlus result sample count mismatch for {task_id}"
            )
        for result in task_results:
            if result.get("task_id") != task_id:
                raise ValueError(
                    f"EvalPlus embedded task identity mismatch for {task_id}"
                )
            base_status = result.get("base_status")
            plus_status = result.get("plus_status")
            if (
                base_status not in allowed_statuses
                or plus_status not in allowed_statuses
            ):
                raise ValueError(
                    f"Unknown EvalPlus status for {task_id}: "
                    f"{base_status}/{plus_status}"
                )
            base_statuses[str(base_status)] += 1
            plus_statuses[str(plus_status)] += 1
            base_passes += int(base_status == "pass")
            plus_passes += int(
                base_status == "pass" and plus_status == "pass"
            )

    denominator = len(expected_ids) * expected_samples
    return {
        "dataset": dataset,
        "tasks": len(expected_ids),
        "samples_per_task": expected_samples,
        "dataset_md5": payload["hash"],
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "base_pass_at_1": base_passes / denominator,
        "plus_pass_at_1": plus_passes / denominator,
        "base_status_counts": dict(sorted(base_statuses.items())),
        "plus_status_counts": dict(sorted(plus_statuses.items())),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
