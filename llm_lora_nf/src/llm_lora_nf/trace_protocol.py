import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


TRACE_OFFICIAL_REPOSITORY = "https://github.com/BeyonderXX/TRACE"
TRACE_OFFICIAL_COMMIT = "462e39f616134f4f819efeb3baea8638c03c7db4"
TRACE_OFFICIAL_ORDER = (
    "C-STANCE",
    "FOMC",
    "MeetingBank",
    "Py150",
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "20Minuten",
)
TRACE_ALTERNATE_ORDER = (
    "NumGLUE-cm",
    "NumGLUE-ds",
    "FOMC",
    "20Minuten",
    "C-STANCE",
    "Py150",
    "MeetingBank",
    "ScienceQA",
)
TRACE_LORA_EPOCHS = {
    "C-STANCE": 5,
    "FOMC": 3,
    "MeetingBank": 7,
    "Py150": 5,
    "ScienceQA": 3,
    "NumGLUE-cm": 5,
    "NumGLUE-ds": 5,
    "20Minuten": 7,
}
TRACE_PRIMARY_METRICS = {
    "C-STANCE": "accuracy",
    "FOMC": "accuracy",
    "MeetingBank": "rouge-L",
    "Py150": "similarity",
    "ScienceQA": "accuracy",
    "NumGLUE-cm": "accuracy",
    "NumGLUE-ds": "accuracy",
    "20Minuten": "sari",
}
TRACE_PRIMARY_METRIC_SCALES = {
    "C-STANCE": 1.0,
    "FOMC": 1.0,
    "MeetingBank": 1.0,
    "Py150": 100.0,
    "ScienceQA": 1.0,
    "NumGLUE-cm": 1.0,
    "NumGLUE-ds": 1.0,
    "20Minuten": 100.0,
}
TRACE_SPLITS = ("train", "eval", "test")

TRACE_TREELORA_PILOT_SOURCE = {
    "repository": "https://github.com/ZinYY/TreeLoRA",
    "commit": "1c7260c42b34e1961283797c742f08b9c3842501",
    "archive_path": (
        "data/LLM-CL-Benchmark/LLM-CL-Benchmark_500.tar.xz"
    ),
    "archive_sha256": (
        "956caf12b59add0c7d961cf8ecbad0307e1abca8db7de8873c37d92dd709e9c2"
    ),
}

# This mirror is deliberately registered as a chain/pilot dataset, not as the
# paper protocol.  The exact identities make accidental promotion to a formal
# result impossible.
TRACE_TREELORA_PILOT_FILES = {
    "C-STANCE": {
        "train": (
            500,
            "646ccbb96141b50fc8d888a7da51c1f3ba448ef75f35552f7e2b235073835dc9",
        ),
        "eval": (
            100,
            "175efa6b3ace29e9a1b8d30482c8e5e762e11a46f2cb6f4d02c12eefb9fc5a66",
        ),
        "test": (
            100,
            "fb29ad754a99f811016e2d797d285c9aac9af954c5d4003f24a43a8c007f8531",
        ),
    },
    "FOMC": {
        "train": (
            500,
            "ad55548875b28afb37d5e6af0ed241887600b6a5ae5272e04ba687d8d5d0ca5f",
        ),
        "eval": (
            100,
            "bc258307c34bb277b5158713e9ec13fc20547b53c306413fddca7da7fc10c255",
        ),
        "test": (
            100,
            "8b257fcda84600ceb000235d3af87b5ef754694d04d81c9e60bda2a632e608ba",
        ),
    },
    "MeetingBank": {
        "train": (
            500,
            "807d84a23285ee412d6367c81661bc72de8fc5dbaa3508e1f5b306d83153a701",
        ),
        "eval": (
            100,
            "2aa60d6d97043f52a9ab56f0b4bad3cc1c6b581a6b24bc3ee33ff543a8496c90",
        ),
        "test": (
            100,
            "11715340354cecc0be52dee54b6261cac2bc87f314f39e46a7db506d4f53f7d2",
        ),
    },
    "Py150": {
        "train": (
            500,
            "8a5439ee66d957f984a3518b3e8aff6a008075c00c9ca21017acf60727bba06c",
        ),
        "eval": (
            100,
            "029302d12cda5332a850e2216e3a388aca61e28aa3e6a7b401ab09ca43838c70",
        ),
        "test": (
            100,
            "418a4767caad60dd824ea20545246ddb7b597bed1529561abe7b1ae7fa65e9b4",
        ),
    },
    "ScienceQA": {
        "train": (
            500,
            "8f796547542ffdf89397d4750486d95c9e8fdbe16d6402b9ba6e3d44900d0599",
        ),
        "eval": (
            100,
            "e4cf69626f1163bed5825b34cae12542b1782698b60b0f67ba6e9cb5b4067195",
        ),
        "test": (
            100,
            "ed1314220ee5f3700fc4b7e93efd5810ad09ee2ce44c9b68a12c648a5ac3e3f9",
        ),
    },
    "NumGLUE-cm": {
        "train": (
            500,
            "722bbb05819949ff1496a6558f451289e64e52a3f6c34ed81c8a8889e2ce5c08",
        ),
        "eval": (
            41,
            "87edcff354115a4c96d6eabb09f2b9c669ad895f9c5b91ff88c48bf985ca929f",
        ),
        "test": (
            81,
            "52a3fd70e14995309cc74696f82329f2991fffe5456c805f1a76ed3c0005da7c",
        ),
    },
    "NumGLUE-ds": {
        "train": (
            500,
            "4eaff949885eef27ec9621db708452663cb24bddb478cd12b8fd66031b804b80",
        ),
        "eval": (
            100,
            "c787627006e4f615f738df3b7d210752b6fe826f1cb1cbcda30a58587959b36e",
        ),
        "test": (
            100,
            "8d74f9d2406df7905b85ea20e2ccce357d6399898273e3a594a800fb82180fe1",
        ),
    },
    "20Minuten": {
        "train": (
            500,
            "f83be5bd734942d55f5f26c75b58173d4de56040140ac4e04a691173ede30031",
        ),
        "eval": (
            100,
            "16e12c727a03247162a816fbc55ed4dbcba76471b71357c3c057232db53aa4ae",
        ),
        "test": (
            100,
            "c9ec851c744e3ae872c24023971ce3d97b96210a17eb8fae1346af8a064e5861",
        ),
    },
}


@dataclass(frozen=True)
class TraceFileRecord:
    task: str
    split: str
    path: str
    rows: int
    size_bytes: int
    sha256: str
    canonical_rows_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_trace_rows(path: Path) -> List[Mapping[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"TRACE split must be a JSON array: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"TRACE row is not an object: {path}:{index}")
        if set(row) != {"prompt", "answer"}:
            raise ValueError(
                f"TRACE row must contain exactly prompt/answer: {path}:{index}"
            )
        if not isinstance(row["prompt"], str) or not isinstance(row["answer"], str):
            raise ValueError(
                f"TRACE prompt/answer must be strings: {path}:{index}"
            )
        if not row["prompt"]:
            raise ValueError(f"TRACE prompt must be non-empty: {path}:{index}")
    return rows


def audit_trace_dataset(
    root: str,
    *,
    protocol: str,
) -> Dict[str, Any]:
    """Audit a TRACE directory without changing or re-ordering its rows.

    ``treelora_500_pilot`` is a fixed, non-formal chain dataset.  ``paper_5k``
    checks the paper-level cardinalities but intentionally refuses to qualify
    an unregistered mirror: callers must also bind the returned file records
    to an externally reviewed, exact source manifest before formal training.
    """

    dataset_root = Path(root).resolve()
    if protocol not in {"treelora_500_pilot", "paper_5k"}:
        raise ValueError(f"Unsupported TRACE protocol: {protocol}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"TRACE root is not a directory: {dataset_root}")

    records: List[TraceFileRecord] = []
    for task in TRACE_OFFICIAL_ORDER:
        task_root = dataset_root / task
        if not task_root.is_dir():
            raise FileNotFoundError(f"Missing TRACE task directory: {task_root}")
        for split in TRACE_SPLITS:
            path = task_root / f"{split}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing TRACE split: {path}")
            rows = _load_trace_rows(path)
            sha256 = _sha256_file(path)
            if protocol == "treelora_500_pilot":
                expected_rows, expected_sha256 = TRACE_TREELORA_PILOT_FILES[
                    task
                ][split]
                if len(rows) != expected_rows or sha256 != expected_sha256:
                    raise ValueError(
                        f"TRACE pilot identity mismatch: {task}/{split}"
                    )
            records.append(
                TraceFileRecord(
                    task=task,
                    split=split,
                    path=path.relative_to(dataset_root).as_posix(),
                    rows=len(rows),
                    size_bytes=int(path.stat().st_size),
                    sha256=sha256,
                    canonical_rows_sha256=_canonical_rows_sha256(rows),
                )
            )

    if protocol == "paper_5k":
        by_task = {
            task: {
                record.split: record
                for record in records
                if record.task == task
            }
            for task in TRACE_OFFICIAL_ORDER
        }
        for task, splits in by_task.items():
            if splits["train"].rows != 5000:
                raise ValueError(
                    f"TRACE paper protocol requires 5000 train rows: {task}"
                )
            if splits["eval"].rows <= 0:
                raise ValueError(
                    f"TRACE paper protocol requires a non-empty eval split: {task}"
                )
            if splits["test"].rows != 2000:
                raise ValueError(
                    "TRACE paper protocol requires 2000 test rows: "
                    f"{task}"
                )
        formal_eligible = False
        qualification = (
            "cardinality_only_requires_locked_exact_source_manifest"
        )
    else:
        formal_eligible = False
        qualification = "chain_and_state_ablation_pilot_only"

    payload = {
        "protocol": protocol,
        "root": str(dataset_root),
        "formal_eligible": formal_eligible,
        "qualification": qualification,
        "official_repository": TRACE_OFFICIAL_REPOSITORY,
        "official_evaluator_commit": TRACE_OFFICIAL_COMMIT,
        "task_order": list(TRACE_OFFICIAL_ORDER),
        "alternate_task_order": list(TRACE_ALTERNATE_ORDER),
        "primary_metrics": dict(TRACE_PRIMARY_METRICS),
        "files": [asdict(record) for record in records],
    }
    if protocol == "treelora_500_pilot":
        payload["source"] = dict(TRACE_TREELORA_PILOT_SOURCE)
    return payload


def normalize_trace_primary_score(task: str, raw_score: float) -> float:
    if task not in TRACE_PRIMARY_METRICS:
        raise KeyError(f"Unknown TRACE task: {task}")
    normalized = float(raw_score) / TRACE_PRIMARY_METRIC_SCALES[task]
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"Normalized TRACE score lies outside [0, 1]: {task}={normalized}"
        )
    return normalized


def validate_trace_score_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    task_count: Optional[int] = None,
) -> List[List[float]]:
    rows = [list(map(float, row)) for row in matrix]
    if task_count is None:
        task_count = len(rows)
    if task_count <= 0:
        raise ValueError("TRACE matrix must contain at least one task")
    if len(rows) != task_count:
        raise ValueError(f"TRACE matrix must have {task_count} rows")
    for time_index, row in enumerate(rows):
        expected = time_index + 1
        if len(row) != expected:
            raise ValueError(
                f"TRACE matrix row {time_index} must have {expected} entries"
            )
        if any(not 0.0 <= value <= 1.0 for value in row):
            raise ValueError("TRACE matrix entries must lie in [0, 1]")
    return rows


def aggregate_trace_matrix(
    matrix: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Compute paper OP/BWT plus final average and standard forgetting.

    The TRACE paper defines BWT at time ``t`` with denominator ``t`` rather
    than the more common ``t-1``.  We preserve that definition and report
    standard final forgetting separately.
    """

    rows = validate_trace_score_matrix(matrix)
    task_count = len(rows)
    op_by_time = [sum(row) / len(row) for row in rows]
    bwt_by_time = [0.0]
    for time_index in range(1, task_count):
        differences = [
            rows[time_index][task] - rows[task][task]
            for task in range(time_index)
        ]
        bwt_by_time.append(sum(differences) / (time_index + 1))

    final_row = rows[-1]
    forgetting_by_task = []
    for task in range(task_count - 1):
        historical = [rows[time][task] for time in range(task, task_count)]
        forgetting_by_task.append(max(historical) - final_row[task])

    return {
        "matrix": rows,
        "op_by_time": op_by_time,
        "bwt_by_time": bwt_by_time,
        "final_average": sum(final_row) / task_count,
        "final_bwt_trace_definition": bwt_by_time[-1],
        "final_forgetting_standard": (
            sum(forgetting_by_task) / len(forgetting_by_task)
            if forgetting_by_task
            else 0.0
        ),
        "forgetting_by_task": forgetting_by_task,
        "bwt_denominator": "tasks_seen_including_current",
        "forgetting_definition": (
            "mean_old_tasks(max_score_from_learning_through_final-final_score)"
        ),
    }


def assert_task_order(order: Iterable[str]) -> str:
    normalized = tuple(order)
    if normalized == TRACE_OFFICIAL_ORDER:
        return "official"
    if normalized == TRACE_ALTERNATE_ORDER:
        return "alternate"
    raise ValueError("TRACE task order is not one of the two preregistered orders")
