import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


EVALPLUS_DATA_MANIFEST_FORMAT_VERSION = 1
EVALPLUS_DATASETS = {
    "humaneval": {
        "release": "HumanEvalPlus",
        "version": "v0.1.10",
        "url": (
            "https://github.com/evalplus/humanevalplus_release/releases/"
            "download/v0.1.10/HumanEvalPlus.jsonl.gz"
        ),
        "compressed_file": "HumanEvalPlus.jsonl.gz",
        "compressed_size_bytes": 925932,
        "compressed_sha256": (
            "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
        ),
        "jsonl_file": "HumanEvalPlus.jsonl",
        "jsonl_size_bytes": 7714666,
        "jsonl_sha256": (
            "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
        ),
        "jsonl_md5": "fe585eb4df8c88d844eeb463ea4d0302",
        "rows": 164,
        "task_prefix": "HumanEval/",
    },
    "mbpp": {
        "release": "MbppPlus",
        "version": "v0.2.0",
        "url": (
            "https://github.com/evalplus/mbppplus_release/releases/"
            "download/v0.2.0/MbppPlus.jsonl.gz"
        ),
        "compressed_file": "MbppPlus.jsonl.gz",
        "compressed_size_bytes": 336032,
        "compressed_sha256": (
            "af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63"
        ),
        "jsonl_file": "MbppPlus.jsonl",
        "jsonl_size_bytes": 2592369,
        "jsonl_sha256": (
            "b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad"
        ),
        "jsonl_md5": "ee43ecabebf20deef4bb776a405ac5b1",
        "rows": 378,
        "task_prefix": "Mbpp/",
    },
}


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    return _digest(path, "sha256")


def md5_file(path: Path) -> str:
    return _digest(path, "md5")


def stream_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    handle = (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8")
    )
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"EvalPlus row {line_number} in {path} is not an object"
                )
            yield row


def validate_evalplus_dataset_file(
    path: Path,
    *,
    dataset: str,
    compressed: bool,
) -> Dict[str, Any]:
    if dataset not in EVALPLUS_DATASETS:
        raise ValueError(f"Unknown EvalPlus dataset: {dataset}")
    specification = EVALPLUS_DATASETS[dataset]
    expected_size = int(
        specification[
            "compressed_size_bytes" if compressed else "jsonl_size_bytes"
        ]
    )
    expected_sha = str(
        specification["compressed_sha256" if compressed else "jsonl_sha256"]
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing EvalPlus dataset file: {path}")
    if path.stat().st_size != expected_size:
        raise ValueError(f"EvalPlus dataset size mismatch: {path}")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"EvalPlus dataset SHA-256 mismatch: {path}")

    required = {
        "task_id",
        "prompt",
        "contract",
        "canonical_solution",
        "entry_point",
        "base_input",
        "plus_input",
        "atol",
    }
    task_ids = []
    for row in stream_jsonl(path):
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"EvalPlus row lacks fields {sorted(missing)}: {path}"
            )
        task_id = str(row["task_id"])
        if not task_id.startswith(str(specification["task_prefix"])):
            raise ValueError(f"Unexpected EvalPlus task ID: {task_id}")
        base_inputs_valid = isinstance(row["base_input"], list)
        plus_inputs_valid = isinstance(row["plus_input"], list)
        if (
            dataset == "mbpp"
            and task_id == "Mbpp/793"
            and row["plus_input"] == {}
        ):
            plus_inputs_valid = True
        if not base_inputs_valid or not plus_inputs_valid:
            raise ValueError(
                f"Unexpected EvalPlus input encoding for task {task_id}"
            )
        task_ids.append(task_id)
    if len(task_ids) != int(specification["rows"]):
        raise ValueError(f"EvalPlus row-count mismatch: {path}")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"Duplicate EvalPlus task IDs: {path}")
    record = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": expected_sha,
        "rows": len(task_ids),
        "task_ids_sha256": hashlib.sha256(
            json.dumps(task_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    if not compressed:
        actual_md5 = md5_file(path)
        if actual_md5 != specification["jsonl_md5"]:
            raise ValueError(f"EvalPlus dataset MD5 mismatch: {path}")
        record["md5"] = actual_md5
    return record


def validate_evalplus_data_manifest(
    manifest_path: str,
    *,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    if (
        payload.get("format_version")
        != EVALPLUS_DATA_MANIFEST_FORMAT_VERSION
        or payload.get("source") != "official_github_release"
    ):
        raise ValueError("Unsupported EvalPlus data manifest")
    records = payload.get("datasets")
    if not isinstance(records, Mapping) or set(records) != set(EVALPLUS_DATASETS):
        raise ValueError("EvalPlus data manifest must contain both datasets")
    if config.get("datasets") != EVALPLUS_DATASETS:
        raise ValueError("EvalPlus config does not match the locked datasets")

    verified = {}
    for dataset, specification in EVALPLUS_DATASETS.items():
        record = records[dataset]
        if any(
            record.get(key) != specification[key]
            for key in ("release", "version", "url")
        ):
            raise ValueError(
                f"EvalPlus manifest identity mismatch: {dataset}"
            )
        compressed_record = validate_evalplus_dataset_file(
            Path(record["compressed"]["path"]).resolve(),
            dataset=dataset,
            compressed=True,
        )
        jsonl_record = validate_evalplus_dataset_file(
            Path(record["jsonl"]["path"]).resolve(),
            dataset=dataset,
            compressed=False,
        )
        if (
            record["compressed"] != compressed_record
            or record["jsonl"] != jsonl_record
        ):
            raise ValueError(
                f"EvalPlus manifest file evidence mismatch: {dataset}"
            )
        verified[dataset] = {
            "compressed": compressed_record,
            "jsonl": jsonl_record,
        }
    return {
        "format_version": EVALPLUS_DATA_MANIFEST_FORMAT_VERSION,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "source": payload["source"],
        "datasets": verified,
    }
