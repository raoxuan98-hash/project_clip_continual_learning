import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


EVAL_DATA_MANIFEST_FORMAT_VERSION = 1
LOCKED_HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def validate_eval_data_manifest(
    manifest_path: str,
    *,
    config: Mapping[str, Any],
    hf_home: str,
) -> Dict[str, Any]:
    path = Path(manifest_path).resolve()
    payload = _load_json(path)
    expected_home = Path(hf_home).resolve()
    if payload.get("format_version") != EVAL_DATA_MANIFEST_FORMAT_VERSION:
        raise ValueError(f"Unsupported evaluation-data manifest: {path}")
    if payload.get("endpoint") != LOCKED_HF_MIRROR_ENDPOINT:
        raise ValueError("Formal evaluation data must be prepared via HF-Mirror")
    if Path(payload.get("hf_home", "")).resolve() != expected_home:
        raise ValueError("Evaluation-data manifest HF_HOME does not match runtime")

    expected_datasets = config.get("datasets")
    records = payload.get("datasets")
    if not isinstance(expected_datasets, Mapping) or not isinstance(
        records,
        Mapping,
    ):
        raise ValueError("Evaluation-data manifest lacks dataset records")
    if set(records) != set(expected_datasets):
        raise ValueError(
            "Evaluation-data manifest must contain every locked dataset exactly"
        )

    verified_files = {}
    for dataset_id, specification in expected_datasets.items():
        record = records[dataset_id]
        if (
            record.get("config") != specification.get("config")
            or record.get("revision") != str(specification["revision"])
        ):
            raise ValueError(
                f"Evaluation dataset identity mismatch: {dataset_id}"
            )
        splits = record.get("splits")
        if not isinstance(splits, Mapping) or not splits:
            raise ValueError(f"Evaluation dataset has no splits: {dataset_id}")
        for split, split_record in splits.items():
            if (
                int(split_record.get("rows", 0)) <= 0
                or not split_record.get("fingerprint")
            ):
                raise ValueError(
                    f"Invalid evaluation dataset split: {dataset_id}/{split}"
                )
            cache_files = split_record.get("cache_files")
            if not isinstance(cache_files, list) or not cache_files:
                raise ValueError(
                    f"Dataset split lacks cache-file evidence: "
                    f"{dataset_id}/{split}"
                )
            for file_record in cache_files:
                relative = str(file_record.get("path", ""))
                cache_path = (expected_home / relative).resolve()
                try:
                    cache_path.relative_to(expected_home)
                except ValueError as error:
                    raise ValueError(
                        f"Dataset cache path escapes HF_HOME: {cache_path}"
                    ) from error
                if not cache_path.is_file():
                    raise FileNotFoundError(
                        f"Missing prepared evaluation cache file: {cache_path}"
                    )
                actual_size = int(cache_path.stat().st_size)
                expected_size = int(file_record.get("size_bytes", -1))
                if actual_size != expected_size:
                    raise ValueError(
                        f"Evaluation cache size mismatch: {cache_path}"
                    )
                expected_sha = str(file_record.get("sha256", ""))
                previous_sha = verified_files.get(cache_path)
                actual_sha = previous_sha or _sha256(cache_path)
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"Evaluation cache checksum mismatch: {cache_path}"
                    )
                verified_files[cache_path] = actual_sha

    return {
        "format_version": EVAL_DATA_MANIFEST_FORMAT_VERSION,
        "manifest_path": str(path),
        "manifest_sha256": _sha256(path),
        "endpoint": payload["endpoint"],
        "hf_home": str(expected_home),
        "datasets": len(records),
        "verified_cache_files": len(verified_files),
    }
