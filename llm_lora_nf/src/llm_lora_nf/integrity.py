import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


INTEGRITY_FORMAT_VERSION = 1
INTEGRITY_FILENAME = "artifact_integrity.json"
INTEGRITY_TEMP_FILENAME = f".{INTEGRITY_FILENAME}.tmp"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _inventory(root: Path) -> List[Dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {INTEGRITY_FILENAME, INTEGRITY_TEMP_FILENAME}:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"Artifact directory contains no files: {root}")
    return files


def write_directory_integrity(root: str, *, kind: str) -> Dict[str, Any]:
    artifact_root = Path(root).resolve()
    if not artifact_root.is_dir():
        raise NotADirectoryError(f"Artifact root is not a directory: {artifact_root}")
    if not kind:
        raise ValueError("Integrity kind must be non-empty")
    files = _inventory(artifact_root)
    manifest = {
        "format_version": INTEGRITY_FORMAT_VERSION,
        "kind": kind,
        "files": files,
        "tree_sha256": _canonical_hash(files),
    }
    encoded = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    temporary = artifact_root / INTEGRITY_TEMP_FILENAME
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(artifact_root / INTEGRITY_FILENAME)
    return {
        **manifest,
        "manifest_sha256": sha256_file(artifact_root / INTEGRITY_FILENAME),
        "manifest_path": str(artifact_root / INTEGRITY_FILENAME),
    }


def validate_directory_integrity(
    root: str,
    *,
    expected_kind: str,
) -> Dict[str, Any]:
    artifact_root = Path(root).resolve()
    manifest_path = artifact_root / INTEGRITY_FILENAME
    temporary_path = artifact_root / INTEGRITY_TEMP_FILENAME
    if temporary_path.exists():
        raise ValueError(
            f"Interrupted artifact-integrity write requires audit: {temporary_path}"
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing artifact integrity manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format_version") != INTEGRITY_FORMAT_VERSION
        or manifest.get("kind") != expected_kind
    ):
        raise ValueError(
            f"Artifact integrity kind/version mismatch: {manifest_path}"
        )
    expected_files = manifest.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError(f"Artifact integrity manifest contains no files: {manifest_path}")
    if manifest.get("tree_sha256") != _canonical_hash(expected_files):
        raise ValueError(f"Artifact integrity tree hash mismatch: {manifest_path}")
    actual_files = _inventory(artifact_root)
    if actual_files != expected_files:
        expected_by_path = {
            str(record.get("path")): record for record in expected_files
        }
        actual_by_path = {
            str(record.get("path")): record for record in actual_files
        }
        changed = sorted(
            path
            for path in set(expected_by_path) | set(actual_by_path)
            if expected_by_path.get(path) != actual_by_path.get(path)
        )
        raise ValueError(
            "Artifact integrity validation failed for: "
            + ", ".join(changed[:20])
        )
    return {
        **manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_path": str(manifest_path),
    }
