import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .calibration import SecondMomentState
from .integrity import INTEGRITY_FILENAME


CALIBRATION_CACHE_FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_snapshot_identity(
    model_path: Path,
    *,
    require_manifest: bool,
) -> Dict[str, Any]:
    manifest = model_path / "local_snapshot_manifest.json"
    integrity_manifest = model_path / INTEGRITY_FILENAME
    if integrity_manifest.exists():
        return {
            "kind": "model_snapshot_integrity",
            "sha256": _sha256(integrity_manifest),
        }
    if manifest.exists():
        return {
            "kind": "local_snapshot_manifest",
            "sha256": _sha256(manifest),
        }
    if require_manifest:
        raise FileNotFoundError(
            "Formal calibration caching requires local_snapshot_manifest.json "
            f"in {model_path}"
        )
    descriptors = []
    for candidate in sorted(model_path.iterdir()):
        if candidate.is_file() and (
            candidate.name in {
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            }
            or candidate.suffix == ".safetensors"
        ):
            descriptors.append(
                {
                    "name": candidate.name,
                    "size": candidate.stat().st_size,
                }
            )
    if not descriptors:
        raise FileNotFoundError(
            f"No snapshot identity files found in {model_path}"
        )
    return {
        "kind": "nonformal_file_descriptor_fallback",
        "sha256": _canonical_hash({"files": descriptors}),
    }


def build_calibration_cache_identity(
    *,
    method: str,
    protocol: str,
    code_commit: str,
    model_path: str,
    requested_model_id: str,
    source_model_id: str,
    model_torch_dtype: str,
    nq_parquet: str,
    nq_revision: str,
    samples: int,
    seed: int,
    sequence_length: int,
    target_modules: Tuple[str, ...],
    require_model_manifest: bool,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if method not in {"lora_nf", "lora_null", "corda"}:
        raise ValueError("Unsupported calibration cache method")
    if samples <= 0 or sequence_length <= 0:
        raise ValueError("Calibration cache dimensions must be positive")
    model_root = Path(model_path).resolve()
    nq_path = Path(nq_parquet).resolve()
    if not nq_path.is_file():
        raise FileNotFoundError(f"Missing NQ calibration file: {nq_path}")
    return {
        "format_version": CALIBRATION_CACHE_FORMAT_VERSION,
        "method": method,
        "protocol": protocol,
        "code_commit": code_commit,
        "model": {
            "requested_id": requested_model_id,
            "source_id": source_model_id,
            "torch_dtype": model_torch_dtype,
            "snapshot": _model_snapshot_identity(
                model_root,
                require_manifest=require_model_manifest,
            ),
        },
        "dataset": {
            "id": "google-research-datasets/nq_open",
            "revision": nq_revision,
            "parquet_sha256": _sha256(nq_path),
        },
        "calibration": {
            "samples": int(samples),
            "seed": int(seed),
            "sequence_length": int(sequence_length),
            "target_modules": list(target_modules),
            "extra": dict(extra or {}),
        },
    }


def calibration_cache_key(identity: Mapping[str, Any]) -> str:
    return _canonical_hash(identity)


def _validate_moments(
    moments: Mapping[str, SecondMomentState],
) -> None:
    if not moments:
        raise ValueError("Calibration cache contains no moments")
    for group, state in moments.items():
        matrix = state.matrix
        if not group or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Invalid cached moment shape for {group!r}")
        if state.observations <= 0 or state.phases <= 0:
            raise ValueError(f"Invalid cached counts for {group}")
        if not torch.isfinite(matrix).all():
            raise ValueError(f"Cached moment contains non-finite values: {group}")
        if not torch.allclose(
            matrix,
            matrix.transpose(0, 1),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError(f"Cached moment is not symmetric: {group}")


def load_calibration_cache(
    cache_root: str,
    identity: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, SecondMomentState]], Dict[str, Any]]:
    root = Path(cache_root).resolve()
    key = calibration_cache_key(identity)
    entry = root / key
    record: Dict[str, Any] = {
        "enabled": True,
        "key": key,
        "entry": str(entry),
        "status": "miss",
    }
    if not entry.exists():
        return None, record
    manifest_path = entry / "manifest.json"
    weights_path = entry / "moments.pt"
    if not manifest_path.is_file() or not weights_path.is_file():
        raise ValueError(f"Incomplete calibration cache entry: {entry}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format_version") != CALIBRATION_CACHE_FORMAT_VERSION
        or manifest.get("kind") != "moments"
    ):
        raise ValueError(f"Unsupported calibration cache version: {entry}")
    if manifest.get("key") != key or manifest.get("identity") != dict(identity):
        raise ValueError(f"Calibration cache identity mismatch: {entry}")
    actual_sha = _sha256(weights_path)
    if manifest.get("moments_sha256") != actual_sha:
        raise ValueError(f"Calibration cache checksum mismatch: {entry}")
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if (
        payload.get("format_version") != CALIBRATION_CACHE_FORMAT_VERSION
        or payload.get("key") != key
    ):
        raise ValueError(f"Calibration cache payload mismatch: {entry}")
    moments = {
        group: SecondMomentState(
            matrix=state["matrix"],
            observations=int(state["observations"]),
            phases=int(state["phases"]),
        )
        for group, state in payload["moments"].items()
    }
    _validate_moments(moments)
    record.update(
        {
            "status": "hit",
            "moments_sha256": actual_sha,
            "groups": len(moments),
        }
    )
    return moments, record


def save_calibration_cache(
    cache_root: str,
    identity: Mapping[str, Any],
    moments: Mapping[str, SecondMomentState],
) -> Dict[str, Any]:
    _validate_moments(moments)
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = calibration_cache_key(identity)
    entry = root / key
    if entry.exists():
        existing, record = load_calibration_cache(str(root), identity)
        if existing is None:
            raise RuntimeError(f"Cache entry disappeared during save: {entry}")
        record["status"] = "race_hit"
        return record
    payload = {
        "format_version": CALIBRATION_CACHE_FORMAT_VERSION,
        "key": key,
        "moments": {
            group: {
                "matrix": state.matrix.detach().cpu(),
                "observations": int(state.observations),
                "phases": int(state.phases),
            }
            for group, state in moments.items()
        },
    }
    with tempfile.TemporaryDirectory(
        prefix=f".{key}.tmp-",
        dir=str(root),
    ) as temporary_name:
        temporary = Path(temporary_name)
        weights_path = temporary / "moments.pt"
        torch.save(payload, weights_path)
        moments_sha256 = _sha256(weights_path)
        manifest = {
            "format_version": CALIBRATION_CACHE_FORMAT_VERSION,
            "kind": "moments",
            "key": key,
            "identity": dict(identity),
            "moments_sha256": moments_sha256,
            "groups": {
                group: {
                    "shape": list(state.matrix.shape),
                    "dtype": str(state.matrix.dtype),
                    "observations": int(state.observations),
                    "phases": int(state.phases),
                }
                for group, state in moments.items()
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            temporary.rename(entry)
        except OSError:
            if not entry.exists():
                raise
            existing, record = load_calibration_cache(str(root), identity)
            if existing is None:
                raise RuntimeError(
                    f"Calibration cache race produced no readable entry: {entry}"
                )
            record["status"] = "race_hit"
            return record
    return {
        "enabled": True,
        "key": key,
        "entry": str(entry),
        "status": "written",
        "moments_sha256": moments_sha256,
        "groups": len(moments),
    }


def load_artifact_cache(
    cache_root: str,
    identity: Mapping[str, Any],
    *,
    artifact_name: str,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    root = Path(cache_root).resolve()
    key = calibration_cache_key(identity)
    entry = root / key
    record: Dict[str, Any] = {
        "enabled": True,
        "key": key,
        "entry": str(entry),
        "status": "miss",
        "artifact_name": artifact_name,
    }
    if not entry.exists():
        return None, record
    manifest_path = entry / "manifest.json"
    artifact_path = entry / artifact_name
    if not manifest_path.is_file() or not artifact_path.is_file():
        raise ValueError(f"Incomplete artifact cache entry: {entry}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format_version") != CALIBRATION_CACHE_FORMAT_VERSION
        or manifest.get("kind") != "artifact"
        or manifest.get("key") != key
        or manifest.get("identity") != dict(identity)
        or manifest.get("artifact_name") != artifact_name
    ):
        raise ValueError(f"Artifact cache identity mismatch: {entry}")
    actual_sha = _sha256(artifact_path)
    if manifest.get("artifact_sha256") != actual_sha:
        raise ValueError(f"Artifact cache checksum mismatch: {entry}")
    record.update(
        {
            "status": "hit",
            "artifact_sha256": actual_sha,
        }
    )
    return artifact_path, record


def save_artifact_cache(
    cache_root: str,
    identity: Mapping[str, Any],
    *,
    source: str,
    artifact_name: str,
) -> Dict[str, Any]:
    source_path = Path(source).resolve()
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing cache artifact: {source_path}")
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    key = calibration_cache_key(identity)
    entry = root / key
    if entry.exists():
        existing, record = load_artifact_cache(
            str(root),
            identity,
            artifact_name=artifact_name,
        )
        if existing is None:
            raise RuntimeError(f"Cache entry disappeared during save: {entry}")
        record["status"] = "race_hit"
        return record
    with tempfile.TemporaryDirectory(
        prefix=f".{key}.tmp-",
        dir=str(root),
    ) as temporary_name:
        temporary = Path(temporary_name)
        destination = temporary / artifact_name
        shutil.copy2(source_path, destination)
        artifact_sha256 = _sha256(destination)
        manifest = {
            "format_version": CALIBRATION_CACHE_FORMAT_VERSION,
            "kind": "artifact",
            "key": key,
            "identity": dict(identity),
            "artifact_name": artifact_name,
            "artifact_sha256": artifact_sha256,
            "size_bytes": destination.stat().st_size,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            temporary.rename(entry)
        except OSError:
            if not entry.exists():
                raise
            existing, record = load_artifact_cache(
                str(root),
                identity,
                artifact_name=artifact_name,
            )
            if existing is None:
                raise RuntimeError(
                    f"Artifact cache race produced no readable entry: {entry}"
                )
            record["status"] = "race_hit"
            return record
    return {
        "enabled": True,
        "key": key,
        "entry": str(entry),
        "status": "written",
        "artifact_name": artifact_name,
        "artifact_sha256": artifact_sha256,
    }
