import json

import pytest
import torch

from llm_lora_nf.calibration import SecondMomentState
from llm_lora_nf.calibration_cache import (
    build_calibration_cache_identity,
    calibration_cache_key,
    load_artifact_cache,
    load_calibration_cache,
    save_artifact_cache,
    save_calibration_cache,
)
from scripts.run_sft import _remove_corda_ephemeral_artifacts


def _identity(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "local_snapshot_manifest.json").write_text(
        json.dumps({"model_id": "test/model", "revision": "abc"}),
        encoding="utf-8",
    )
    nq = tmp_path / "nq.parquet"
    nq.write_bytes(b"fixture")
    return build_calibration_cache_identity(
        method="lora_nf",
        protocol="unit_test",
        code_commit="deadbeef",
        model_path=str(model),
        requested_model_id="test/model",
        source_model_id="test/model",
        model_torch_dtype="torch.float32",
        nq_parquet=str(nq),
        nq_revision="nq-revision",
        samples=2,
        seed=233,
        sequence_length=64,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        require_model_manifest=True,
    )


def test_calibration_cache_roundtrip_is_content_addressed(tmp_path):
    identity = _identity(tmp_path)
    moments = {
        "layer.qkv_shared": SecondMomentState(
            matrix=torch.eye(4),
            observations=8,
        )
    }
    missing, miss_record = load_calibration_cache(
        str(tmp_path / "cache"),
        identity,
    )
    assert missing is None
    assert miss_record["status"] == "miss"
    written = save_calibration_cache(
        str(tmp_path / "cache"),
        identity,
        moments,
    )
    loaded, hit_record = load_calibration_cache(
        str(tmp_path / "cache"),
        identity,
    )
    assert written["key"] == calibration_cache_key(identity)
    assert hit_record["status"] == "hit"
    torch.testing.assert_close(
        loaded["layer.qkv_shared"].matrix,
        moments["layer.qkv_shared"].matrix,
    )
    assert loaded["layer.qkv_shared"].observations == 8


def test_cache_key_changes_with_protocol_identity(tmp_path):
    identity = _identity(tmp_path)
    changed = dict(identity)
    changed["protocol"] = "different"
    assert calibration_cache_key(identity) != calibration_cache_key(changed)


def test_generic_artifact_cache_roundtrip(tmp_path):
    identity = _identity(tmp_path)
    identity["method"] = "corda"
    source = tmp_path / "eigens.pt"
    source.write_bytes(b"trusted-local-eigens")
    written = save_artifact_cache(
        str(tmp_path / "artifact-cache"),
        identity,
        source=str(source),
        artifact_name="corda_eigens.pt",
    )
    artifact, loaded = load_artifact_cache(
        str(tmp_path / "artifact-cache"),
        identity,
        artifact_name="corda_eigens.pt",
    )
    assert written["status"] == "written"
    assert loaded["status"] == "hit"
    assert artifact.read_bytes() == source.read_bytes()


def test_corda_cleanup_removes_only_declared_ephemeral_files(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "corda_svd_cache.pt").write_bytes(b"eigens")
    (output / "corda_covariance.pt").write_bytes(b"covariance")
    (output / "run_report.json").write_text("keep", encoding="utf-8")

    removed = _remove_corda_ephemeral_artifacts(
        output,
        require_eigens=True,
    )
    assert removed == ["corda_covariance.pt", "corda_svd_cache.pt"]
    assert (output / "run_report.json").read_text(encoding="utf-8") == "keep"

    with pytest.raises(FileNotFoundError, match="eigens artifact"):
        _remove_corda_ephemeral_artifacts(
            output,
            require_eigens=True,
        )
