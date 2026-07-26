import pytest

from llm_lora_nf.integrity import (
    validate_directory_integrity,
    write_directory_integrity,
)


def test_directory_integrity_roundtrip_and_tamper_detection(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"tiny"}\n')
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    written = write_directory_integrity(
        str(tmp_path),
        kind="model_snapshot",
    )
    validated = validate_directory_integrity(
        str(tmp_path),
        expected_kind="model_snapshot",
    )
    assert validated["tree_sha256"] == written["tree_sha256"]
    assert validated["manifest_sha256"] == written["manifest_sha256"]

    (tmp_path / "weights.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="weights.safetensors"):
        validate_directory_integrity(
            str(tmp_path),
            expected_kind="model_snapshot",
        )


def test_directory_integrity_rejects_unregistered_file(tmp_path):
    (tmp_path / "adapter.pt").write_bytes(b"adapter")
    write_directory_integrity(str(tmp_path), kind="adapter_checkpoint")
    (tmp_path / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="extra.bin"):
        validate_directory_integrity(
            str(tmp_path),
            expected_kind="adapter_checkpoint",
        )


def test_nested_integrity_named_file_is_not_silently_excluded(tmp_path):
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "artifact_integrity.json").write_text(
        '{"payload":"not the root manifest"}\n',
        encoding="utf-8",
    )
    written = write_directory_integrity(str(tmp_path), kind="model_snapshot")
    paths = {record["path"] for record in written["files"]}
    assert "nested/artifact_integrity.json" in paths


def test_root_atomic_temp_file_is_not_hashed_into_its_own_manifest(tmp_path):
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    (tmp_path / ".artifact_integrity.json.tmp").write_text(
        "stale interrupted write\n",
        encoding="utf-8",
    )
    write_directory_integrity(str(tmp_path), kind="model_snapshot")
    validate_directory_integrity(
        str(tmp_path),
        expected_kind="model_snapshot",
    )
    paths = {
        record["path"]
        for record in write_directory_integrity(
            str(tmp_path),
            kind="model_snapshot",
        )["files"]
    }
    assert ".artifact_integrity.json.tmp" not in paths

    (tmp_path / ".artifact_integrity.json.tmp").write_text(
        "new interrupted write\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Interrupted"):
        validate_directory_integrity(
            str(tmp_path),
            expected_kind="model_snapshot",
        )
