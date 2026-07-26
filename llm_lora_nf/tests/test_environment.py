from pathlib import Path

import pytest

from llm_lora_nf.environment import (
    LOCKED_CODE_EVALUATION_PACKAGE_VERSIONS,
    LOCKED_PACKAGE_VERSIONS,
    execution_environment_identity,
    software_environment_identity,
    validate_code_evaluation_package_versions,
    validate_locked_package_versions,
)
from llm_lora_nf.provenance import validate_derivation_source


def test_locked_environment_accepts_exact_versions():
    validate_locked_package_versions(dict(LOCKED_PACKAGE_VERSIONS))


def test_locked_environment_rejects_peft_drift():
    changed = dict(LOCKED_PACKAGE_VERSIONS)
    changed["peft"] = "0.18.0"
    with pytest.raises(RuntimeError, match="peft=0.18.0"):
        validate_locked_package_versions(changed)


def test_locked_code_evaluation_versions_reject_drift():
    validate_code_evaluation_package_versions(
        LOCKED_CODE_EVALUATION_PACKAGE_VERSIONS
    )
    changed = dict(LOCKED_CODE_EVALUATION_PACKAGE_VERSIONS)
    changed["evalplus"] = "future"
    with pytest.raises(RuntimeError, match="evalplus=future"):
        validate_code_evaluation_package_versions(changed)


def test_locked_versions_match_server_requirements_file():
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements-server.txt"
    ).read_text(encoding="utf-8")
    pins = {
        name: version
        for line in requirements.splitlines()
        if line and not line.startswith("#") and "==" in line
        for name, version in [line.split("==", 1)]
    }
    for package, version in LOCKED_PACKAGE_VERSIONS.items():
        assert pins[package] == version


def test_code_evaluation_versions_match_requirements_file():
    requirements = (
        Path(__file__).resolve().parents[1]
        / "requirements-code-evaluation.txt"
    ).read_text(encoding="utf-8")
    pins = {
        name.replace("_", "-"): version
        for line in requirements.splitlines()
        if line and not line.startswith("#") and "==" in line
        for name, version in [line.split("==", 1)]
    }
    for package, version in (
        LOCKED_CODE_EVALUATION_PACKAGE_VERSIONS.items()
    ):
        if package == "evalplus":
            continue
        assert pins[package] == version


def test_software_environment_identity_excludes_machine_local_paths():
    record = {
        "python": "3.10.20",
        "python_executable": "/private/venv/bin/python",
        "packages": dict(LOCKED_PACKAGE_VERSIONS),
        "installed_distributions_sha256": "distribution-tree",
        "torch_cuda_version": "12.4",
        "cudnn_version": 90100,
        "cuda_visible_devices": "2",
        "device": "cuda:0",
        "cuda_device": {
            "logical_index": 0,
            "name": "Test GPU",
            "capability": [9, 0],
            "total_memory_bytes": 100,
        },
    }
    identity = software_environment_identity(record)
    assert "python_executable" not in identity
    assert "cuda_visible_devices" not in identity
    assert identity["installed_distributions_sha256"] == "distribution-tree"
    execution = execution_environment_identity(record)
    assert "logical_index" not in execution["cuda_device"]
    assert execution["cuda_device"]["name"] == "Test GPU"


def test_formal_derivation_requires_exact_clean_source():
    validate_derivation_source(
        {"commit_sha": "exact", "source_dirty": False},
        expected_commit="exact",
        formal=True,
    )
    with pytest.raises(ValueError, match="exact clean"):
        validate_derivation_source(
            {"commit_sha": "exact", "source_dirty": True},
            expected_commit="exact",
            formal=True,
        )
    with pytest.raises(ValueError, match="exact clean"):
        validate_derivation_source(
            {"commit_sha": "other", "source_dirty": False},
            expected_commit="exact",
            formal=True,
        )
