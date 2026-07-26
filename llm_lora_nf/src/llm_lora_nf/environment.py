import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from typing import Any, Dict, Mapping

import torch


LOCKED_PACKAGE_VERSIONS = {
    "torch": "2.5.1",
    "transformers": "4.57.3",
    "datasets": "3.6.0",
    "peft": "0.17.1",
    "accelerate": "1.10.1",
}


def package_versions() -> Dict[str, str]:
    versions = {}
    for package in (*LOCKED_PACKAGE_VERSIONS, "numpy", "PyYAML"):
        versions[package] = importlib.metadata.version(package)
    return versions


def validate_locked_package_versions(
    actual: Mapping[str, str],
) -> None:
    mismatches = {
        package: {
            "expected": expected,
            "actual": actual.get(package),
        }
        for package, expected in LOCKED_PACKAGE_VERSIONS.items()
        if actual.get(package) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Formal environment package mismatch: "
            + ", ".join(
                f"{package}={values['actual']} (expected {values['expected']})"
                for package, values in sorted(mismatches.items())
            )
        )


def software_environment_identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    required = (
        "python",
        "packages",
        "installed_distributions_sha256",
        "torch_cuda_version",
        "cudnn_version",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(
            "Environment record lacks identity fields: "
            + ", ".join(missing)
        )
    return {
        "python": str(record["python"]),
        "packages": dict(record["packages"]),
        "installed_distributions_sha256": str(
            record["installed_distributions_sha256"]
        ),
        "torch_cuda_version": record["torch_cuda_version"],
        "cudnn_version": record["cudnn_version"],
    }


def execution_environment_identity(
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    if "device" not in record or "cuda_device" not in record:
        raise ValueError("Environment record lacks execution-device identity")
    cuda_device = record["cuda_device"]
    if cuda_device is not None:
        if not isinstance(cuda_device, Mapping):
            raise ValueError("cuda_device must be a mapping or null")
        cuda_device = {
            "name": str(cuda_device["name"]),
            "capability": list(cuda_device["capability"]),
            "total_memory_bytes": int(cuda_device["total_memory_bytes"]),
        }
    return {
        "software": software_environment_identity(record),
        "device_type": str(record["device"]).split(":", 1)[0],
        "cuda_device": cuda_device,
    }


def runtime_environment(
    device: torch.device,
    *,
    validate_formal: bool,
) -> Dict[str, Any]:
    versions = package_versions()
    installed_distributions = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    distributions_sha256 = hashlib.sha256(
        json.dumps(
            installed_distributions,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if validate_formal:
        validate_locked_package_versions(versions)
    cuda_device = None
    if device.type == "cuda":
        cuda_device = {
            "logical_index": int(device.index or 0),
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        }
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "installed_distributions": installed_distributions,
        "installed_distributions_sha256": distributions_sha256,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "cuda_device": cuda_device,
    }
