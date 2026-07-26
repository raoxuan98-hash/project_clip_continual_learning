import hashlib
import json

import pytest

from llm_lora_nf.artifact_retention import LOCKED_ARTIFACT_RETENTION
from llm_lora_nf.integrity import write_directory_integrity
from scripts.aggregate_lm_eval import _model_identity, _validated_run
from scripts.aggregate_seed_results import _validate_summary_derivations
from scripts.launch_math_evaluation import (
    _assert_single_merged_slot,
    _completed_evaluation,
    _delete_ephemeral_merged,
    _validate_summary_identity,
    _validate_training_run,
)
from scripts.launch_math_matrix import (
    _completed_formal_run as _completed_training_matrix_run,
)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_validated_run_requires_untampered_evaluation_output(tmp_path):
    task_definitions = tmp_path / "task_definitions"
    task_definitions.mkdir()
    (task_definitions / "task.yaml").write_text(
        "task: pinned\n",
        encoding="utf-8",
    )
    _write_json(tmp_path / "result.json", {"results": {"task": {"acc": 1.0}}})
    evaluator_seeds = {
        "random": 0,
        "numpy": 1234,
        "torch": 1234,
        "fewshot": 1234,
    }
    environment = {
        "python": "3.10.20",
        "packages": {
            "torch": "2.5.1",
            "transformers": "4.57.3",
        },
        "installed_distributions_sha256": "environment-sha",
        "torch_cuda_version": "12.4",
        "cudnn_version": 90100,
    }
    software_environment = {
        key: environment[key]
        for key in (
            "python",
            "packages",
            "installed_distributions_sha256",
            "torch_cuda_version",
            "cudnn_version",
        )
    }
    request_cache_identity = {
        "format_version": 2,
        "execution_mode": "gpu_formal",
        "tasks": ["task"],
        "limit": None,
        "evaluator_revision": "evaluator-sha",
        "protocol_config_hash": "protocol-sha",
        "dataset_manifest_sha256": "dataset-sha",
        "task_definition_tree_sha256": "task-tree-sha",
        "base_snapshot_integrity_sha256": "base-sha",
        "tokenizer_path": "/models/base",
        "evaluator_seeds": evaluator_seeds,
        "software_environment": software_environment,
        "cache_requests_enabled": True,
    }
    request_cache_key = hashlib.sha256(
        json.dumps(
            request_cache_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json(
        tmp_path / "evaluation_run.json",
        {
            "format_version": 2,
            "status": "passed",
            "returncode": 0,
            "formal_result_eligible": True,
            "execution_mode": "gpu_formal",
            "evaluator_seeds": evaluator_seeds,
            "environment": environment,
            "limit": None,
            "source": {
                "commit_sha": "source-sha",
                "source_dirty": False,
            },
            "model_role": "merged",
            "model_integrity": {
                "kind": "merged_checkpoint",
                "manifest_sha256": "merged-sha",
            },
            "dataset_evidence": {
                "manifest_sha256": "dataset-sha",
                "verified_cache_files": 5,
            },
            "task_definitions": {
                "root": str(task_definitions.resolve()),
                "tree_sha256": "task-tree-sha",
            },
            "request_cache": {
                "identity": request_cache_identity,
                "key": request_cache_key,
                "path": str(tmp_path / "request_cache" / request_cache_key),
                "integrity": {
                    "tree_sha256": "request-cache-tree-sha",
                },
                "retention": (
                    "shared_content_addressed_per_base_model_protocol"
                ),
            },
        },
    )
    write_directory_integrity(str(tmp_path), kind="evaluation_output")
    manifest, integrity = _validated_run(
        tmp_path,
        allow_cpu_smoke=False,
    )
    assert manifest["status"] == "passed"
    assert integrity["kind"] == "evaluation_output"

    _write_json(tmp_path / "result.json", {"results": {"task": {"acc": 0.0}}})
    with pytest.raises(ValueError, match="result.json"):
        _validated_run(tmp_path, allow_cpu_smoke=False)


def test_merged_model_identity_binds_export_to_base_snapshot():
    run_manifest = {
        "model_role": "merged",
        "model_integrity": {"manifest_sha256": "merged-sha"},
    }
    model_manifest = {
        "base_model": {
            "requested_id": "publisher/instruct",
            "source_id": "mirror/instruct",
            "snapshot_integrity_sha256": "snapshot-sha",
            "snapshot_manifest_sha256": "provenance-sha",
        },
        "base_model_integrity_manifest_sha256": "snapshot-sha",
    }
    identity = _model_identity(run_manifest, model_manifest)
    assert identity["base_snapshot_integrity_sha256"] == "snapshot-sha"

    model_manifest["base_model_integrity_manifest_sha256"] = "other"
    with pytest.raises(ValueError, match="validated base snapshot"):
        _model_identity(run_manifest, model_manifest)


def test_formal_merged_identity_requires_fp32_ephemeral_export():
    run_manifest = {
        "formal_result_eligible": True,
        "source": {
            "commit_sha": "source-sha",
            "source_dirty": False,
        },
        "model_role": "merged",
        "model_integrity": {"manifest_sha256": "merged-sha"},
    }
    model_manifest = {
        "formal_result_eligible": True,
        "execution_mode": "gpu_formal",
        "training_commit_sha": "source-sha",
        "export_commit_sha": "source-sha",
        "export_source_dirty": False,
        "dtype": "torch.bfloat16",
        "checkpoint_retention": (
            "ephemeral_delete_after_qualified_evaluation"
        ),
        "base_model": {
            "requested_id": "publisher/instruct",
            "source_id": "mirror/instruct",
            "snapshot_integrity_sha256": "snapshot-sha",
            "snapshot_manifest_sha256": "provenance-sha",
        },
        "base_model_integrity_manifest_sha256": "snapshot-sha",
    }
    with pytest.raises(ValueError, match="FP32"):
        _model_identity(run_manifest, model_manifest)
    model_manifest["dtype"] = "torch.float32"
    identity = _model_identity(run_manifest, model_manifest)
    assert identity["evaluated_checkpoint_integrity_sha256"] == "merged-sha"


def test_summary_aggregate_values_are_recomputed_from_task_membership():
    summary = {
        "scores": {
            "math": {"score": 0.5},
            "knowledge": {"score": 0.8},
        },
        "task_groups": {
            "downstream": ["math"],
            "retention": ["knowledge"],
        },
        "downstream_average": 0.5,
        "retention_average": 0.8,
        "skill_retention_geometric_mean": (0.5 * 0.8) ** 0.5,
    }
    _validate_summary_derivations(summary)
    summary["retention_average"] = 0.7
    with pytest.raises(ValueError, match="retention_average"):
        _validate_summary_derivations(summary)


def test_only_qualified_merged_checkpoint_is_deleted(tmp_path):
    merged_root = tmp_path / "merged"
    checkpoint = merged_root / "model" / "method" / "seed_42"
    checkpoint.mkdir(parents=True)
    _write_json(
        checkpoint / "merged_export_manifest.json",
        {"formal_result_eligible": True},
    )
    (checkpoint / "model.safetensors").write_bytes(b"merged")
    write_directory_integrity(str(checkpoint), kind="merged_checkpoint")

    _delete_ephemeral_merged(checkpoint, merged_root=merged_root)
    assert not checkpoint.exists()

    with pytest.raises(ValueError, match="root"):
        _delete_ephemeral_merged(merged_root, merged_root=merged_root)


def test_completed_evaluation_requires_exact_formal_state(tmp_path):
    _write_json(tmp_path / "result.json", {"results": {}})
    _write_json(
        tmp_path / "evaluation_run.json",
        {
            "format_version": 2,
            "formal_result_eligible": True,
            "status": "passed",
            "returncode": 1,
            "execution_mode": "gpu_formal",
            "limit": None,
            "model_role": "base",
            "source": {"source_dirty": False},
        },
    )
    write_directory_integrity(str(tmp_path), kind="evaluation_output")
    assert not _completed_evaluation(tmp_path, expected_role="base")


def test_training_resume_validation_binds_method_model_seed_and_checkpoint(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
    checkpoint_integrity = write_directory_integrity(
        str(checkpoint),
        kind="adapter_checkpoint",
    )
    report = {
        "formal_result_eligible": True,
        "status": "trained",
        "source_dirty": False,
        "artifact_retention": LOCKED_ARTIFACT_RETENTION,
        "metadata": {
            "execution_mode": "gpu_formal",
            "method": "lora_nf",
            "model_id": "publisher/instruct",
            "seed": 42,
            "commit_sha": "training-sha",
        },
        "checkpoint_integrity": checkpoint_integrity,
    }
    _write_json(tmp_path / "run_report.json", report)
    validated = _validate_training_run(
        tmp_path,
        expected_method="lora_nf",
        expected_model_id="publisher/instruct",
        expected_seed=42,
        expected_commit="training-sha",
    )
    assert validated["metadata"]["seed"] == 42
    assert _completed_training_matrix_run(
        tmp_path,
        expected_method="lora_nf",
        expected_model_id="publisher/instruct",
        expected_seed=42,
        expected_commit="training-sha",
    )
    assert not _completed_training_matrix_run(
        tmp_path,
        expected_method="lora_nf",
        expected_model_id="publisher/instruct",
        expected_seed=42,
        expected_commit="new-source-sha",
    )

    report["metadata"]["seed"] = 43
    _write_json(tmp_path / "run_report.json", report)
    with pytest.raises(ValueError, match="mismatched training run"):
        _validate_training_run(
            tmp_path,
            expected_method="lora_nf",
            expected_model_id="publisher/instruct",
            expected_seed=42,
            expected_commit="training-sha",
        )


def test_merged_slot_rejects_a_second_checkpoint(tmp_path):
    merged_root = tmp_path / "merged"
    current = merged_root / "model_a" / "lora_nf" / "seed_42"
    current.mkdir(parents=True)
    (current / "model.safetensors").write_bytes(b"current")
    _assert_single_merged_slot(merged_root, allowed_path=current)

    foreign = merged_root / "model_b" / "lora" / "seed_42"
    foreign.mkdir(parents=True)
    (foreign / "model.safetensors").write_bytes(b"foreign")
    with pytest.raises(RuntimeError, match="one-checkpoint merged slot"):
        _assert_single_merged_slot(merged_root, allowed_path=current)


def test_merged_deletion_gate_requires_summary_to_bind_exact_evidence(
    tmp_path,
):
    evaluation_dir = tmp_path / "adapted"
    base_run = tmp_path / "base"
    evaluation_dir.mkdir()
    base_run.mkdir()
    evaluator_seeds = {
        "random": 0,
        "numpy": 1234,
        "torch": 1234,
        "fewshot": 1234,
    }
    dataset_evidence = {"manifest_sha256": "dataset-sha"}
    source = {"commit_sha": "source-sha", "source_dirty": False}
    software_environment = {
        "python": "3.10.20",
        "packages": {"torch": "2.5.1"},
        "installed_distributions_sha256": "environment-sha",
        "torch_cuda_version": "12.4",
        "cudnn_version": 90100,
    }
    _write_json(
        evaluation_dir / "evaluation_run.json",
        {
            "model_integrity": {"manifest_sha256": "merged-sha"},
            "dataset_evidence": dataset_evidence,
            "task_definitions": {"tree_sha256": "tasks-sha"},
            "tasks": ["task"],
            "evaluator_revision": "evaluator-sha",
            "evaluator_seeds": evaluator_seeds,
            "protocol_config_hash": "protocol-sha",
            "source": source,
            "request_cache": {
                "key": "cache-key",
                "integrity": {"tree_sha256": "cache-tree"},
                "identity": {
                    "software_environment": software_environment,
                },
            },
        },
    )
    _write_json(
        base_run / "evaluation_run.json",
        {
            "model_integrity": {"manifest_sha256": "base-sha"},
            "dataset_evidence": dataset_evidence,
            "task_definitions": {"tree_sha256": "tasks-sha"},
            "tasks": ["task"],
            "evaluator_revision": "evaluator-sha",
            "evaluator_seeds": evaluator_seeds,
            "protocol_config_hash": "protocol-sha",
            "source": source,
            "request_cache": {
                "key": "cache-key",
                "integrity": {"tree_sha256": "cache-tree"},
            },
        },
    )
    evaluation_integrity = write_directory_integrity(
        str(evaluation_dir),
        kind="evaluation_output",
    )
    base_integrity = write_directory_integrity(
        str(base_run),
        kind="evaluation_output",
    )
    evaluation_manifest = evaluation_dir / "evaluation_run.json"
    base_manifest = base_run / "evaluation_run.json"
    summary_path = tmp_path / "summary.json"
    summary = {
        "format_version": 2,
        "formal_result_eligible": True,
        "run_manifest": str(evaluation_manifest),
        "run_manifest_sha256": hashlib.sha256(
            evaluation_manifest.read_bytes()
        ).hexdigest(),
        "evaluation_output_integrity": evaluation_integrity,
        "model_identity": {
            "base_snapshot_integrity_sha256": "base-sha",
            "evaluated_checkpoint_integrity_sha256": "merged-sha",
        },
        "dataset_evidence": dataset_evidence,
        "task_definition_tree_sha256": "tasks-sha",
        "evaluator_seeds": evaluator_seeds,
        "request_cache_key": "cache-key",
        "request_cache_tree_sha256": "cache-tree",
        "aggregation_source": source,
        "aggregation_software_environment": software_environment,
        "base_comparison": {
            "base_run_manifest": str(base_manifest),
            "base_run_manifest_sha256": hashlib.sha256(
                base_manifest.read_bytes()
            ).hexdigest(),
            "base_evaluation_output_integrity": base_integrity,
            "base_model_identity": {
                "evaluated_checkpoint_integrity_sha256": "base-sha",
            },
        },
    }
    _write_json(summary_path, summary)
    validated = _validate_summary_identity(
        summary_path,
        evaluation_dir=evaluation_dir,
        base_run=base_run,
    )
    assert validated["formal_result_eligible"]

    summary["request_cache_tree_sha256"] = "other-tree"
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="exact qualified"):
        _validate_summary_identity(
            summary_path,
            evaluation_dir=evaluation_dir,
            base_run=base_run,
        )
