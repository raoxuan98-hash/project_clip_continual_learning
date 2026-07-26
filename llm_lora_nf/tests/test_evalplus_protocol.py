import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from llm_lora_nf.config_io import load_yaml_config
from llm_lora_nf.evalplus_evidence import (
    validate_evalplus_data_manifest,
    validate_evalplus_dataset_file,
)
from llm_lora_nf.evalplus_results import (
    validate_evalplus_result,
    validate_evalplus_samples,
)
from llm_lora_nf.protocol_validation import (
    validate_formal_code_evaluation_config,
)
from llm_lora_nf.integrity import write_directory_integrity
from scripts.launch_code_evaluation import (
    _completed_codegen,
    _completed_execution,
)
from scripts.aggregate_evalplus_seed_results import (
    METRICS,
    _aggregate_group,
    _validate_execution_run,
    _validate_paired_compatibility,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "evaluation"
    / "track_b_code_evalplus.yaml"
)


def _jsonl_bytes(rows):
    return b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _row(task_id):
    return {
        "task_id": task_id,
        "prompt": "def answer():\n",
        "contract": "",
        "canonical_solution": "    return 42\n",
        "entry_point": "answer",
        "base_input": [[]],
        "plus_input": [[], []],
        "atol": 0,
    }


def test_locked_evalplus_config_passes_and_rejects_generation_drift():
    config = load_yaml_config(str(CONFIG_PATH))
    validate_formal_code_evaluation_config(config)
    changed = copy.deepcopy(config)
    changed["generation"]["num_samples"] = 10
    with pytest.raises(ValueError, match="generation.num_samples"):
        validate_formal_code_evaluation_config(changed)


def test_evalplus_manifest_and_schema_are_file_identity_bound(
    tmp_path,
    monkeypatch,
):
    import llm_lora_nf.evalplus_evidence as evidence

    rows = [_row("HumanEval/0")]
    jsonl_bytes = _jsonl_bytes(rows)
    compressed_bytes = gzip.compress(jsonl_bytes, mtime=0)
    jsonl = tmp_path / "HumanEvalPlus.jsonl"
    compressed = tmp_path / "HumanEvalPlus.jsonl.gz"
    jsonl.write_bytes(jsonl_bytes)
    compressed.write_bytes(compressed_bytes)
    specification = {
        "humaneval": {
            "release": "HumanEvalPlus",
            "version": "fixture",
            "url": "https://example.invalid/fixture",
            "compressed_file": compressed.name,
            "compressed_size_bytes": len(compressed_bytes),
            "compressed_sha256": hashlib.sha256(compressed_bytes).hexdigest(),
            "jsonl_file": jsonl.name,
            "jsonl_size_bytes": len(jsonl_bytes),
            "jsonl_sha256": hashlib.sha256(jsonl_bytes).hexdigest(),
            "jsonl_md5": hashlib.md5(jsonl_bytes).hexdigest(),
            "rows": 1,
            "task_prefix": "HumanEval/",
        }
    }
    monkeypatch.setattr(evidence, "EVALPLUS_DATASETS", specification)
    compressed_record = validate_evalplus_dataset_file(
        compressed,
        dataset="humaneval",
        compressed=True,
    )
    jsonl_record = validate_evalplus_dataset_file(
        jsonl,
        dataset="humaneval",
        compressed=False,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "source": "official_github_release",
                "datasets": {
                    "humaneval": {
                        "release": "HumanEvalPlus",
                        "version": "fixture",
                        "url": "https://example.invalid/fixture",
                        "compressed": compressed_record,
                        "jsonl": jsonl_record,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    validated = validate_evalplus_data_manifest(
        str(manifest),
        config={"datasets": specification},
    )
    assert validated["datasets"]["humaneval"]["jsonl"]["rows"] == 1

    jsonl.write_bytes(jsonl_bytes + b"\n")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_evalplus_data_manifest(
            str(manifest),
            config={"datasets": specification},
        )


def test_evalplus_rejects_auto_converted_schema(tmp_path, monkeypatch):
    import llm_lora_nf.evalplus_evidence as evidence

    rows = [{"task_id": "HumanEval/0", "prompt": "p", "test": "combined"}]
    encoded = _jsonl_bytes(rows)
    path = tmp_path / "converted.jsonl"
    path.write_bytes(encoded)
    specification = {
        "humaneval": {
            "jsonl_size_bytes": len(encoded),
            "jsonl_sha256": hashlib.sha256(encoded).hexdigest(),
            "jsonl_md5": hashlib.md5(encoded).hexdigest(),
            "rows": 1,
            "task_prefix": "HumanEval/",
        }
    }
    monkeypatch.setattr(evidence, "EVALPLUS_DATASETS", specification)
    with pytest.raises(ValueError, match="lacks fields"):
        validate_evalplus_dataset_file(
            path,
            dataset="humaneval",
            compressed=False,
        )


def test_evalplus_sample_and_result_metrics_are_recomputed(tmp_path):
    dataset = tmp_path / "HumanEvalPlus.jsonl"
    dataset.write_bytes(
        _jsonl_bytes([_row("HumanEval/0"), _row("HumanEval/1")])
    )
    samples = tmp_path / "samples.jsonl"
    samples.write_bytes(
        _jsonl_bytes(
            [
                {"task_id": "HumanEval/0", "solution": "def answer(): return 42"},
                {"task_id": "HumanEval/1", "solution": "def answer(): return 0"},
            ]
        )
    )
    sample_record = validate_evalplus_samples(
        samples,
        dataset_path=dataset,
    )
    assert sample_record["rows"] == 2
    assert sample_record["empty_solution_count"] == 0

    samples.write_bytes(
        _jsonl_bytes(
            [
                {"task_id": "HumanEval/0", "solution": ""},
                {"task_id": "HumanEval/1", "solution": "def answer(): return 0"},
            ]
        )
    )
    assert validate_evalplus_samples(
        samples,
        dataset_path=dataset,
    )["empty_solution_count"] == 1

    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "hash": "fixture-md5",
                "eval": {
                    "HumanEval/0": [
                        {
                            "task_id": "HumanEval/0",
                            "base_status": "pass",
                            "plus_status": "pass",
                        }
                    ],
                    "HumanEval/1": [
                        {
                            "task_id": "HumanEval/1",
                            "base_status": "pass",
                            "plus_status": "fail",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    import llm_lora_nf.evalplus_results as result_module

    original = result_module.EVALPLUS_DATASETS
    result_module.EVALPLUS_DATASETS = {
        "humaneval": {"jsonl_md5": "fixture-md5"}
    }
    try:
        metrics = validate_evalplus_result(
            result,
            dataset="humaneval",
            dataset_path=dataset,
        )
    finally:
        result_module.EVALPLUS_DATASETS = original
    assert metrics["base_pass_at_1"] == 1.0
    assert metrics["plus_pass_at_1"] == 0.5


def test_code_evaluation_launcher_resume_requires_sealed_formal_outputs(
    tmp_path,
):
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / "code_generation_run.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "status": "generated",
                "formal_result_eligible": True,
                "execution_mode": "gpu_formal",
                "model_role": "merged",
                "source": {
                    "commit_sha": "source",
                    "source_dirty": False,
                },
            }
        ),
        encoding="utf-8",
    )
    write_directory_integrity(
        str(generation),
        kind="code_generation_output",
    )
    assert _completed_codegen(generation, expected_role="merged")
    assert not _completed_codegen(generation, expected_role="base")

    execution = tmp_path / "execution"
    execution.mkdir()
    (execution / "code_execution_run.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "status": "passed",
                "formal_result_eligible": True,
                "execution_mode": "cpu_sandbox_formal",
                "source": {
                    "commit_sha": "source",
                    "source_dirty": False,
                },
                "metrics": {
                    "humaneval_pass_at_1": 0.1,
                    "humaneval_plus_pass_at_1": 0.05,
                    "mbpp_pass_at_1": 0.2,
                    "mbpp_plus_pass_at_1": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    write_directory_integrity(
        str(execution),
        kind="code_execution_output",
    )
    assert _completed_execution(execution)


def _aggregate_fixture(method, seed, score):
    return {
        "seed": seed,
        "model_id": "qwen3_0_6b",
        "method": method,
        "protocol_track": "B",
        "base_snapshot_integrity_sha256": "base-sha",
        "training_commit_sha": "source-sha",
        "training_protocol_config_hash": f"{method}-protocol",
        "training_comparison_config_hash": "comparison-sha",
        "training_environment_identity": {"software": "training"},
        "evaluation_protocol_config_hash": "eval-protocol",
        "dataset_evidence": {"manifest_sha256": "dataset-sha"},
        "evaluator": {"commit_sha": "evaluator-sha"},
        "code_evaluation_packages": {"evalplus": "0.3.1"},
        "evaluation_software_environment": {"python": "3.10.20"},
        "source": {
            "commit_sha": "source-sha",
            "source_dirty": False,
        },
        "formal_result_eligible": True,
        "metrics": {metric: score for metric in METRICS},
        "run_dir": f"/execution/{method}/{seed}",
        "run_report_sha256": f"report-{method}-{seed}",
        "generation_dir": f"/generation/{method}/{seed}",
    }


def test_evalplus_seed_aggregation_and_paired_identity_are_exact():
    rows = [
        _aggregate_fixture("lora_nf", seed, score)
        for seed, score in ((42, 0.4), (43, 0.5), (44, 0.6))
    ]
    aggregate = _aggregate_group(
        rows,
        expected_seeds=[42, 43, 44],
        bootstrap_samples=100,
        base_metrics={metric: 0.25 for metric in METRICS},
    )
    assert aggregate["metrics"]["humaneval_pass_at_1"]["mean"] == 0.5
    assert (
        aggregate["adapted_minus_base"]["humaneval_pass_at_1"]["mean"]
        == 0.25
    )

    comparison = _aggregate_group(
        [
            _aggregate_fixture("lora", seed, score)
            for seed, score in ((42, 0.3), (43, 0.4), (44, 0.5))
        ],
        expected_seeds=[42, 43, 44],
        bootstrap_samples=100,
    )
    _validate_paired_compatibility(aggregate, comparison)
    comparison["training_comparison_config_hash"] = "drifted"
    with pytest.raises(ValueError, match="normalized training protocol"):
        _validate_paired_compatibility(aggregate, comparison)

    with pytest.raises(ValueError, match="exact seeds"):
        _aggregate_group(
            rows[:2],
            expected_seeds=[42, 43, 44],
            bootstrap_samples=100,
        )


def test_evalplus_aggregation_revalidates_generation_and_execution(tmp_path):
    environment = {
        "python": "3.10.20",
        "packages": {"torch": "2.5.1"},
        "installed_distributions_sha256": "environment-sha",
        "torch_cuda_version": "12.4",
        "cudnn_version": 90100,
    }
    source = {"commit_sha": "source-sha", "source_dirty": False}
    evaluator = {"commit_sha": "evaluator-sha", "source_dirty": False}
    dataset_evidence = {"manifest_sha256": "dataset-sha"}
    packages = {"evalplus": "0.3.1"}
    model_manifest = {
        "formal_result_eligible": True,
        "execution_mode": "gpu_formal",
        "dtype": "torch.float32",
        "checkpoint_retention": (
            "ephemeral_delete_after_qualified_evaluation"
        ),
        "base_model_integrity_manifest_sha256": "base-sha",
        "training_seed": 42,
        "training_model_id": "qwen3_0_6b",
        "training_commit_sha": "source-sha",
        "method": "lora_nf",
        "protocol_track": "B",
        "run_id": "fixture",
        "training_protocol_config_hash": "training-protocol",
        "training_comparison_config_hash": "comparison-protocol",
        "training_environment_identity": {"software": "training"},
        "checkpoint_integrity_manifest_sha256": "adapter-manifest-sha",
        "checkpoint_integrity_tree_sha256": "adapter-tree-sha",
    }
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    samples = {}
    for dataset in ("humaneval", "mbpp"):
        path = generation_dir / f"{dataset}_samples.jsonl"
        path.write_text(
            json.dumps({"task_id": f"{dataset}/0", "solution": "pass"})
            + "\n",
            encoding="utf-8",
        )
        samples[dataset] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    generation_report = {
        "format_version": 1,
        "status": "generated",
        "formal_result_eligible": True,
        "execution_mode": "gpu_formal",
        "source": source,
        "evaluator": evaluator,
        "protocol_config_hash": "evaluation-protocol",
        "seed": 42,
        "model_role": "merged",
        "model_manifest": model_manifest,
        "model_integrity": {"manifest_sha256": "merged-sha"},
        "dataset_evidence": dataset_evidence,
        "samples": samples,
        "environment": environment,
        "code_evaluation_packages": packages,
    }
    (generation_dir / "code_generation_run.json").write_text(
        json.dumps(generation_report),
        encoding="utf-8",
    )
    generation_integrity = write_directory_integrity(
        str(generation_dir),
        kind="code_generation_output",
    )

    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()
    for dataset in ("humaneval", "mbpp"):
        (execution_dir / f"{dataset}_samples.jsonl").write_bytes(
            (generation_dir / f"{dataset}_samples.jsonl").read_bytes()
        )
    execution_report = {
        "format_version": 1,
        "status": "passed",
        "formal_result_eligible": True,
        "execution_mode": "cpu_sandbox_formal",
        "source": source,
        "evaluator": evaluator,
        "protocol_config_hash": "evaluation-protocol",
        "seed": 42,
        "generation_dir": str(generation_dir.resolve()),
        "generation_integrity": generation_integrity,
        "generation_report": generation_report,
        "dataset_evidence": dataset_evidence,
        "sandbox": {"network": "disabled", "unshare_all": True},
        "metrics": {metric: 0.5 for metric in METRICS},
        "environment": environment,
        "code_evaluation_packages": packages,
    }
    (execution_dir / "code_execution_run.json").write_text(
        json.dumps(execution_report),
        encoding="utf-8",
    )
    write_directory_integrity(
        str(execution_dir),
        kind="code_execution_output",
    )
    record = _validate_execution_run(
        execution_dir,
        allow_cpu_smoke=False,
    )
    assert record["seed"] == 42
    assert record["method"] == "lora_nf"
    assert record["base_snapshot_integrity_sha256"] == "base-sha"
