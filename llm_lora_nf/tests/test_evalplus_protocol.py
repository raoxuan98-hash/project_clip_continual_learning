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
