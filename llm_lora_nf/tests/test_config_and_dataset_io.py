import hashlib
import json
from pathlib import Path

import pytest

from llm_lora_nf.config_io import (
    canonical_comparison_config_hash,
    load_yaml_config,
)
from llm_lora_nf.dataset_io import (
    load_metamath_examples,
    load_pissa_codefeedback_python_examples,
)


def test_yaml_extends_deep_merges(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "run:\n  seed: 42\n  mode: base\nadapter:\n  rank: 8\n  alpha: 8\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: base.yaml\nrun:\n  mode: child\nadapter:\n  rank: 4\n",
        encoding="utf-8",
    )
    config = load_yaml_config(str(child))
    assert config["run"] == {"seed": 42, "mode": "child"}
    assert config["adapter"] == {"rank": 4, "alpha": 8}


def test_metamath_loader_preserves_official_order(tmp_path):
    path = tmp_path / "metamath.json"
    rows = [
        {"query": "q0", "response": "a0"},
        {"query": "q1", "response": "a1"},
        {"query": "q2", "response": "a2"},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    examples = load_metamath_examples(str(path), first_n=2)
    assert [example.instruction for example in examples] == ["q0", "q1"]
    assert [example.response for example in examples] == ["a0", "a1"]


def test_pissa_codefeedback_loader_validates_schema_order_and_identity(tmp_path):
    path = tmp_path / "train.json"
    rows = [
        {"instruction": "write zero", "output": "def zero(): return 0"},
        {"instruction": "write one", "output": "def one(): return 1"},
    ]
    encoded = json.dumps(rows).encode("utf-8")
    path.write_bytes(encoded)
    examples = load_pissa_codefeedback_python_examples(
        str(path),
        first_n=2,
        expected_sha256=hashlib.sha256(encoded).hexdigest(),
        expected_size_bytes=len(encoded),
        expected_total_rows=2,
    )
    assert [example.instruction for example in examples] == [
        "write zero",
        "write one",
    ]
    assert [example.response for example in examples] == [
        "def zero(): return 0",
        "def one(): return 1",
    ]


def test_pissa_codefeedback_loader_rejects_identity_and_schema_drift(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps([{"instruction": "q", "wrong": "a"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        load_pissa_codefeedback_python_examples(
            str(path),
            first_n=1,
            expected_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="instruction/output"):
        load_pissa_codefeedback_python_examples(str(path), first_n=1)


def test_pissa_codefeedback_loader_locks_known_empty_output_indices(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(
            [
                {"instruction": "q0", "output": "a0"},
                {"instruction": "q1", "output": ""},
            ]
        ),
        encoding="utf-8",
    )
    examples = load_pissa_codefeedback_python_examples(
        str(path),
        first_n=2,
        allowed_empty_output_indices=(1,),
    )
    assert examples[1].response == ""
    with pytest.raises(ValueError, match="empty output"):
        load_pissa_codefeedback_python_examples(str(path), first_n=2)


def test_track_b_method_configs_share_normalized_training_protocol():
    config_root = Path(__file__).resolve().parents[1] / "configs" / "paper"
    methods = [
        "lora",
        "dora",
        "lora_null",
        "pissa",
        "milora",
        "corda",
        "lora_nf",
    ]
    hashes = {
        canonical_comparison_config_hash(
            load_yaml_config(str(config_root / f"math_{method}.yaml"))
        )
        for method in methods
    }
    assert len(hashes) == 1


def test_code_method_configs_share_normalized_training_protocol():
    config_root = Path(__file__).resolve().parents[1] / "configs" / "paper"
    methods = [
        "lora",
        "dora",
        "lora_null",
        "pissa",
        "milora",
        "corda",
        "lora_nf",
    ]
    hashes = {
        canonical_comparison_config_hash(
            load_yaml_config(str(config_root / f"code_{method}.yaml"))
        )
        for method in methods
    }
    assert len(hashes) == 1
