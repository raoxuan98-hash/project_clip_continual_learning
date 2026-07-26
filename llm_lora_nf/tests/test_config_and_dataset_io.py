import json
from pathlib import Path

from llm_lora_nf.config_io import (
    canonical_comparison_config_hash,
    load_yaml_config,
)
from llm_lora_nf.dataset_io import load_metamath_examples


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
