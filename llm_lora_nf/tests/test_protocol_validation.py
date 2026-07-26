import copy
from pathlib import Path

import pytest

from llm_lora_nf.artifact_retention import (
    LOCKED_ARTIFACT_RETENTION,
    validate_artifact_retention,
)
from llm_lora_nf.config_io import load_yaml_config
from llm_lora_nf.protocol_validation import (
    validate_track_a_formal_config,
    validate_track_b_formal_config,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "paper"


def test_locked_artifact_retention_policy_is_exact():
    validate_artifact_retention(LOCKED_ARTIFACT_RETENTION)
    changed = copy.deepcopy(LOCKED_ARTIFACT_RETENTION)
    changed["max_concurrent_merged_checkpoints"] = 2
    with pytest.raises(ValueError, match="low-storage"):
        validate_artifact_retention(changed)


def test_all_track_b_paper_configs_pass_locked_validation():
    for method in (
        "lora",
        "dora",
        "lora_null",
        "pissa",
        "milora",
        "corda",
        "lora_nf",
    ):
        config = load_yaml_config(
            str(CONFIG_ROOT / f"math_{method}.yaml")
        )
        validate_track_b_formal_config(config)


def test_track_b_validation_rejects_budget_drift():
    config = load_yaml_config(str(CONFIG_ROOT / "math_lora_nf.yaml"))
    changed = copy.deepcopy(config)
    changed["train"]["learning_rate"] = 3e-5
    with pytest.raises(ValueError, match="train.learning_rate"):
        validate_track_b_formal_config(changed)


def test_track_b_validation_rejects_checkpoint_retention_drift():
    config = load_yaml_config(str(CONFIG_ROOT / "math_lora_nf.yaml"))
    changed = copy.deepcopy(config)
    changed["artifact_retention"]["save_scheduler_state"] = True
    with pytest.raises(
        ValueError,
        match="artifact_retention.save_scheduler_state",
    ):
        validate_track_b_formal_config(changed)


def test_track_a_paper_config_passes_locked_validation():
    config = load_yaml_config(
        str(CONFIG_ROOT / "math_track_a_llama3p2_3b.yaml")
    )
    validate_track_a_formal_config(config)
