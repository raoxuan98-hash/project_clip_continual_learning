import copy
from pathlib import Path

import pytest
import yaml

from llm_lora_nf.artifact_retention import (
    LOCKED_ARTIFACT_RETENTION,
    validate_artifact_retention,
)
from llm_lora_nf.config_io import (
    canonical_comparison_config_hash,
    load_yaml_config,
)
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


def test_all_code_track_b_paper_configs_pass_locked_validation():
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
            str(CONFIG_ROOT / f"code_{method}.yaml")
        )
        validate_track_b_formal_config(config)


def test_code_track_b_validation_rejects_data_identity_drift():
    config = load_yaml_config(str(CONFIG_ROOT / "code_lora_nf.yaml"))
    changed = copy.deepcopy(config)
    changed["train"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="train.file_sha256"):
        validate_track_b_formal_config(changed)


def test_code_track_b_validation_rejects_optimizer_step_drift():
    config = load_yaml_config(str(CONFIG_ROOT / "code_lora_nf.yaml"))
    changed = copy.deepcopy(config)
    changed["train"]["optimizer_steps_per_epoch"] = 820
    with pytest.raises(ValueError, match="optimizer_steps_per_epoch"):
        validate_track_b_formal_config(changed)


def test_track_b_validation_rejects_budget_drift():
    config = load_yaml_config(str(CONFIG_ROOT / "math_lora_nf.yaml"))
    changed = copy.deepcopy(config)
    changed["train"]["learning_rate"] = 3e-5
    with pytest.raises(ValueError, match="train.learning_rate"):
        validate_track_b_formal_config(changed)


def test_track_b_validation_rejects_response_losing_truncation():
    config = load_yaml_config(str(CONFIG_ROOT / "math_lora_nf.yaml"))
    changed = copy.deepcopy(config)
    changed["train"]["truncation_strategy"] = "right"
    with pytest.raises(ValueError, match="train.truncation_strategy"):
        validate_track_b_formal_config(changed)


def test_track_b_validation_rejects_dataloader_worker_drift():
    config = load_yaml_config(str(CONFIG_ROOT / "math_lora_nf.yaml"))
    changed = copy.deepcopy(config)
    changed["train"]["dataloader_num_workers"] = 2
    with pytest.raises(ValueError, match="train.dataloader_num_workers"):
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


@pytest.mark.parametrize("matrix_name", ["math_matrix.yaml", "code_matrix.yaml"])
def test_main_matrix_is_complete_and_comparison_identical(matrix_name):
    matrix = yaml.safe_load(
        (CONFIG_ROOT / matrix_name).read_text(encoding="utf-8")
    )
    main = matrix["stages"]["main"]
    assert main["runner"] == "track_b"
    assert main["models"] == ["llama3p2_3b", "qwen25_1p5b"]
    assert main["methods"] == [
        "lora",
        "dora",
        "lora_null",
        "pissa",
        "milora",
        "corda",
        "lora_nf",
    ]
    assert main["seeds"] == [42, 43, 44]

    for model_key in main["models"]:
        model_spec = matrix["models"][model_key]
        base = load_yaml_config(str(CONFIG_ROOT / model_spec["config"]))
        comparison_hashes = set()
        for seed in main["seeds"]:
            for method in main["methods"]:
                config = copy.deepcopy(base)
                config["run"]["seed"] = seed
                config["run"]["name"] = (
                    f"main_{model_key}_{method}_seed{seed}"
                )
                config["adapter"]["method"] = method
                if method == "corda":
                    config["adapter"]["corda_mode"] = "kpm"
                validate_track_b_formal_config(config)
                comparison_hashes.add(
                    canonical_comparison_config_hash(config)
                )
        assert len(comparison_hashes) == 1
