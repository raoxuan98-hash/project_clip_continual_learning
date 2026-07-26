import copy
from pathlib import Path

from llm_lora_nf.artifact_retention import validate_artifact_retention
from llm_lora_nf.config_io import (
    canonical_comparison_config_hash,
    canonical_trace_state_ablation_config_hash,
    load_yaml_config,
)
from llm_lora_nf.trace_protocol import TRACE_OFFICIAL_ORDER


CONFIG_ROOT = (
    Path(__file__).resolve().parents[1] / "configs" / "continual"
)


def _load(name: str):
    return load_yaml_config(str(CONFIG_ROOT / name))


def test_trace_main_methods_share_one_comparison_protocol():
    names = (
        "trace_qwen3_0p6b_pilot_lora.yaml",
        "trace_qwen3_0p6b_pilot_dora.yaml",
        "trace_qwen3_0p6b_pilot_lora_null.yaml",
        "trace_qwen3_0p6b_pilot_lora_nf_fixed.yaml",
    )
    configs = [_load(name) for name in names]
    hashes = {
        canonical_comparison_config_hash(config)
        for config in configs
    }
    assert len(hashes) == 1
    assert [config["adapter"]["method"] for config in configs] == [
        "lora",
        "dora",
        "lora_null",
        "lora_nf",
    ]


def test_trace_config_locks_attention_instruct_and_low_storage_policy():
    config = _load("trace_qwen3_0p6b_pilot_base.yaml")
    validate_artifact_retention(config["artifact_retention"])
    assert config["model"]["id"] == "Qwen/Qwen3-0.6B"
    assert config["model"]["checkpoint_type"] == "instruct"
    assert config["adapter"]["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert config["train"]["global_batch_size"] == 128
    assert config["train"]["learning_rate"] == 1e-4
    assert config["train"]["max_sequence_length"] == 1536
    assert config["train"]["truncation"] == {
        "protocol": "trace_official_combined_left_v1",
        "max_prompt_length": 1024,
        "max_answer_length": 512,
        "side": "left",
        "preserve_response": True,
    }
    assert config["generation"]["do_sample"] is False
    assert config["generation"]["num_beams"] == 1
    assert len(TRACE_OFFICIAL_ORDER) == 8


def test_trace_history_ablation_changes_only_declared_state_rule():
    fixed = _load("trace_qwen3_0p6b_pilot_lora_nf_fixed.yaml")
    history = _load("trace_qwen3_0p6b_pilot_lora_nf_history.yaml")
    assert canonical_trace_state_ablation_config_hash(
        fixed
    ) == canonical_trace_state_ablation_config_hash(history)
    fixed["run"].pop("name")
    history["run"].pop("name")
    fixed["trace"].pop("state_update")
    history["trace"].pop("state_update")
    assert fixed == history


def test_trace_four_method_pilot_is_preregistered_and_nonformal():
    spec = _load("trace_qwen3_0p6b_method_pilot.yaml")
    assert spec["format_version"] == 1
    assert list(spec["jobs"]) == [
        "lora",
        "dora",
        "lora_null",
        "lora_nf",
    ]
    assert spec["pilot_overrides"] == {
        "train_first_n": None,
        "test_first_n": 20,
        "max_steps": None,
        "max_new_tokens": 128,
        "calibration_samples": 64,
        "delete_checkpoint_after_smoke": True,
    }
    assert spec["resources"] == {
        "max_parallel_gpus": 3,
        "reserve_idle_gpus": 1,
    }
    assert spec["qualification"]["formal_result_eligible"] is False
    assert spec["locked_lora_nf_state"] == "reference_fixed"


def test_trace_filter_check_is_small_preregistered_and_fixed_state():
    spec = _load("trace_qwen3_0p6b_filter_ablation.yaml")
    assert list(spec["settings"]) == [
        "e20_r02",
        "e20_r05",
        "e20_r10",
        "e20_r20",
        "e10_r10",
    ]
    assert spec["selection"] == {
        "reference_method": "lora",
        "maximum_final_average_gap": 0.005,
        "minimum_forgetting_improvement": 0.01,
        "maximum_final_bwt_regression": 0.005,
        "tie_break_order": [
            "highest_final_average",
            "lowest_forgetting",
            "highest_bwt",
        ],
        "default_on_failed_gate": "e20_r02",
    }
    base = _load("trace_qwen3_0p6b_pilot_lora_nf_fixed.yaml")
    for setting in ("e20_r05", "e20_r10", "e20_r20", "e10_r10"):
        config = _load(
            f"trace_qwen3_0p6b_pilot_lora_nf_{setting}.yaml"
        )
        assert config["trace"]["state_update"] == "reference_fixed"
        assert config["adapter"]["method"] == "lora_nf"
        assert config["adapter"]["filter"]["energy_fraction"] == (
            spec["settings"][setting]["energy_fraction"]
        )
        assert config["adapter"]["filter"]["leakage"] == (
            spec["settings"][setting]["leakage"]
        )
        normalized = copy.deepcopy(config)
        reference = copy.deepcopy(base)
        normalized["run"]["name"] = reference["run"]["name"]
        normalized["adapter"]["filter"] = reference["adapter"]["filter"]
        assert normalized == reference
