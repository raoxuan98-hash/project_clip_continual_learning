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
