import copy

import pytest

from llm_lora_nf.trace_method_pilot import (
    TRACE_PILOT_METHODS,
    aggregate_trace_method_reports,
)


def _spec():
    return {
        "seed": 42,
        "order": "official",
        "jobs": {method: {} for method in TRACE_PILOT_METHODS},
        "pilot_overrides": {
            "train_first_n": None,
            "test_first_n": 20,
            "max_steps": None,
            "max_new_tokens": 128,
        },
        "qualification": {
            "data_protocol": "treelora_500_pilot",
            "execution_mode": "gpu_chain_smoke_only",
            "formal_result_eligible": False,
            "purpose": (
                "controlled_four_method_trace500_pilot_after_state_selection"
            ),
        },
        "locked_lora_nf_state": "reference_fixed",
    }


def _reports():
    reports = {}
    for index, method in enumerate(TRACE_PILOT_METHODS):
        reports[method] = {
            "formal_result_eligible": False,
            "identity": {
                "method": method,
                "commit_sha": "abc123",
                "seed": 42,
                "order": "official",
                "execution_mode": "gpu_chain_smoke_only",
                "comparison_config_hash": "shared",
                "state_update": "reference_fixed",
                "data_identity": {"protocol": "treelora_500_pilot"},
            },
            "effective_limits": {
                "tasks": 8,
                "train_first_n": None,
                "test_first_n": 20,
                "max_steps": None,
                "generation_max_new_tokens": 128,
                "training_max_sequence_length": 1536,
                "generation_max_prompt_length": 1024,
            },
            "checkpoint_retention": {
                "deleted_after_smoke_audit": True,
            },
            "summary": {
                "final_average": 0.2 + 0.01 * index,
                "final_bwt_trace_definition": -0.1,
                "final_forgetting_standard": 0.05,
            },
            "peak_gpu_memory_bytes": 1000 + index,
            "wall_seconds": 10.0 + index,
        }
    return reports


def test_trace_method_pilot_aggregates_one_locked_comparison():
    result = aggregate_trace_method_reports(
        _spec(),
        _reports(),
        expected_run_commit="abc123",
    )
    assert result["formal_result_eligible"] is False
    assert result["comparison_config_hash"] == "shared"
    assert result["ranking_by_final_average"] == [
        "lora_nf",
        "lora_null",
        "dora",
        "lora",
    ]
    assert result["rows"][-1]["final_average_minus_lora_nf"] == 0.0


def test_trace_method_pilot_rejects_cross_protocol_report():
    reports = copy.deepcopy(_reports())
    reports["dora"]["identity"]["comparison_config_hash"] = "different"
    with pytest.raises(ValueError, match="comparison hash"):
        aggregate_trace_method_reports(
            _spec(),
            reports,
            expected_run_commit="abc123",
        )
