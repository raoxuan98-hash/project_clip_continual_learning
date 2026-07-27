import copy

import pytest

from llm_lora_nf.trace_filter_ablation import (
    TRACE_FILTER_SETTINGS,
    aggregate_trace_filter_reports,
)


def _spec():
    return {
        "seed": 42,
        "order": "official",
        "settings": {
            setting: {} for setting in TRACE_FILTER_SETTINGS
        },
        "pilot_overrides": {
            "train_first_n": None,
            "test_first_n": 20,
            "max_steps": None,
            "max_new_tokens": 128,
            "calibration_samples": 64,
            "delete_checkpoint_after_smoke": True,
        },
        "selection": {
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
        },
        "resources": {
            "max_parallel_gpus": 3,
            "reserve_idle_gpus": 1,
        },
        "qualification": {
            "data_protocol": "treelora_500_pilot",
            "execution_mode": "gpu_chain_smoke_only",
            "formal_result_eligible": False,
            "purpose": (
                "one_preregistered_filter_strength_check_after_negative_pilot"
            ),
        },
    }


def _report(method, label, average, forgetting, bwt):
    return {
        "identity": {
            "method": method,
            "commit_sha": f"commit-{label}",
            "config_hash": f"config-{label}",
            "comparison_config_hash": f"comparison-{label}",
            "seed": 42,
            "order": "official",
            "execution_mode": "gpu_chain_smoke_only",
            "state_update": (
                "reference_fixed" if method == "lora_nf" else "none"
            ),
            "model_id": "Qwen/Qwen3-0.6B",
            "active_task_order": [f"task-{index}" for index in range(8)],
            "data_identity": {
                "protocol": "treelora_500_pilot",
                "files_sha256": "data-sha",
            },
        },
        "formal_result_eligible": False,
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
            "final_average": average,
            "final_forgetting_standard": forgetting,
            "final_bwt_trace_definition": bwt,
        },
        "peak_gpu_memory_bytes": 123,
        "wall_seconds": 456.0,
    }


def _inputs():
    reference = _report("lora", "lora", 0.33, 0.09, -0.04)
    candidates = {
        "e20_r02": _report(
            "lora_nf", "e20_r02", 0.31, 0.05, -0.03
        ),
        "e20_r05": _report(
            "lora_nf", "e20_r05", 0.327, 0.07, -0.041
        ),
        "e20_r10": _report(
            "lora_nf", "e20_r10", 0.34, 0.085, -0.03
        ),
        "e20_r20": _report(
            "lora_nf", "e20_r20", 0.30, 0.04, -0.02
        ),
        "e10_r10": _report(
            "lora_nf", "e10_r10", 0.32, 0.06, -0.06
        ),
    }
    commits = {
        label: f"commit-{label}"
        for label in ("lora", *TRACE_FILTER_SETTINGS)
    }
    hashes = {
        label: f"config-{label}"
        for label in ("lora", *TRACE_FILTER_SETTINGS)
    }
    return reference, candidates, commits, hashes


def test_trace_filter_ablation_validates_and_selects():
    reference, candidates, commits, hashes = _inputs()
    result = aggregate_trace_filter_reports(
        _spec(),
        reference,
        candidates,
        expected_run_commits=commits,
        expected_config_hashes=hashes,
    )
    assert result["formal_result_eligible"] is False
    assert result["selection"]["passed_settings"] == ["e20_r05"]
    assert result["selection"]["selected_setting"] == "e20_r05"


def test_trace_filter_ablation_rejects_config_drift():
    reference, candidates, commits, hashes = _inputs()
    drifted = copy.deepcopy(candidates)
    drifted["e20_r10"]["identity"]["config_hash"] = "wrong"
    with pytest.raises(ValueError, match="config hash mismatch"):
        aggregate_trace_filter_reports(
            _spec(),
            reference,
            drifted,
            expected_run_commits=commits,
            expected_config_hashes=hashes,
        )


def test_trace_filter_ablation_rejects_data_drift():
    reference, candidates, commits, hashes = _inputs()
    drifted = copy.deepcopy(candidates)
    drifted["e10_r10"]["identity"]["data_identity"][
        "files_sha256"
    ] = "different"
    with pytest.raises(ValueError, match="comparison identity mismatch"):
        aggregate_trace_filter_reports(
            _spec(),
            reference,
            drifted,
            expected_run_commits=commits,
            expected_config_hashes=hashes,
        )
