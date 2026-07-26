import copy

import pytest

from llm_lora_nf.trace_state_selection import select_trace_state


def _report(state, *, average, bwt, wall):
    return {
        "status": "passed",
        "formal_result_eligible": False,
        "source_dirty": False,
        "identity": {
            "model_id": "Qwen/Qwen3-0.6B",
            "model_snapshot_revision": "revision",
            "seed": 42,
            "commit_sha": "a" * 40,
            "comparison_config_hash": "b" * 64,
            "state_ablation_config_hash": "d" * 64,
            "data_identity": {"files_sha256": "c" * 64},
            "execution_mode": "gpu_chain_smoke_only",
            "method": "lora_nf",
            "order": "official",
            "active_task_order": ["C-STANCE", "FOMC"],
            "state_update": state,
        },
        "checkpoint_retention": {
            "deleted_after_smoke_audit": True,
        },
        "summary": {
            "final_average": average,
            "final_bwt_trace_definition": bwt,
        },
        "wall_seconds": wall,
    }


def test_history_state_must_pass_all_preregistered_gates():
    fixed = _report(
        "reference_fixed",
        average=0.50,
        bwt=-0.02,
        wall=100.0,
    )
    history = _report(
        "reference_plus_history",
        average=0.51,
        bwt=-0.021,
        wall=120.0,
    )
    result = select_trace_state(
        fixed,
        history,
        minimum_final_average_gain=0.005,
        maximum_final_bwt_regression=0.005,
        maximum_wall_time_ratio=1.25,
    )
    assert result["selected_state"] == "reference_plus_history"
    assert all(result["gates"].values())

    slow = copy.deepcopy(history)
    slow["wall_seconds"] = 130.0
    result = select_trace_state(
        fixed,
        slow,
        minimum_final_average_gain=0.005,
        maximum_final_bwt_regression=0.005,
        maximum_wall_time_ratio=1.25,
    )
    assert result["selected_state"] == "reference_fixed"
    assert result["gates"]["maximum_wall_time_ratio"] is False


def test_state_selection_rejects_unpaired_or_formal_reports():
    fixed = _report(
        "reference_fixed",
        average=0.50,
        bwt=-0.02,
        wall=100.0,
    )
    history = _report(
        "reference_plus_history",
        average=0.51,
        bwt=-0.02,
        wall=110.0,
    )
    history["identity"]["seed"] = 43
    with pytest.raises(ValueError, match="identity mismatch: seed"):
        select_trace_state(
            fixed,
            history,
            minimum_final_average_gain=0.005,
            maximum_final_bwt_regression=0.005,
            maximum_wall_time_ratio=1.25,
        )
