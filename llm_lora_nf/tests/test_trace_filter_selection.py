from llm_lora_nf.trace_filter_selection import (
    select_trace_filter_setting,
)


def _summary(average, forgetting, bwt):
    return {
        "final_average": average,
        "final_forgetting_standard": forgetting,
        "final_bwt_trace_definition": bwt,
    }


def test_trace_filter_selection_applies_all_preregistered_gates():
    result = select_trace_filter_setting(
        _summary(0.33, 0.09, -0.04),
        {
            "default": _summary(0.31, 0.05, -0.03),
            "balanced": _summary(0.327, 0.07, -0.041),
            "adaptive": _summary(0.34, 0.085, -0.03),
        },
        maximum_final_average_gap=0.005,
        minimum_forgetting_improvement=0.01,
        maximum_final_bwt_regression=0.005,
        default_on_failed_gate="default",
    )
    assert result["passed_settings"] == ["balanced"]
    assert result["selected_setting"] == "balanced"
    assert result["candidates"]["adaptive"]["gates"]["forgetting"] is False


def test_trace_filter_selection_defaults_when_no_setting_passes():
    result = select_trace_filter_setting(
        _summary(0.33, 0.09, -0.04),
        {
            "default": _summary(0.31, 0.05, -0.03),
            "other": _summary(0.32, 0.08, -0.06),
        },
        maximum_final_average_gap=0.005,
        minimum_forgetting_improvement=0.01,
        maximum_final_bwt_regression=0.005,
        default_on_failed_gate="default",
    )
    assert result["passed_settings"] == []
    assert result["selected_setting"] == "default"
