import pytest

from llm_lora_nf.stats import paired_differences, summarize_values


def test_summarize_values_reports_sample_standard_deviation():
    summary = summarize_values(
        [1.0, 2.0, 3.0],
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert summary["count"] == 3
    assert summary["mean"] == 2.0
    assert summary["sample_standard_deviation"] == 1.0
    lower, upper = summary["seed_level_bootstrap_95ci"]
    assert lower <= summary["mean"] <= upper


def test_paired_differences_are_ordered_by_seed():
    differences = paired_differences(
        {43: 0.8, 42: 0.7, 44: 0.9},
        {42: 0.5, 43: 0.4, 44: 0.6},
    )
    assert differences == pytest.approx([0.2, 0.4, 0.3])


def test_paired_differences_reject_mismatched_seeds():
    with pytest.raises(ValueError, match="identical seed sets"):
        paired_differences({42: 1.0}, {43: 1.0})
