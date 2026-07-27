import pytest

from llm_lora_nf.supervision_audit import (
    summarize_integer_series,
    summarize_shuffled_zero_schedule,
)


def test_supervision_summary_preserves_zero_row_identity():
    summary = summarize_integer_series([4, 0, 2, 0, 8])
    assert summary["rows"] == 5
    assert summary["sum"] == 14
    assert summary["mean"] == 2.8
    assert summary["minimum"] == 0
    assert summary["p50_lower"] == 2
    assert summary["maximum"] == 8
    assert summary["zero_count"] == 2
    assert summary["zero_indices"] == [1, 3]
    assert len(summary["zero_indices_sha256"]) == 64


def test_supervision_summary_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="At least one"):
        summarize_integer_series([])
    with pytest.raises(ValueError, match="non-negative"):
        summarize_integer_series([1, -1])


def test_shuffled_zero_schedule_tracks_groups_and_discarded_rows():
    summary = summarize_shuffled_zero_schedule(
        [2, 0, 3, 1, 5, 4],
        zero_source_indices=[0, 1, 4],
        consumed_rows=4,
        accumulation_steps=2,
    )
    assert summary["zero_total"] == 3
    assert summary["zero_consumed"] == 2
    assert summary["zero_discarded"] == 1
    assert summary["zero_consumed_positions"] == [1, 3]
    assert summary["zero_discarded_source_indices"] == [4]
    assert summary["max_zero_micro_batches_in_group"] == 1
    assert summary["all_zero_group_count"] == 0


def test_shuffled_zero_schedule_rejects_incomplete_group():
    with pytest.raises(ValueError, match="complete accumulation"):
        summarize_shuffled_zero_schedule(
            [0, 1, 2],
            zero_source_indices=[],
            consumed_rows=3,
            accumulation_steps=2,
        )
