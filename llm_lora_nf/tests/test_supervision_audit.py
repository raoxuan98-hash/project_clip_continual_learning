import pytest

from llm_lora_nf.supervision_audit import summarize_integer_series


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
