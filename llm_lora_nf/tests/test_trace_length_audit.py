from llm_lora_nf.trace_length_audit import summarize_trace_token_lengths


def test_trace_length_summary_detects_fully_truncated_responses():
    records = [
        {
            "source_tokens": 100,
            "full_tokens": 110,
            "response_start_token": 101,
            "response_tokens": 9,
        },
        {
            "source_tokens": 2000,
            "full_tokens": 2050,
            "response_start_token": 2001,
            "response_tokens": 49,
        },
    ]
    summary = summarize_trace_token_lengths(
        records,
        thresholds=(1536, 2048, 4096),
    )
    assert summary["rows"] == 2
    assert summary["minimum_length_for_one_response_token_all_rows"] == 2002
    assert summary["minimum_length_for_full_coverage_all_rows"] == 2050
    assert summary["smallest_audited_safe_length"] == 2048
    assert (
        summary["coverage_by_max_sequence_length"]["1536"][
            "fully_truncated_response_rows"
        ]
        == 1
    )
    assert (
        summary["coverage_by_max_sequence_length"]["2048"][
            "partially_truncated_full_rows"
        ]
        == 1
    )
