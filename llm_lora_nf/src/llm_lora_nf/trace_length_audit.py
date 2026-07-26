import math
from typing import Any, Dict, Mapping, Sequence


TRACE_LENGTH_THRESHOLDS = (1536, 2048, 4096, 8192, 16384)


def _longest_common_prefix(
    left: Sequence[int],
    right: Sequence[int],
) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def trace_example_token_lengths(
    tokenizer: Any,
    *,
    prompt: str,
    answer: str,
    enable_thinking: bool,
) -> Dict[str, int]:
    source_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=bool(enable_thinking),
    )
    source_ids = tokenizer(
        source_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    response_start = _longest_common_prefix(source_ids, full_ids)
    if response_start >= len(full_ids):
        raise ValueError("TRACE chat template produced no supervised response tokens")
    return {
        "source_tokens": len(source_ids),
        "full_tokens": len(full_ids),
        "response_start_token": response_start,
        "response_tokens": len(full_ids) - response_start,
    }


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ValueError("TRACE length summary requires non-empty values")
    ordered = sorted(int(value) for value in values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def summarize_trace_token_lengths(
    records: Sequence[Mapping[str, int]],
    *,
    thresholds: Sequence[int] = TRACE_LENGTH_THRESHOLDS,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("TRACE length audit requires non-empty records")
    fields = (
        "source_tokens",
        "full_tokens",
        "response_start_token",
        "response_tokens",
    )
    summary: Dict[str, Any] = {"rows": len(records)}
    for field in fields:
        values = [int(record[field]) for record in records]
        summary[field] = {
            "minimum": min(values),
            "p50": _nearest_rank(values, 0.50),
            "p95": _nearest_rank(values, 0.95),
            "p99": _nearest_rank(values, 0.99),
            "maximum": max(values),
        }
    coverage = {}
    for threshold in thresholds:
        if threshold <= 0:
            raise ValueError("TRACE length thresholds must be positive")
        right_fully_truncated = sum(
            int(record["response_start_token"]) >= threshold
            for record in records
        )
        right_partially_truncated = sum(
            int(record["full_tokens"]) > threshold
            and int(record["response_start_token"]) < threshold
            for record in records
        )
        fully_covered = sum(
            int(record["full_tokens"]) <= threshold
            for record in records
        )
        left_partial_response = sum(
            int(record["response_tokens"]) > threshold
            for record in records
        )
        coverage[str(int(threshold))] = {
            "right_truncation": {
                "fully_truncated_response_rows": right_fully_truncated,
                "partially_truncated_full_rows": right_partially_truncated,
                "fully_covered_rows": fully_covered,
            },
            "left_preserve_response": {
                "prompt_left_truncated_rows": len(records) - fully_covered,
                "fully_preserved_response_rows": (
                    len(records) - left_partial_response
                ),
                "partially_truncated_response_rows": left_partial_response,
                "fully_truncated_response_rows": 0,
                "fully_covered_rows": fully_covered,
                "minimum_retained_response_tokens": min(
                    min(int(record["response_tokens"]), int(threshold))
                    for record in records
                ),
            },
            # Backward-compatible aliases describe the old right-truncation
            # behavior so existing audit readers fail neither silently nor
            # ambiguously.
            "fully_truncated_response_rows": right_fully_truncated,
            "partially_truncated_full_rows": right_partially_truncated,
            "fully_covered_rows": fully_covered,
        }
    minimum_one_response_token = max(
        int(record["response_start_token"]) + 1
        for record in records
    )
    full_coverage = max(int(record["full_tokens"]) for record in records)
    summary["coverage_by_max_sequence_length"] = coverage
    summary["minimum_length_for_one_response_token_all_rows"] = (
        minimum_one_response_token
    )
    summary["minimum_length_for_full_coverage_all_rows"] = full_coverage
    summary["smallest_audited_safe_length"] = next(
        (
            int(threshold)
            for threshold in thresholds
            if threshold >= minimum_one_response_token
        ),
        None,
    )
    return summary
