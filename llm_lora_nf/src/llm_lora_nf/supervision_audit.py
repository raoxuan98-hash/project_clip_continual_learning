import hashlib
from typing import Dict, Iterable, List


def summarize_integer_series(values: Iterable[int]) -> Dict[str, object]:
    observed = [int(value) for value in values]
    if not observed:
        raise ValueError("At least one value is required")
    if any(value < 0 for value in observed):
        raise ValueError("Audit values must be non-negative")
    ordered = sorted(observed)

    def lower_quantile(fraction: float) -> int:
        return ordered[int((len(ordered) - 1) * fraction)]

    zero_indices: List[int] = [
        index for index, value in enumerate(observed) if value == 0
    ]
    zero_index_bytes = "\n".join(str(index) for index in zero_indices).encode(
        "ascii"
    )
    return {
        "rows": len(observed),
        "sum": sum(observed),
        "mean": sum(observed) / len(observed),
        "minimum": ordered[0],
        "p50_lower": lower_quantile(0.50),
        "p95_lower": lower_quantile(0.95),
        "p99_lower": lower_quantile(0.99),
        "maximum": ordered[-1],
        "zero_count": len(zero_indices),
        "zero_fraction": len(zero_indices) / len(observed),
        "zero_indices": zero_indices,
        "zero_indices_sha256": hashlib.sha256(zero_index_bytes).hexdigest(),
    }


def summarize_shuffled_zero_schedule(
    ordered_indices: Iterable[int],
    *,
    zero_source_indices: Iterable[int],
    consumed_rows: int,
    accumulation_steps: int,
) -> Dict[str, object]:
    order = [int(index) for index in ordered_indices]
    if not order:
        raise ValueError("Shuffled order must be non-empty")
    if sorted(order) != list(range(len(order))):
        raise ValueError("Shuffled order must be a permutation of row indices")
    if consumed_rows <= 0 or consumed_rows > len(order):
        raise ValueError("consumed_rows must lie within the shuffled order")
    if accumulation_steps <= 0 or consumed_rows % accumulation_steps != 0:
        raise ValueError(
            "consumed_rows must contain complete accumulation groups"
        )
    zero_sources = {int(index) for index in zero_source_indices}
    if any(index < 0 or index >= len(order) for index in zero_sources):
        raise ValueError("Zero-supervision source index is out of range")

    zero_positions = [
        position
        for position, source_index in enumerate(order)
        if source_index in zero_sources
    ]
    consumed_positions = [
        position for position in zero_positions if position < consumed_rows
    ]
    discarded_positions = [
        position for position in zero_positions if position >= consumed_rows
    ]
    group_counts: Dict[int, int] = {}
    for position in consumed_positions:
        group_index = position // accumulation_steps
        group_counts[group_index] = group_counts.get(group_index, 0) + 1
    nonempty_groups = [
        {
            "group_index": group_index,
            "zero_micro_batches": group_counts[group_index],
        }
        for group_index in sorted(group_counts)
    ]
    return {
        "rows": len(order),
        "consumed_rows": consumed_rows,
        "discarded_rows": len(order) - consumed_rows,
        "accumulation_steps": accumulation_steps,
        "optimizer_groups": consumed_rows // accumulation_steps,
        "zero_total": len(zero_positions),
        "zero_consumed": len(consumed_positions),
        "zero_discarded": len(discarded_positions),
        "zero_shuffled_positions": zero_positions,
        "zero_consumed_positions": consumed_positions,
        "zero_discarded_positions": discarded_positions,
        "zero_discarded_source_indices": [
            order[position] for position in discarded_positions
        ],
        "groups_with_zero_supervision": nonempty_groups,
        "max_zero_micro_batches_in_group": max(group_counts.values(), default=0),
        "all_zero_group_count": sum(
            count == accumulation_steps for count in group_counts.values()
        ),
    }
