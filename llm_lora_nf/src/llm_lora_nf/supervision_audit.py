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
