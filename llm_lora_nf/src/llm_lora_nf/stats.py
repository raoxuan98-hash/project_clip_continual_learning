import math
import random
from typing import Dict, Iterable, List


def _percentile(sorted_values: List[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty collection")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def summarize_values(
    values: Iterable[float],
    *,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20260725,
) -> Dict[str, object]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("At least one value is required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    count = len(materialized)
    mean = sum(materialized) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in materialized) / (
            count - 1
        )
        standard_deviation = math.sqrt(variance)
    else:
        standard_deviation = 0.0
    generator = random.Random(bootstrap_seed)
    bootstrap_means = sorted(
        sum(generator.choices(materialized, k=count)) / count
        for _ in range(bootstrap_samples)
    )
    return {
        "values": materialized,
        "count": count,
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "seed_level_bootstrap_95ci": [
            _percentile(bootstrap_means, 0.025),
            _percentile(bootstrap_means, 0.975),
        ],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def paired_differences(
    reference_by_seed: Dict[int, float],
    comparison_by_seed: Dict[int, float],
) -> List[float]:
    if set(reference_by_seed) != set(comparison_by_seed):
        raise ValueError("Paired comparison requires identical seed sets")
    return [
        float(reference_by_seed[seed]) - float(comparison_by_seed[seed])
        for seed in sorted(reference_by_seed)
    ]
