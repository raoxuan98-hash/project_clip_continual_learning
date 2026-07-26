import json

import numpy as np

from scripts.bootstrap_paired_samples import (
    _bootstrap_means,
    _metric_and_filter,
    _paired_values,
)


def test_metric_and_filter_parses_lm_eval_metric_key():
    assert _metric_and_filter("exact_match,flexible-extract") == (
        "exact_match",
        "flexible-extract",
    )
    assert _metric_and_filter("math_verify,none") == ("math_verify", "none")


def test_paired_bootstrap_reuses_document_indices():
    comparison = np.asarray([0.0, 0.2, 0.8, 1.0])
    reference = comparison + 0.25
    reference_means, comparison_means = _bootstrap_means(
        reference,
        comparison,
        samples=100,
        seed=11,
        chunk_size=9,
    )
    np.testing.assert_allclose(
        reference_means - comparison_means,
        np.full(100, 0.25),
    )


def test_paired_values_selects_filter_and_validates_hashes(tmp_path):
    rows = [
        {
            "doc_id": 0,
            "filter": "none",
            "doc_hash": "doc0",
            "prompt_hash": "prompt0",
            "target_hash": "target0",
            "exact_match": 1,
        },
        {
            "doc_id": 0,
            "filter": "other",
            "doc_hash": "doc0",
            "prompt_hash": "prompt0",
            "target_hash": "target0",
            "exact_match": 0,
        },
    ]
    reference_path = tmp_path / "reference.jsonl"
    comparison_path = tmp_path / "comparison.jsonl"
    encoded = "".join(json.dumps(row) + "\n" for row in rows)
    reference_path.write_text(encoded, encoding="utf-8")
    comparison_path.write_text(encoded, encoding="utf-8")
    reference, comparison, audit = _paired_values(
        reference_path,
        comparison_path,
        metric="exact_match",
        filter_name="none",
    )
    np.testing.assert_array_equal(reference, [1.0])
    np.testing.assert_array_equal(comparison, [1.0])
    assert audit[0]["doc_hash"] == "doc0"
