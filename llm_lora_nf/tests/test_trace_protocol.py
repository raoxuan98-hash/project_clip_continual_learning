import json

import pytest

from llm_lora_nf.trace_protocol import (
    TRACE_ALTERNATE_ORDER,
    TRACE_OFFICIAL_ORDER,
    TRACE_TREELORA_PILOT_FILES,
    aggregate_trace_matrix,
    assert_task_order,
    audit_trace_dataset,
    normalize_trace_primary_score,
)


def _write_dataset(root, *, train_rows=5000, eval_rows=1000, test_rows=1000):
    for task in TRACE_OFFICIAL_ORDER:
        task_root = root / task
        task_root.mkdir(parents=True)
        for split, count in {
            "train": train_rows,
            "eval": eval_rows,
            "test": test_rows,
        }.items():
            rows = [
                {"prompt": f"{task}-{split}-{index}", "answer": "A"}
                for index in range(count)
            ]
            (task_root / f"{split}.json").write_text(
                json.dumps(rows, ensure_ascii=False),
                encoding="utf-8",
            )


def test_paper_trace_audit_checks_cardinality_but_does_not_self_qualify(tmp_path):
    _write_dataset(tmp_path)
    audit = audit_trace_dataset(str(tmp_path), protocol="paper_5k")
    assert len(audit["files"]) == 24
    assert audit["formal_eligible"] is False
    assert audit["qualification"] == (
        "cardinality_only_requires_locked_exact_source_manifest"
    )


def test_paper_trace_audit_rejects_reduced_training_or_evaluation(tmp_path):
    _write_dataset(tmp_path, train_rows=500, eval_rows=100, test_rows=100)
    with pytest.raises(ValueError, match="5000 train rows"):
        audit_trace_dataset(str(tmp_path), protocol="paper_5k")


def test_trace_audit_rejects_schema_drift(tmp_path):
    _write_dataset(tmp_path)
    path = tmp_path / "C-STANCE" / "train.json"
    path.write_text(
        json.dumps([{"instruction": "q", "answer": "A"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly prompt/answer"):
        audit_trace_dataset(str(tmp_path), protocol="paper_5k")


def test_registered_pilot_has_all_exact_file_identities():
    assert set(TRACE_TREELORA_PILOT_FILES) == set(TRACE_OFFICIAL_ORDER)
    for task in TRACE_OFFICIAL_ORDER:
        assert set(TRACE_TREELORA_PILOT_FILES[task]) == {
            "train",
            "eval",
            "test",
        }
        for rows, sha256 in TRACE_TREELORA_PILOT_FILES[task].values():
            assert rows > 0
            assert len(sha256) == 64


def test_trace_primary_score_normalization_locks_metric_scales():
    assert normalize_trace_primary_score("C-STANCE", 0.75) == 0.75
    assert normalize_trace_primary_score("Py150", 75.0) == 0.75
    assert normalize_trace_primary_score("20Minuten", 42.0) == 0.42
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        normalize_trace_primary_score("Py150", 101.0)


def test_trace_matrix_metrics_preserve_paper_bwt_denominator():
    matrix = [
        [0.8],
        [0.7, 0.6],
        [0.6, 0.5, 0.9],
    ]
    result = aggregate_trace_matrix(matrix)
    assert result["op_by_time"] == pytest.approx([0.8, 0.65, 2.0 / 3.0])
    # TRACE divides the two old-task differences by all three tasks seen.
    assert result["final_bwt_trace_definition"] == pytest.approx(-0.1)
    assert result["final_average"] == pytest.approx(2.0 / 3.0)
    assert result["forgetting_by_task"] == pytest.approx([0.2, 0.1])
    assert result["final_forgetting_standard"] == pytest.approx(0.15)


def test_trace_matrix_rejects_non_triangular_or_out_of_range_values():
    with pytest.raises(ValueError, match="row 1"):
        aggregate_trace_matrix([[0.5], [0.4]])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        aggregate_trace_matrix([[0.5], [0.4, 1.2]])


def test_only_two_preregistered_orders_are_accepted():
    assert assert_task_order(TRACE_OFFICIAL_ORDER) == "official"
    assert assert_task_order(TRACE_ALTERNATE_ORDER) == "alternate"
    with pytest.raises(ValueError, match="not one of"):
        assert_task_order(reversed(TRACE_OFFICIAL_ORDER))
