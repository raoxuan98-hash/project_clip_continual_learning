import pytest

from llm_lora_nf.trace_data import write_trace_prediction_artifact
from llm_lora_nf.trace_metrics import (
    _sari_sentence,
    aggregate_trace_prediction_artifacts,
    evaluate_trace_rows,
)


def _rows(prediction, answer, prompt="source"):
    return [
        {
            "index": 0,
            "prompt": prompt,
            "prediction": prediction,
            "answer": answer,
        }
    ]


def test_trace_exact_match_and_scienceqa_first_character():
    exact = evaluate_trace_rows("C-STANCE", _rows("A", "A"))
    assert exact["normalized_score"] == 1.0
    strict = evaluate_trace_rows("FOMC", _rows("A because", "A"))
    assert strict["normalized_score"] == 0.0
    science = evaluate_trace_rows(
        "ScienceQA",
        _rows("B\nbecause model", "B\nbecause reference"),
    )
    assert science["normalized_score"] == 1.0


def test_trace_py150_reproduces_literal_postprocess_and_0_100_scale():
    result = evaluate_trace_rows(
        "Py150",
        _rows(
            "value = <NUM_LIT>",
            "value = 0",
        ),
    )
    assert result["raw_score"] == 100.0
    assert result["normalized_score"] == 1.0


def test_sari_matches_huggingface_evaluate_reference_example():
    score = 100.0 * _sari_sentence(
        "About 95 species are currently accepted .",
        "About 95 you now get in .",
        [
            "About 95 species are currently known .",
            "About 95 species are now accepted .",
            "95 species are now accepted .",
        ],
    )
    assert score == pytest.approx(26.953601953601954)


def test_meetingbank_rouge_l_uses_official_0_1_scale():
    result = evaluate_trace_rows(
        "MeetingBank",
        _rows("the same summary", "the same summary"),
    )
    assert result["raw_score"] == pytest.approx(1.0)
    assert result["normalized_score"] == pytest.approx(1.0)


def test_trace_artifact_aggregation_requires_complete_triangle(tmp_path):
    order = ("C-STANCE", "FOMC")
    paths = []
    for stage, tasks in ((1, order[:1]), (2, order)):
        for task in tasks:
            record = write_trace_prediction_artifact(
                str(tmp_path / f"stage_{stage}_{task}.json.gz"),
                task=task,
                stage=stage,
                task_order=order,
                predictions=_rows("A", "A"),
                generation={"do_sample": False, "num_beams": 1},
                identity={"seed": 42, "commit_sha": "a" * 40},
            )
            paths.append(record["path"])
    result = aggregate_trace_prediction_artifacts(
        paths,
        task_order=order,
    )
    assert result["summary"]["final_average"] == 1.0
    with pytest.raises(ValueError, match="triangular artifact set mismatch"):
        aggregate_trace_prediction_artifacts(
            paths[:-1],
            task_order=order,
        )
