import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .trace_data import read_trace_prediction_artifact
from .trace_protocol import (
    TRACE_OFFICIAL_ORDER,
    TRACE_PRIMARY_METRICS,
    aggregate_trace_matrix,
    normalize_trace_primary_score,
)


TRACE_EVALUATOR_COMMIT = "462e39f616134f4f819efeb3baea8638c03c7db4"
SARI_EVALUATE_COMMIT = "a7dd338386a4fae9a1767e05eb9ef9479513d9e8"
SARI_SOURCE_BLOB = "7d021184d0f45ae1fb01d86f8c5320a4a62fdf89"
TRACE_METRICS_FORMAT_VERSION = 1


def _accuracy(predictions: Sequence[str], answers: Sequence[str]) -> float:
    if len(predictions) != len(answers) or not predictions:
        raise ValueError("TRACE accuracy inputs must have equal non-zero length")
    return sum(
        prediction == answer
        for prediction, answer in zip(predictions, answers)
        if prediction != "" and answer != ""
    ) / len(predictions)


def _scienceqa_accuracy(
    predictions: Sequence[str],
    answers: Sequence[str],
) -> float:
    return _accuracy(
        [prediction[0] if prediction else "" for prediction in predictions],
        [answer[0] if answer else "" for answer in answers],
    )


def _postprocess_py150(code: str) -> str:
    result = (
        code.replace("<NUM_LIT>", "0")
        .replace("<STR_LIT>", "")
        .replace("<CHAR_LIT>", "")
    )
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    for literal_type, literal in pattern.findall(result):
        result = result.replace(
            f"<{literal_type}_LIT:{literal}>",
            literal,
        )
    return result


def _fuzzywuzzy_ratio(left: str, right: str) -> int:
    # fuzzywuzzy 0.18.0 without the optional python-Levenshtein acceleration
    # uses difflib.SequenceMatcher and rounds to the nearest integer.
    return int(round(100.0 * SequenceMatcher(None, left, right).ratio()))


def _py150_similarity(
    predictions: Sequence[str],
    answers: Sequence[str],
) -> float:
    if len(predictions) != len(answers) or not predictions:
        raise ValueError("TRACE Py150 inputs must have equal non-zero length")
    total = 0
    for prediction, answer in zip(predictions, answers):
        if prediction == "" or answer == "":
            continue
        total += _fuzzywuzzy_ratio(
            _postprocess_py150(prediction),
            _postprocess_py150(answer),
        )
    return total / len(predictions)


def _rouge_l(
    predictions: Sequence[str],
    answers: Sequence[str],
) -> float:
    if len(predictions) != len(answers) or not predictions:
        raise ValueError("TRACE ROUGE inputs must have equal non-zero length")
    try:
        from rouge import Rouge
    except ImportError as error:
        raise RuntimeError(
            "TRACE MeetingBank evaluation requires rouge==1.0.1"
        ) from error
    evaluator = Rouge(metrics=["rouge-l"])
    total = 0.0
    for prediction, answer in zip(predictions, answers):
        if prediction == "" or answer == "":
            continue
        total += float(
            evaluator.get_scores(answer, prediction, avg=True)["rouge-l"]["f"]
        )
    return total / len(predictions)


def _sari_ngram(
    source_ngrams: Sequence[str],
    candidate_ngrams: Sequence[str],
    reference_ngrams: Sequence[Sequence[str]],
) -> Tuple[float, float, float]:
    reference_count = len(reference_ngrams)
    reference_counter = Counter(
        ngram
        for reference in reference_ngrams
        for ngram in reference
    )
    source_counter = Counter(source_ngrams)
    source_repeated = Counter(
        {
            ngram: count * reference_count
            for ngram, count in source_counter.items()
        }
    )
    candidate_counter = Counter(candidate_ngrams)
    candidate_repeated = Counter(
        {
            ngram: count * reference_count
            for ngram, count in candidate_counter.items()
        }
    )

    keep = source_repeated & candidate_repeated
    good_keep = keep & reference_counter
    all_keep = source_repeated & reference_counter
    keep_precision = 1.0
    keep_recall = 1.0
    if keep:
        keep_precision = (
            sum(good_keep[ngram] / keep[ngram] for ngram in good_keep)
            / len(keep)
        )
    if all_keep:
        keep_recall = sum(good_keep.values()) / sum(all_keep.values())
    keep_score = 0.0
    if keep_precision > 0.0 or keep_recall > 0.0:
        keep_score = (
            2.0
            * keep_precision
            * keep_recall
            / (keep_precision + keep_recall)
        )

    deleted = source_repeated - candidate_repeated
    good_deleted = deleted - reference_counter
    delete_precision = 1.0
    if deleted:
        delete_precision = (
            sum(
                good_deleted[ngram] / deleted[ngram]
                for ngram in good_deleted
            )
            / len(deleted)
        )

    added = set(candidate_counter) - set(source_counter)
    good_added = added & set(reference_counter)
    all_added = set(reference_counter) - set(source_counter)
    add_precision = 1.0 if not added else len(good_added) / len(added)
    add_recall = 1.0 if not all_added else len(good_added) / len(all_added)
    add_score = 0.0
    if add_precision > 0.0 or add_recall > 0.0:
        add_score = (
            2.0
            * add_precision
            * add_recall
            / (add_precision + add_recall)
        )
    return keep_score, delete_precision, add_score


def _ngrams(tokens: Sequence[str], order: int) -> List[str]:
    if order == 1:
        return list(tokens)
    return [
        " ".join(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    ]


def _normalize_sari(text: str) -> str:
    try:
        from sacrebleu.metrics.bleu import _get_tokenizer
    except ImportError as error:
        raise RuntimeError(
            "TRACE 20Minuten evaluation requires sacrebleu==2.6.0"
        ) from error
    return _get_tokenizer("13a")()(text.lower())


def _sari_sentence(source: str, candidate: str, references: Sequence[str]) -> float:
    normalized_source = _normalize_sari(source).split(" ")
    normalized_candidate = _normalize_sari(candidate).split(" ")
    normalized_references = [
        _normalize_sari(reference).split(" ")
        for reference in references
    ]
    keep_scores = []
    delete_scores = []
    add_scores = []
    for order in range(1, 5):
        keep, delete, add = _sari_ngram(
            _ngrams(normalized_source, order),
            _ngrams(normalized_candidate, order),
            [
                _ngrams(reference, order)
                for reference in normalized_references
            ],
        )
        keep_scores.append(keep)
        delete_scores.append(delete)
        add_scores.append(add)
    return (
        sum(keep_scores) / 4.0
        + sum(delete_scores) / 4.0
        + sum(add_scores) / 4.0
    ) / 3.0


def _sari(
    sources: Sequence[str],
    predictions: Sequence[str],
    answers: Sequence[str],
) -> float:
    if not (
        len(sources) == len(predictions) == len(answers)
        and len(predictions) > 0
    ):
        raise ValueError("TRACE SARI inputs must have equal non-zero length")
    return 100.0 * sum(
        _sari_sentence(source, prediction, [answer])
        for source, prediction, answer in zip(
            sources,
            predictions,
            answers,
        )
    ) / len(predictions)


def evaluate_trace_rows(
    task: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if task not in TRACE_PRIMARY_METRICS:
        raise ValueError(f"Unknown TRACE task: {task}")
    if not rows:
        raise ValueError("TRACE task evaluation requires prediction rows")
    if [int(row["index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("TRACE task prediction order changed")
    predictions = [str(row["prediction"]) for row in rows]
    answers = [str(row["answer"]) for row in rows]
    sources = [str(row["prompt"]) for row in rows]
    if task in {"C-STANCE", "FOMC", "NumGLUE-cm", "NumGLUE-ds"}:
        raw_score = _accuracy(predictions, answers)
    elif task == "ScienceQA":
        raw_score = _scienceqa_accuracy(predictions, answers)
    elif task == "MeetingBank":
        raw_score = _rouge_l(predictions, answers)
    elif task == "Py150":
        raw_score = _py150_similarity(predictions, answers)
    elif task == "20Minuten":
        raw_score = _sari(sources, predictions, answers)
    else:
        raise AssertionError(f"Unhandled TRACE task: {task}")
    return {
        "task": task,
        "primary_metric": TRACE_PRIMARY_METRICS[task],
        "raw_score": raw_score,
        "normalized_score": normalize_trace_primary_score(task, raw_score),
        "rows": len(rows),
        "trace_evaluator_commit": TRACE_EVALUATOR_COMMIT,
        "sari_evaluate_commit": (
            SARI_EVALUATE_COMMIT if task == "20Minuten" else None
        ),
        "sari_source_blob": (
            SARI_SOURCE_BLOB if task == "20Minuten" else None
        ),
    }


def aggregate_trace_prediction_artifacts(
    paths: Sequence[str],
    *,
    task_order: Sequence[str] = TRACE_OFFICIAL_ORDER,
) -> Dict[str, Any]:
    order = list(task_order)
    expected = {
        (stage, task)
        for stage in range(1, len(order) + 1)
        for task in order[:stage]
    }
    artifacts: Dict[Tuple[int, str], Dict[str, Any]] = {}
    shared_identity = None
    records = []
    for path in paths:
        payload = read_trace_prediction_artifact(path)
        if payload["task_order"] != order:
            raise ValueError("TRACE prediction task order mismatch")
        key = (int(payload["stage"]), str(payload["task"]))
        if key in artifacts:
            raise ValueError(f"Duplicate TRACE prediction artifact: {key}")
        if shared_identity is None:
            shared_identity = payload["identity"]
        elif payload["identity"] != shared_identity:
            raise ValueError("TRACE prediction run identity changed")
        artifacts[key] = payload
        records.append(
            {
                "path": str(Path(path).resolve()),
                "stage": key[0],
                "task": key[1],
                "rows_sha256": payload["rows_sha256"],
            }
        )
    if set(artifacts) != expected:
        missing = sorted(expected - set(artifacts))
        extra = sorted(set(artifacts) - expected)
        raise ValueError(
            f"TRACE triangular artifact set mismatch: missing={missing}, extra={extra}"
        )

    matrix = []
    task_metrics = []
    for stage in range(1, len(order) + 1):
        row = []
        for task in order[:stage]:
            result = evaluate_trace_rows(
                task,
                artifacts[(stage, task)]["rows"],
            )
            row.append(result["normalized_score"])
            task_metrics.append({"stage": stage, **result})
        matrix.append(row)
    return {
        "format_version": TRACE_METRICS_FORMAT_VERSION,
        "identity": shared_identity,
        "task_order": order,
        "artifacts": sorted(
            records,
            key=lambda record: (record["stage"], order.index(record["task"])),
        ),
        "task_metrics": task_metrics,
        "summary": aggregate_trace_matrix(matrix),
    }


def write_trace_metrics(
    output_path: str,
    result: Mapping[str, Any],
) -> str:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return str(destination.resolve())
