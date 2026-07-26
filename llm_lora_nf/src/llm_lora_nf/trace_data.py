import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from .data import ChatExample
from .trace_protocol import TRACE_OFFICIAL_ORDER


TRACE_PREDICTION_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TraceExample:
    index: int
    prompt: str
    answer: str


def load_trace_split(
    root: str,
    *,
    task: str,
    split: str,
) -> List[TraceExample]:
    if task not in TRACE_OFFICIAL_ORDER:
        raise ValueError(f"Unknown TRACE task: {task}")
    if split not in {"train", "eval", "test"}:
        raise ValueError(f"Unknown TRACE split: {split}")
    path = Path(root).resolve() / task / f"{split}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"TRACE split must be a non-empty JSON array: {path}")
    examples = []
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"prompt", "answer"}
            or not isinstance(row["prompt"], str)
            or not isinstance(row["answer"], str)
        ):
            raise ValueError(f"Invalid TRACE row: {path}:{index}")
        examples.append(
            TraceExample(
                index=index,
                prompt=row["prompt"],
                answer=row["answer"],
            )
        )
    return examples


def trace_chat_examples(
    rows: Iterable[TraceExample],
) -> List[ChatExample]:
    return [
        ChatExample(instruction=row.prompt, response=row.answer)
        for row in rows
    ]


def _chat_generation_text(
    tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool,
) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )


@torch.no_grad()
def generate_trace_predictions(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[TraceExample],
    *,
    device: torch.device,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    enable_thinking: bool = False,
) -> List[Dict[str, Any]]:
    if not rows:
        raise ValueError("TRACE generation requires at least one row")
    if batch_size <= 0 or max_prompt_length <= 0 or max_new_tokens <= 0:
        raise ValueError("TRACE generation sizes must be positive")
    if tokenizer.pad_token_id is None:
        raise ValueError("TRACE generation requires tokenizer.pad_token_id")

    was_training = model.training
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model.eval()
    predictions: List[Dict[str, Any]] = []
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            texts = [
                _chat_generation_text(
                    tokenizer,
                    row.prompt,
                    enable_thinking=enable_thinking,
                )
                for row in batch_rows
            ]
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = generated[:, input_ids.shape[1] :]
            decoded = tokenizer.batch_decode(
                continuation,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(decoded) != len(batch_rows):
                raise RuntimeError("TRACE generation batch cardinality changed")
            for row, prediction in zip(batch_rows, decoded):
                predictions.append(
                    {
                        "index": row.index,
                        "prompt": row.prompt,
                        "prediction": prediction.strip(),
                        "answer": row.answer,
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side
        model.train(was_training)
    if [row["index"] for row in predictions] != [
        row.index for row in rows
    ]:
        raise RuntimeError("TRACE prediction order changed")
    return predictions


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_trace_prediction_artifact(
    output_path: str,
    *,
    task: str,
    stage: int,
    task_order: Sequence[str],
    predictions: Sequence[Mapping[str, Any]],
    generation: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> Dict[str, Any]:
    if task not in task_order:
        raise ValueError("TRACE prediction task is not in the task order")
    if stage <= 0 or stage > len(task_order):
        raise ValueError("TRACE prediction stage lies outside the task order")
    expected_indices = list(range(len(predictions)))
    actual_indices = [int(row["index"]) for row in predictions]
    if actual_indices != expected_indices:
        raise ValueError("TRACE prediction rows must preserve source order")
    payload = {
        "format_version": TRACE_PREDICTION_FORMAT_VERSION,
        "task": task,
        "stage": int(stage),
        "task_order": list(task_order),
        "generation": dict(generation),
        "identity": dict(identity),
        "rows_sha256": _canonical_hash(list(predictions)),
        "rows": [dict(row) for row in predictions],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    encoded = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            compressed.write(encoded)
    temporary.replace(destination)
    return {
        "path": str(destination.resolve()),
        "size_bytes": int(destination.stat().st_size),
        "sha256": _sha256_file(destination),
        "rows": len(predictions),
        "rows_sha256": payload["rows_sha256"],
        "task": task,
        "stage": int(stage),
    }


def read_trace_prediction_artifact(
    path: str,
) -> Dict[str, Any]:
    source = Path(path)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format_version") != TRACE_PREDICTION_FORMAT_VERSION:
        raise ValueError(f"Unsupported TRACE prediction artifact: {source}")
    rows = payload.get("rows")
    if (
        not isinstance(rows, list)
        or payload.get("rows_sha256") != _canonical_hash(rows)
    ):
        raise ValueError(f"TRACE prediction content hash mismatch: {source}")
    if [int(row["index"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"TRACE prediction order mismatch: {source}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
