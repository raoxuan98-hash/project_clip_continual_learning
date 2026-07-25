import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .data import ChatExample


METAMATH_REVISION = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
NQ_OPEN_REVISION = "5dd9790a83002ad084ddeb7c420dc716852c6f28"


def load_metamath_examples(path: str, *, first_n: int) -> List[ChatExample]:
    if first_n <= 0:
        raise ValueError("first_n must be positive")
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if len(rows) < first_n:
        raise ValueError(f"Requested {first_n} rows but only {len(rows)} are available")
    selected = rows[:first_n]
    examples = []
    for index, row in enumerate(selected):
        if "query" not in row or "response" not in row:
            raise ValueError(f"MetaMath row {index} lacks query/response fields")
        examples.append(
            ChatExample(
                instruction=str(row["query"]),
                response=str(row["response"]),
            )
        )
    return examples


def load_nq_open_rows(parquet_path: str) -> Sequence[Dict[str, Any]]:
    from datasets import load_dataset

    return load_dataset(
        "parquet",
        data_files=str(Path(parquet_path)),
        split="train",
    )


def sample_nq_questions(
    parquet_path: str,
    *,
    samples: int,
    seed: int,
) -> List[str]:
    rows = load_nq_open_rows(parquet_path)
    if samples <= 0 or samples > len(rows):
        raise ValueError(f"samples must lie in [1, {len(rows)}]")
    indices = random.Random(seed).sample(range(len(rows)), samples)
    return [str(rows[index]["question"]) for index in indices]


def official_lora_null_nq_character_spans(
    parquet_path: str,
    tokenizer: Any,
    *,
    samples: int = 256,
    seed: int = 233,
    sequence_length: int = 2048,
) -> List[Dict[str, Any]]:
    """Reproduce the LoRA-Null repository's raw NQ character-span sampler."""

    import torch

    rows = load_nq_open_rows(parquet_path)
    text = "\n\n".join(str(question) for question in rows["question"])
    generator = random.Random(seed)
    output = []
    for _ in range(samples):
        start = generator.randint(0, len(text) - sequence_length - 1)
        fragment = text[start : start + sequence_length * 10]
        encoded = tokenizer(fragment, return_tensors="pt")
        input_ids = encoded["input_ids"][:, :sequence_length]
        output.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )
    return output
