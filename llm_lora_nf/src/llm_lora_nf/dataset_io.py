import json
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .data import ChatExample


METAMATH_REVISION = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
NQ_OPEN_REVISION = "5dd9790a83002ad084ddeb7c420dc716852c6f28"
PISSA_DATASET_ID = "fxmeng/pissa-dataset"
PISSA_DATASET_REVISION = "d4746ceca8314940af8a61333bc2d395d9e259c9"
PISSA_CODEFEEDBACK_PYTHON_FILE = "python/train.json"
PISSA_CODEFEEDBACK_PYTHON_ROWS = 104848
PISSA_CODEFEEDBACK_PYTHON_SIZE_BYTES = 244222081
PISSA_CODEFEEDBACK_PYTHON_SHA256 = (
    "2fc75475ecb65aa9fa7a0f7135e4c0b8c59cfd52ef27a9ea0040ef143b891a82"
)
PISSA_CODEFEEDBACK_PYTHON_EMPTY_OUTPUT_INDICES = (25233,)
CODEFEEDBACK_SOURCE_ID = "m-a-p/CodeFeedback-Filtered-Instruction"
CODEFEEDBACK_SOURCE_REVISION = "a08c213a9748c66c15d0225814be80a2e77adf4a"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_pissa_codefeedback_python_examples(
    path: str,
    *,
    first_n: int,
    expected_sha256: Optional[str] = None,
    expected_size_bytes: Optional[int] = None,
    expected_total_rows: Optional[int] = None,
    allowed_empty_output_indices: Sequence[int] = (),
) -> List[ChatExample]:
    """Load the public PiSSA Python-only CodeFeedback training split.

    Formal callers provide every expected value. Tests and CPU-only fixtures may
    omit them so the same schema parser can be exercised on a small file.
    """

    if first_n <= 0:
        raise ValueError("first_n must be positive")
    source = Path(path)
    if expected_size_bytes is not None:
        actual_size = source.stat().st_size
        if actual_size != expected_size_bytes:
            raise ValueError(
                "PiSSA CodeFeedback Python file size mismatch: "
                f"expected {expected_size_bytes}, got {actual_size}"
            )
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(str(source))
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "PiSSA CodeFeedback Python SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
    with source.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("PiSSA CodeFeedback Python file must contain a JSON list")
    if expected_total_rows is not None and len(rows) != expected_total_rows:
        raise ValueError(
            "PiSSA CodeFeedback Python row-count mismatch: "
            f"expected {expected_total_rows}, got {len(rows)}"
        )
    if len(rows) < first_n:
        raise ValueError(f"Requested {first_n} rows but only {len(rows)} are available")
    examples = []
    observed_empty_output_indices = []
    allowed_empty = set(int(index) for index in allowed_empty_output_indices)
    for index, row in enumerate(rows[:first_n]):
        if not isinstance(row, dict) or "instruction" not in row or "output" not in row:
            raise ValueError(
                f"PiSSA CodeFeedback Python row {index} lacks instruction/output fields"
            )
        instruction = str(row["instruction"])
        response = str(row["output"])
        if not instruction.strip():
            raise ValueError(
                f"PiSSA CodeFeedback Python row {index} has an empty instruction"
            )
        if not response.strip():
            if index not in allowed_empty:
                raise ValueError(
                    f"PiSSA CodeFeedback Python row {index} has an empty output"
                )
            observed_empty_output_indices.append(index)
        examples.append(ChatExample(instruction=instruction, response=response))
    expected_empty = sorted(index for index in allowed_empty if index < first_n)
    if observed_empty_output_indices != expected_empty:
        raise ValueError(
            "PiSSA CodeFeedback Python empty-output identity mismatch: "
            f"expected {expected_empty}, got {observed_empty_output_indices}"
        )
    return examples


def load_training_examples(
    dataset: str,
    path: str,
    *,
    first_n: int,
    formal: bool,
) -> List[ChatExample]:
    if dataset == "metamathqa":
        return load_metamath_examples(path, first_n=first_n)
    if dataset == "pissa_codefeedback_python":
        return load_pissa_codefeedback_python_examples(
            path,
            first_n=first_n,
            expected_sha256=(
                PISSA_CODEFEEDBACK_PYTHON_SHA256 if formal else None
            ),
            expected_size_bytes=(
                PISSA_CODEFEEDBACK_PYTHON_SIZE_BYTES if formal else None
            ),
            expected_total_rows=(
                PISSA_CODEFEEDBACK_PYTHON_ROWS if formal else None
            ),
            allowed_empty_output_indices=(
                PISSA_CODEFEEDBACK_PYTHON_EMPTY_OUTPUT_INDICES
                if formal
                else ()
            ),
        )
    raise ValueError(f"Unsupported training dataset: {dataset}")


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
