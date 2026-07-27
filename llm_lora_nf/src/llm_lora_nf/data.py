from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


IGNORE_INDEX = -100
OFFICIAL_LORA_NULL_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


@dataclass(frozen=True)
class ChatExample:
    instruction: str
    response: str
    input_text: str = ""
    system: Optional[str] = None

    def user_content(self) -> str:
        if not self.input_text:
            return self.instruction
        return f"{self.instruction}\n\nInput:\n{self.input_text}"


def _template_kwargs(enable_thinking: bool) -> Dict[str, Any]:
    return {"enable_thinking": bool(enable_thinking)}


def _messages(example: ChatExample, include_response: bool) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if example.system:
        messages.append({"role": "system", "content": example.system})
    messages.append({"role": "user", "content": example.user_content()})
    if include_response:
        messages.append({"role": "assistant", "content": example.response})
    return messages


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def encode_chat_example(
    example: ChatExample,
    tokenizer: Any,
    *,
    max_length: int,
    enable_thinking: bool = False,
    truncation_strategy: str = "right",
) -> Dict[str, torch.Tensor]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if truncation_strategy not in {"right", "left_preserve_response"}:
        raise ValueError(
            "truncation_strategy must be 'right' or "
            "'left_preserve_response'"
        )
    source_text = tokenizer.apply_chat_template(
        _messages(example, include_response=False),
        tokenize=False,
        add_generation_prompt=True,
        **_template_kwargs(enable_thinking),
    )
    full_text = tokenizer.apply_chat_template(
        _messages(example, include_response=True),
        tokenize=False,
        add_generation_prompt=False,
        **_template_kwargs(enable_thinking),
    )
    if truncation_strategy == "left_preserve_response":
        source_ids = tokenizer(
            source_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        untruncated_full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        boundary = _longest_common_prefix(source_ids, untruncated_full_ids)
        window_start = max(0, len(untruncated_full_ids) - max_length)
        full_ids = list(untruncated_full_ids[window_start:])
        labels = [
            IGNORE_INDEX if window_start + index < boundary else token_id
            for index, token_id in enumerate(full_ids)
        ]
    else:
        source_ids = tokenizer(
            source_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        boundary = _longest_common_prefix(source_ids, full_ids)
        labels = [IGNORE_INDEX] * boundary + list(full_ids[boundary:])
        labels = labels[: len(full_ids)]
    if not labels or all(label == IGNORE_INDEX for label in labels):
        raise ValueError(
            "The response was fully truncated; increase max_length or shorten the prompt"
        )
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def encode_official_lora_null_math_example(
    example: ChatExample,
    tokenizer: Any,
    *,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Reproduce the official LoRA-Null MetaMathQA training formatter.

    Track A intentionally uses the repository's Alpaca-style prompt instead of
    an Instruct checkpoint's chat template. The source and source+target are
    tokenized independently with the tokenizer defaults, matching the official
    ``train_model.py`` preprocessing path.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if example.input_text or example.system:
        raise ValueError(
            "The official LoRA-Null math protocol accepts query/response only"
        )
    if tokenizer.eos_token is None:
        raise ValueError("The official LoRA-Null formatter requires an EOS token")
    source = OFFICIAL_LORA_NULL_PROMPT.format(
        instruction=example.instruction
    )
    target = f"{example.response}{tokenizer.eos_token}"
    source_ids = tokenizer(
        source,
        padding="longest",
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    full_ids = tokenizer(
        source + target,
        padding="longest",
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    source_length = sum(
        int(token_id != tokenizer.pad_token_id) for token_id in source_ids
    )
    labels = list(full_ids)
    labels[: min(source_length, len(labels))] = [IGNORE_INDEX] * min(
        source_length, len(labels)
    )
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SupervisedChatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: Iterable[ChatExample],
        tokenizer: Any,
        *,
        max_length: int,
        enable_thinking: bool = False,
        truncation_strategy: str = "right",
    ) -> None:
        self.rows = [
            encode_chat_example(
                example,
                tokenizer,
                max_length=max_length,
                enable_thinking=enable_thinking,
                truncation_strategy=truncation_strategy,
            )
            for example in examples
        ]
        if not self.rows:
            raise ValueError("At least one chat example is required")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.rows[index]


class LazySupervisedChatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: Sequence[ChatExample],
        tokenizer: Any,
        *,
        max_length: int,
        enable_thinking: bool = False,
        truncation_strategy: str = "right",
    ) -> None:
        if not examples:
            raise ValueError("At least one chat example is required")
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.enable_thinking = bool(enable_thinking)
        self.truncation_strategy = str(truncation_strategy)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return encode_chat_example(
            self.examples[index],
            self.tokenizer,
            max_length=self.max_length,
            enable_thinking=self.enable_thinking,
            truncation_strategy=self.truncation_strategy,
        )


class OfficialLoRANullMathDataset(torch.utils.data.Dataset):
    """Lazy MetaMathQA dataset for the isolated Track A reproduction."""

    def __init__(
        self,
        examples: Sequence[ChatExample],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        if not examples:
            raise ValueError("At least one chat example is required")
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return encode_official_lora_null_math_example(
            self.examples[index],
            self.tokenizer,
            max_length=self.max_length,
        )


class CalibrationQuestionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        questions: Sequence[str],
        tokenizer: Any,
        *,
        max_length: int,
        enable_thinking: bool = False,
    ) -> None:
        if not questions:
            raise ValueError("At least one calibration question is required")
        self.rows = []
        for question in questions:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
                **_template_kwargs(enable_thinking),
            )
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.rows.append(
                {
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.rows[index]


class CalibrationCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(
        self,
        features: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        response_collator = ResponseOnlyCollator(self.pad_token_id)
        compatible = [
            {
                **row,
                "labels": torch.full_like(row["input_ids"], IGNORE_INDEX),
            }
            for row in features
        ]
        batch = response_collator(compatible)
        del batch["labels"]
        return batch


class ResponseOnlyCollator:
    def __init__(
        self,
        pad_token_id: int,
        *,
        padding_side: str = "right",
    ) -> None:
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")
        self.pad_token_id = int(pad_token_id)
        self.padding_side = padding_side

    def __call__(
        self,
        features: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(int(row["input_ids"].numel()) for row in features)
        batch_size = len(features)
        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full(
            (batch_size, max_length),
            IGNORE_INDEX,
            dtype=torch.long,
        )
        for row_index, row in enumerate(features):
            length = int(row["input_ids"].numel())
            destination = (
                slice(max_length - length, max_length)
                if self.padding_side == "left"
                else slice(0, length)
            )
            input_ids[row_index, destination] = row["input_ids"]
            attention_mask[row_index, destination] = row["attention_mask"]
            labels[row_index, destination] = row["labels"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def synthetic_smoke_examples(count: int) -> List[ChatExample]:
    pool = [
        ChatExample("Compute 7 + 5. Return only the number.", "12"),
        ChatExample("Name the capital of France.", "Paris."),
        ChatExample("Complete the sequence: 2, 4, 6, 8, ...", "10"),
        ChatExample("Reply with the lowercase form of HELLO.", "hello"),
    ]
    if count <= 0:
        raise ValueError("count must be positive")
    return [pool[index % len(pool)] for index in range(count)]
