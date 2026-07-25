from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


IGNORE_INDEX = -100


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
) -> Dict[str, torch.Tensor]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
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


class SupervisedChatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: Iterable[ChatExample],
        tokenizer: Any,
        *,
        max_length: int,
        enable_thinking: bool = False,
    ) -> None:
        self.rows = [
            encode_chat_example(
                example,
                tokenizer,
                max_length=max_length,
                enable_thinking=enable_thinking,
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
    ) -> None:
        if not examples:
            raise ValueError("At least one chat example is required")
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.enable_thinking = bool(enable_thinking)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return encode_chat_example(
            self.examples[index],
            self.tokenizer,
            max_length=self.max_length,
            enable_thinking=self.enable_thinking,
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
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

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
            input_ids[row_index, :length] = row["input_ids"]
            attention_mask[row_index, :length] = row["attention_mask"]
            labels[row_index, :length] = row["labels"]
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
