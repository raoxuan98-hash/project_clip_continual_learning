import torch

from llm_lora_nf.data import (
    IGNORE_INDEX,
    ChatExample,
    OFFICIAL_LORA_NULL_PROMPT,
    ResponseOnlyCollator,
    encode_chat_example,
    encode_official_lora_null_math_example,
)


class CharacterTokenizer:
    chat_template = "test-template"
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert not tokenize
        assert not enable_thinking
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=True,
        max_length=None,
    ):
        assert not add_special_tokens
        input_ids = [ord(character) for character in text]
        if truncation:
            input_ids = input_ids[:max_length]
        return {"input_ids": input_ids}


def test_response_only_labels_mask_prompt():
    row = encode_chat_example(
        ChatExample("Question", "Answer"),
        CharacterTokenizer(),
        max_length=128,
    )
    labels = row["labels"]
    supervised = labels[labels != IGNORE_INDEX]
    assert supervised.numel() > 0
    decoded = "".join(chr(value) for value in supervised.tolist())
    assert decoded.startswith("Answer")


def test_collator_masks_padding_in_labels():
    tokenizer = CharacterTokenizer()
    short = encode_chat_example(
        ChatExample("Q", "A"),
        tokenizer,
        max_length=128,
    )
    long = encode_chat_example(
        ChatExample("Longer question", "Longer answer"),
        tokenizer,
        max_length=128,
    )
    batch = ResponseOnlyCollator(tokenizer.pad_token_id)([short, long])
    short_length = short["input_ids"].numel()
    assert torch.all(batch["attention_mask"][0, short_length:] == 0)
    assert torch.all(batch["labels"][0, short_length:] == IGNORE_INDEX)


def test_trace_style_left_truncation_preserves_response_tokens():
    row = encode_chat_example(
        ChatExample("P" * 200, "Answer"),
        CharacterTokenizer(),
        max_length=32,
        truncation_strategy="left_preserve_response",
    )
    assert row["input_ids"].numel() == 32
    supervised = row["labels"][row["labels"] != IGNORE_INDEX]
    assert "".join(chr(value) for value in supervised.tolist()) == "Answer"


def test_left_padding_keeps_labels_aligned_at_batch_end():
    tokenizer = CharacterTokenizer()
    short = encode_chat_example(
        ChatExample("Q", "A"),
        tokenizer,
        max_length=128,
    )
    long = encode_chat_example(
        ChatExample("Longer question", "Longer answer"),
        tokenizer,
        max_length=128,
    )
    batch = ResponseOnlyCollator(
        tokenizer.pad_token_id,
        padding_side="left",
    )([short, long])
    short_length = short["input_ids"].numel()
    pad_length = long["input_ids"].numel() - short_length
    assert torch.all(batch["attention_mask"][0, :pad_length] == 0)
    assert torch.equal(
        batch["labels"][0, pad_length:],
        short["labels"],
    )


class OfficialCharacterTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, *, padding, truncation, max_length):
        assert padding == "longest"
        assert truncation
        return {"input_ids": [ord(character) for character in text[:max_length]]}


def test_official_lora_null_formatter_uses_repository_prompt():
    tokenizer = OfficialCharacterTokenizer()
    row = encode_official_lora_null_math_example(
        ChatExample("Compute 1+1.", "2"),
        tokenizer,
        max_length=512,
    )
    source = OFFICIAL_LORA_NULL_PROMPT.format(instruction="Compute 1+1.")
    assert row["input_ids"][: len(source)].tolist() == [
        ord(character) for character in source
    ]
    assert torch.all(row["labels"][: len(source)] == IGNORE_INDEX)
    supervised = "".join(
        chr(value)
        for value in row["labels"][row["labels"] != IGNORE_INDEX].tolist()
    )
    assert supervised == "2<eos>"
