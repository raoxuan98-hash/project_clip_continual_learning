import torch

from llm_lora_nf.data import (
    IGNORE_INDEX,
    ChatExample,
    ResponseOnlyCollator,
    encode_chat_example,
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
        assert truncation
        return {"input_ids": [ord(character) for character in text[:max_length]]}


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
