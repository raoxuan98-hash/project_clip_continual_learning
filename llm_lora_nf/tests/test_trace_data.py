import copy

import torch
import torch.nn as nn

from llm_lora_nf.trace_data import (
    TraceExample,
    generate_trace_predictions,
    read_trace_prediction_artifact,
    trace_chat_examples,
    write_trace_prediction_artifact,
)
from llm_lora_nf.trace_protocol import TRACE_OFFICIAL_ORDER


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self):
        self.padding_side = "right"
        self.truncation_side = "right"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return f"user:{messages[0]['content']}\nassistant:"

    def __call__(
        self,
        texts,
        *,
        add_special_tokens,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        assert add_special_tokens is False
        assert padding is True
        assert truncation is True
        assert return_tensors == "pt"
        encoded = [
            [3 + (ord(character) % 20) for character in text][-max_length:]
            for text in texts
        ]
        width = max(len(row) for row in encoded)
        padded = []
        masks = []
        for row in encoded:
            padding_length = width - len(row)
            padded.append([0] * padding_length + row)
            masks.append([0] * padding_length + [1] * len(row))
        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(masks),
        }

    def batch_decode(
        self,
        rows,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return ["answer" if row.tolist() == [7, 8] else "wrong" for row in rows]


class DummyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.arguments = None

    def generate(self, *, input_ids, attention_mask, **kwargs):
        self.arguments = {
            "attention_mask": attention_mask.detach().clone(),
            **copy.deepcopy(kwargs),
        }
        suffix = torch.tensor(
            [[7, 8]] * input_ids.shape[0],
            device=input_ids.device,
        )
        return torch.cat([input_ids, suffix], dim=1)


def test_trace_generation_is_greedy_ordered_and_restores_tokenizer_state():
    tokenizer = DummyTokenizer()
    model = DummyGenerator()
    model.train()
    rows = [
        TraceExample(index=0, prompt="first", answer="A"),
        TraceExample(index=1, prompt="second", answer="B"),
    ]
    predictions = generate_trace_predictions(
        model,
        tokenizer,
        rows,
        device=torch.device("cpu"),
        batch_size=2,
        max_prompt_length=64,
        max_new_tokens=8,
    )
    assert [row["prediction"] for row in predictions] == ["answer", "answer"]
    assert [row["index"] for row in predictions] == [0, 1]
    assert model.arguments["do_sample"] is False
    assert model.arguments["num_beams"] == 1
    assert tokenizer.padding_side == "right"
    assert tokenizer.truncation_side == "right"
    assert model.training is True


def test_trace_prediction_artifact_roundtrip_binds_content(tmp_path):
    predictions = [
        {
            "index": 0,
            "prompt": "question",
            "prediction": "answer",
            "answer": "answer",
        }
    ]
    record = write_trace_prediction_artifact(
        str(tmp_path / "stage_01_C-STANCE.json.gz"),
        task="C-STANCE",
        stage=1,
        task_order=TRACE_OFFICIAL_ORDER,
        predictions=predictions,
        generation={"do_sample": False},
        identity={"seed": 42},
    )
    payload = read_trace_prediction_artifact(record["path"])
    assert payload["rows"] == predictions
    assert record["rows_sha256"] == payload["rows_sha256"]
    assert len(record["sha256"]) == 64


def test_trace_rows_convert_to_instruct_chat_examples():
    converted = trace_chat_examples(
        [TraceExample(index=0, prompt="prompt", answer="response")]
    )
    assert converted[0].instruction == "prompt"
    assert converted[0].response == "response"
