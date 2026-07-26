import torch
import torch.nn as nn

from llm_lora_nf.training import train_epochs


class TinyLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask, labels):
        del attention_mask, labels
        loss = (self.weight * input_ids.float().mean()).square()
        return type("Output", (), {"loss": loss})()


def _batches(count):
    return [
        {
            "input_ids": torch.tensor([[index + 1, index + 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "labels": torch.tensor([[index + 1, index + 2]]),
        }
        for index in range(count)
    ]


def test_formal_style_accumulation_drops_incomplete_remainder():
    summary = train_epochs(
        TinyLossModel(),
        _batches(10),
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        gradient_accumulation_steps=4,
        precision="fp32",
        max_grad_norm=1.0,
    )
    assert summary.steps == 2
    assert summary.micro_steps == 8
    assert summary.examples == 8
    assert summary.non_padding_tokens == 16
    assert summary.discarded_micro_batches == 2
    assert len(summary.losses) == 2
    assert summary.tokens_per_second > 0.0


def test_tiny_smoke_batch_uses_one_complete_effective_group():
    summary = train_epochs(
        TinyLossModel(),
        _batches(2),
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        gradient_accumulation_steps=128,
        precision="fp32",
        max_grad_norm=1.0,
    )
    assert summary.steps == 1
    assert summary.micro_steps == 2
    assert summary.examples == 2
    assert summary.discarded_micro_batches == 0
