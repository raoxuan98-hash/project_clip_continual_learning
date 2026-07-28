import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from llm_lora_nf.training import train_epochs


class TinyLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask, labels):
        del attention_mask, labels
        loss = (self.weight * input_ids.float().mean()).square()
        return type("Output", (), {"loss": loss})()


class NonFiniteLossModel(TinyLossModel):
    def forward(self, input_ids, attention_mask, labels):
        output = super().forward(input_ids, attention_mask, labels)
        output.loss = output.loss * torch.tensor(float("nan"))
        return output


class _FiniteForwardNonFiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        del ctx
        return value.square()

    @staticmethod
    def backward(ctx, gradient):
        del ctx
        return torch.full_like(gradient, float("nan"))


class NonFiniteGradientModel(TinyLossModel):
    def forward(self, input_ids, attention_mask, labels):
        del attention_mask, labels
        value = self.weight * input_ids.float().mean()
        loss = _FiniteForwardNonFiniteBackward.apply(value)
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


def _shuffled_order(*, workers):
    generator = torch.Generator().manual_seed(42)
    return [
        int(batch.item())
        for batch in DataLoader(
            range(256),
            batch_size=1,
            shuffle=True,
            generator=generator,
            num_workers=workers,
        )
    ]


def test_dataloader_workers_preserve_seeded_sampler_order():
    assert _shuffled_order(workers=0) == _shuffled_order(workers=2)


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
    assert summary.supervised_tokens == 16
    assert summary.zero_supervision_micro_batches == 0
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


def test_all_ignored_labels_consume_schedule_without_nan():
    batches = _batches(2)
    batches[0]["labels"].fill_(-100)
    summary = train_epochs(
        TinyLossModel(),
        batches,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        gradient_accumulation_steps=2,
        precision="fp32",
        max_grad_norm=1.0,
    )
    assert summary.steps == 1
    assert summary.micro_steps == 2
    assert summary.examples == 2
    assert summary.supervised_tokens == 2
    assert summary.zero_supervision_micro_batches == 1
    assert len(summary.losses) == 1
    assert torch.isfinite(torch.tensor(summary.losses)).all()


def test_progress_interval_validation_rejects_negative_value():
    with pytest.raises(ValueError, match="progress_every_steps"):
        train_epochs(
            TinyLossModel(),
            _batches(2),
            device=torch.device("cpu"),
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_ratio=0.0,
            gradient_accumulation_steps=2,
            precision="fp32",
            progress_every_steps=-1,
        )


def test_progress_interval_emits_optimizer_budget(capsys):
    train_epochs(
        TinyLossModel(),
        _batches(2),
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        gradient_accumulation_steps=2,
        precision="fp32",
        progress_every_steps=1,
        progress_label="unit_test",
    )
    output = capsys.readouterr().out
    assert "training_progress run=unit_test steps=1/1" in output
    assert "micro_steps=2" in output


def test_non_finite_loss_fails_before_optimizer_step():
    with pytest.raises(FloatingPointError, match="Non-finite"):
        train_epochs(
            NonFiniteLossModel(),
            _batches(2),
            device=torch.device("cpu"),
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_ratio=0.0,
            gradient_accumulation_steps=2,
            precision="fp32",
            max_grad_norm=1.0,
        )


def test_non_finite_gradient_fails_before_optimizer_step():
    with pytest.raises(FloatingPointError, match="gradient norm"):
        train_epochs(
            NonFiniteGradientModel(),
            _batches(2),
            device=torch.device("cpu"),
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_ratio=0.0,
            gradient_accumulation_steps=2,
            precision="fp32",
            max_grad_norm=1.0,
        )
