import random
import time
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TrainingSummary:
    steps: int
    micro_steps: int
    examples: int
    non_padding_tokens: int
    tokens_per_second: float
    discarded_micro_batches: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    losses: List[float]
    wall_seconds: float
    trainable_parameters: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_steps(
    model: nn.Module,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    device: torch.device,
    max_steps: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    gradient_accumulation_steps: int = 1,
) -> TrainingSummary:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    model.train()
    losses: List[float] = []
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    optimizer_steps = 0
    micro_step = 0
    examples = 0
    non_padding_tokens = 0
    accumulated_loss = torch.zeros((), device=device, dtype=torch.float32)
    while optimizer_steps < max_steps:
        saw_batch = False
        for batch in batches:
            saw_batch = True
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            input_ids = batch.get("input_ids")
            if input_ids is not None:
                examples += int(input_ids.shape[0])
                attention_mask = batch.get("attention_mask")
                non_padding_tokens += int(
                    attention_mask.sum().item()
                    if attention_mask is not None
                    else input_ids.numel()
                )
            outputs = model(**moved)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            accumulated_loss.add_(outputs.loss.detach().float())
            micro_step += 1
            if micro_step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(
                    float(
                        (
                            accumulated_loss / gradient_accumulation_steps
                        ).item()
                    )
                )
                accumulated_loss.zero_()
                optimizer_steps += 1
                if optimizer_steps >= max_steps:
                    break
        if not saw_batch:
            raise ValueError("Training iterable produced no batches")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    return TrainingSummary(
        steps=optimizer_steps,
        micro_steps=micro_step,
        examples=examples,
        non_padding_tokens=non_padding_tokens,
        tokens_per_second=non_padding_tokens / elapsed,
        discarded_micro_batches=0,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=0.0,
        losses=losses,
        wall_seconds=elapsed,
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
    )


def train_epochs(
    model: nn.Module,
    batches,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    precision: str,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    max_grad_norm: float = 1.0,
    max_steps: int = None,
) -> TrainingSummary:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not hasattr(batches, "__len__") or len(batches) <= 0:
        raise ValueError("batches must be a non-empty sized iterable")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must lie in [0, 1)")
    if not 0.0 <= adam_beta1 < 1.0 or not 0.0 <= adam_beta2 < 1.0:
        raise ValueError("Adam beta values must lie in [0, 1)")
    if adam_epsilon <= 0.0:
        raise ValueError("adam_epsilon must be positive")
    if max_grad_norm < 0.0:
        raise ValueError("max_grad_norm must be non-negative")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )
    # Transformers Trainer 4.47.0, used by the audited LoRA-Null protocol,
    # computes update steps with floor division. For 100,000 micro-batches and
    # accumulation 128 this is 781 steps; the final 32 examples are not used.
    updates_per_epoch = max(len(batches) // gradient_accumulation_steps, 1)
    effective_group_size = min(
        gradient_accumulation_steps,
        len(batches),
    )
    planned_steps = updates_per_epoch * epochs
    total_steps = planned_steps if max_steps is None else min(planned_steps, max_steps)
    if total_steps <= 0:
        raise ValueError("Training schedule has no optimizer steps")
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    use_bf16 = precision == "bf16" and device.type == "cuda"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: List[float] = []
    optimizer_steps = 0
    micro_step = 0
    examples = 0
    non_padding_tokens = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    accumulated_loss = torch.zeros((), device=device, dtype=torch.float32)
    for _ in range(epochs):
        for batch_index, batch in enumerate(batches):
            if (
                len(batches) >= gradient_accumulation_steps
                and batch_index >= updates_per_epoch * gradient_accumulation_steps
            ):
                break
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            input_ids = batch.get("input_ids")
            if input_ids is not None:
                examples += int(input_ids.shape[0])
                attention_mask = batch.get("attention_mask")
                non_padding_tokens += int(
                    attention_mask.sum().item()
                    if attention_mask is not None
                    else input_ids.numel()
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                outputs = model(**moved)
                scaled_loss = outputs.loss / effective_group_size
            scaled_loss.backward()
            accumulated_loss.add_(outputs.loss.detach().float())
            micro_step += 1
            should_step = (
                (batch_index + 1) % effective_group_size == 0
            )
            if should_step:
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(
                    float((accumulated_loss / effective_group_size).item())
                )
                accumulated_loss.zero_()
                optimizer_steps += 1
                if optimizer_steps >= total_steps:
                    break
        if optimizer_steps >= total_steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    total_available_micro_batches = len(batches) * epochs
    return TrainingSummary(
        steps=optimizer_steps,
        micro_steps=micro_step,
        examples=examples,
        non_padding_tokens=non_padding_tokens,
        tokens_per_second=non_padding_tokens / elapsed,
        discarded_micro_batches=total_available_micro_batches - micro_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        losses=losses,
        wall_seconds=elapsed,
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
    )
