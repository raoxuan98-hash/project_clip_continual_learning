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
    started = time.monotonic()
    optimizer_steps = 0
    micro_step = 0
    while optimizer_steps < max_steps:
        saw_batch = False
        for batch in batches:
            saw_batch = True
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(**moved)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            losses.append(float(outputs.loss.detach().cpu().item()))
            micro_step += 1
            if micro_step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps >= max_steps:
                    break
        if not saw_batch:
            raise ValueError("Training iterable produced no batches")
    elapsed = time.monotonic() - started
    return TrainingSummary(
        steps=optimizer_steps,
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
    max_steps: int = None,
) -> TrainingSummary:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not hasattr(batches, "__len__") or len(batches) <= 0:
        raise ValueError("batches must be a non-empty sized iterable")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must lie in [0, 1)")
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
    planned_steps = math.ceil(
        len(batches) * epochs / gradient_accumulation_steps
    )
    total_steps = planned_steps if max_steps is None else min(planned_steps, max_steps)
    if total_steps <= 0:
        raise ValueError("Training schedule has no optimizer steps")
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    use_bf16 = precision == "bf16" and device.type == "cuda"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: List[float] = []
    optimizer_steps = 0
    micro_step = 0
    started = time.monotonic()
    for _ in range(epochs):
        for batch_index, batch in enumerate(batches):
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                outputs = model(**moved)
                scaled_loss = outputs.loss / gradient_accumulation_steps
            scaled_loss.backward()
            losses.append(float(outputs.loss.detach().cpu().item()))
            micro_step += 1
            is_last_batch = batch_index + 1 == len(batches)
            should_step = (
                micro_step % gradient_accumulation_steps == 0 or is_last_batch
            )
            if should_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps >= total_steps:
                    break
        if optimizer_steps >= total_steps:
            break
    return TrainingSummary(
        steps=optimizer_steps,
        losses=losses,
        wall_seconds=time.monotonic() - started,
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
    )
