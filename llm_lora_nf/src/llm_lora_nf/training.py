import random
import time
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
