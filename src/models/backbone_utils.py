"""Small compatibility layer for CLIP-family dual encoders.

The formal CLIP experiments use ``CLIPModel``.  SigLIP 2 checkpoints use the
same high-level dual-encoder API but do not expose CLIP's separate
``text_projection`` module (``get_text_features`` already returns projected
features).  Keeping that distinction here prevents model-family conditionals
from leaking into the trainer, classifier, and retrieval evaluator.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch


def is_siglip2_model_name(model_name: str | None) -> bool:
    """Return True if the checkpoint name/path indicates SigLIP 2."""
    return bool(model_name and "siglip2" in str(model_name).lower())


def is_siglip_family(model: Any = None, model_name: str | None = None) -> bool:
    """Return True if the checkpoint/backbone belongs to the SigLIP family."""
    if is_siglip2_model_name(model_name):
        return True
    if model_name is not None and "siglip" in str(model_name).lower():
        return True
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) in {"siglip", "siglip2"}:
        return True
    return "siglip" in type(model).__name__.lower()


def text_tokenize_kwargs(model: Any = None, model_name: str | None = None) -> dict:
    """Return the checkpoint-compatible text preprocessing arguments.

    SigLIP 2 was pretrained with fixed length-64 text sequences.  The Hugging
    Face processor supplies lowercasing itself, while this function keeps
    padding/truncation consistent across training and retrieval.
    """
    if is_siglip_family(model=model, model_name=model_name):
        return {"padding": "max_length", "truncation": True, "max_length": 64}
    return {"padding": True, "truncation": True}


def tokenize_texts(processor, texts: str | Iterable[str], model: Any = None,
                   model_name: str | None = None):
    return processor(text=texts, return_tensors="pt",
                     **text_tokenize_kwargs(model=model, model_name=model_name))


def _pooled_features(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    for field in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(outputs, field, None)
        if value is not None:
            return value
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state[:, -1, :]
    if isinstance(outputs, (tuple, list)):
        for value in reversed(outputs):
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                return value
    raise TypeError(f"Cannot obtain pooled features from {type(outputs)!r}")


def encode_image_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Encode images into the common image/text embedding space."""
    try:
        outputs = model.get_image_features(pixel_values=pixel_values)
    except TypeError:
        outputs = model.get_image_features(pixel_values)
    return _pooled_features(outputs)


def encode_text_features(model, text_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Encode tokenized text into the common image/text embedding space."""
    return _pooled_features(model.get_text_features(**text_inputs))


def embedding_dim(model) -> int:
    config = getattr(model, "config", None)
    projection_dim = getattr(config, "projection_dim", None)
    if projection_dim is not None:
        return int(projection_dim)
    text_config = getattr(config, "text_config", None)
    for field in ("projection_size", "hidden_size"):
        value = getattr(text_config, field, None)
        if value is not None:
            return int(value)
    text_head = getattr(getattr(model, "text_model", None), "head", None)
    if getattr(text_head, "out_features", None) is not None:
        return int(text_head.out_features)
    raise AttributeError("Cannot infer embedding dimension for this backbone")
