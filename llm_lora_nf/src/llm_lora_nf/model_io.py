from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


QWEN3_POST_TRAINED_IDS = {
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
}


@dataclass(frozen=True)
class ModelLoadRecord:
    requested_id: str
    resolved_path: str
    checkpoint_type: str
    torch_dtype: str
    local_files_only: bool
    tokenizer_class: str
    model_class: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assert_instruct_checkpoint(
    requested_id: str,
    tokenizer: Any,
    *,
    checkpoint_type: str,
) -> None:
    if checkpoint_type.lower() != "instruct":
        raise ValueError("This branch accepts Instruct/post-trained checkpoints only")
    normalized_id = requested_id.rstrip("/")
    recognized_name = (
        "instruct" in normalized_id.lower()
        or "chat" in normalized_id.lower()
        or normalized_id in QWEN3_POST_TRAINED_IDS
        or Path(normalized_id).name
        in {model_id.rsplit("/", 1)[-1] for model_id in QWEN3_POST_TRAINED_IDS}
    )
    if not recognized_name:
        raise ValueError(
            f"Checkpoint {requested_id!r} is not an approved Instruct/post-trained ID"
        )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Instruct checkpoint tokenizer must provide a chat template")


def load_instruct_model(
    *,
    requested_id: str,
    resolved_path: str,
    checkpoint_type: str = "instruct",
    device: str = "cpu",
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any, ModelLoadRecord]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cpu":
        load_dtype = torch.float32 if torch_dtype is None else torch_dtype
    else:
        load_dtype = torch.bfloat16 if torch_dtype is None else torch_dtype

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    assert_instruct_checkpoint(
        requested_id,
        tokenizer,
        checkpoint_type=checkpoint_type,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        dtype=load_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.to(torch.device(device))
    record = ModelLoadRecord(
        requested_id=requested_id,
        resolved_path=str(Path(resolved_path).resolve()),
        checkpoint_type=checkpoint_type,
        torch_dtype=str(load_dtype),
        local_files_only=local_files_only,
        tokenizer_class=type(tokenizer).__name__,
        model_class=type(model).__name__,
    )
    return model, tokenizer, record
