#!/usr/bin/env python3
"""Protocol-aligned Instruct reproduction of LoRA and LoRA-Null."""

import argparse
import copy
import gc
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from llm_lora_nf.config_io import (
    canonical_comparison_config_hash,
    canonical_config_hash,
    load_yaml_config,
)
from llm_lora_nf.data import OfficialLoRANullMathDataset, ResponseOnlyCollator
from llm_lora_nf.dataset_io import (
    load_metamath_examples,
    official_lora_null_nq_character_spans,
)
from llm_lora_nf.environment import runtime_environment
from llm_lora_nf.inject import count_parameters
from llm_lora_nf.integrity import (
    validate_directory_integrity,
    write_directory_integrity,
)
from llm_lora_nf.model_io import load_instruct_model
from llm_lora_nf.protocol_validation import validate_track_a_formal_config
from llm_lora_nf.resource_guard import inspect_admission, project_gpu_lock
from llm_lora_nf.result_metadata import RunMetadata
from llm_lora_nf.track_a import (
    TRACK_A_OFFICIAL_COMMIT,
    OfficialLoRANullCalibrator,
    TrackAAdapterConfig,
    build_track_a_native_adapter,
    build_track_a_peft_lora,
    save_track_a_native_adapter,
)
from llm_lora_nf.training import set_reproducible_seed, train_epochs


def _git_state() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "llm_lora_nf"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit_sha": commit, "source_dirty": dirty}


@torch.no_grad()
def _logits(model, batch, device):
    model.eval()
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key != "labels"
    }
    return model(**inputs).logits.detach().float().cpu()


def _run(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = load_yaml_config(args.config)
    if config.get("protocol_track") != "A":
        raise ValueError("Track A runner requires protocol_track: A")
    if args.method is not None:
        config["adapter"]["method"] = args.method
    if args.seed is not None:
        config["run"]["seed"] = args.seed
    if args.run_name is not None:
        config["run"]["name"] = args.run_name
    method = str(config["adapter"]["method"])
    if method not in {"lora", "lora_null"}:
        raise ValueError("Track A supports only LoRA and LoRA-Null")
    if "name" not in config["run"]:
        model_slug = config["model"]["id"].rsplit("/", 1)[-1].lower()
        config["run"]["name"] = (
            f"track_a_{model_slug}_{method}_seed{config['run']['seed']}"
        )

    config_hash = canonical_config_hash(config)
    protocol_config = copy.deepcopy(config)
    protocol_config["run"].pop("seed", None)
    protocol_config["run"].pop("name", None)
    protocol_config_hash = canonical_config_hash(protocol_config)
    comparison_config_hash = canonical_comparison_config_hash(config)
    git_state = _git_state()
    admission = inspect_admission(requested=1)
    requested_mode = config["run"]["execution_mode"]
    if requested_mode == "gpu_formal" and admission.mode == "gpu":
        validate_track_a_formal_config(config)
        if git_state["source_dirty"]:
            raise RuntimeError("Formal Track A training requires a clean worktree")
        if any(
            value is not None
            for value in (
                args.max_steps,
                args.train_first_n,
                args.calibration_samples,
            )
        ):
            raise ValueError(
                "Formal Track A training forbids max-step, row-count, and "
                "calibration overrides"
            )
        selected_gpu = admission.selected_gpu_indices[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        execution_mode = "gpu_formal"
        device = torch.device("cuda:0")
        # The official adapter-building stage explicitly loads FP16. Its LoRA
        # comparator is constructed directly in the training stage.
        load_dtype = torch.float16 if method == "lora_null" else torch.float32
    elif args.cpu_smoke_fallback:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        execution_mode = "cpu_smoke_only"
        device = torch.device("cpu")
        load_dtype = torch.float32
    else:
        raise RuntimeError(
            "No admissible GPU is available; --cpu-smoke-fallback may only "
            "validate the chain without collecting results"
        )
    environment_record = runtime_environment(
        device,
        validate_formal=execution_mode == "gpu_formal",
    )

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["run"]["seed"])
    set_reproducible_seed(seed)
    if device.type == "cuda":
        allow_tf32 = bool(config["train"].get("tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    total_started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_integrity_record = None
    model_integrity_seconds = 0.0
    if execution_mode == "gpu_formal":
        model_integrity_started = time.monotonic()
        model_integrity_record = validate_directory_integrity(
            args.model_path,
            expected_kind="model_snapshot",
        )
        model_integrity_seconds = time.monotonic() - model_integrity_started
    model, tokenizer, load_record = load_instruct_model(
        requested_id=config["model"]["id"],
        source_id=config["model"].get("source_id", config["model"]["id"]),
        resolved_path=args.model_path,
        checkpoint_type=config["model"]["checkpoint_type"],
        device=str(device),
        local_files_only=True,
        torch_dtype=load_dtype,
    )
    if (
        execution_mode == "gpu_formal"
        and (
            load_record.snapshot_manifest_sha256 is None
            or load_record.snapshot_integrity_sha256 is None
        )
    ):
        raise RuntimeError(
            "Formal Track A training requires hashed provenance and integrity "
            "manifests for the local model snapshot"
        )
    adapter_config = TrackAAdapterConfig(
        method=method,
        rank=int(config["adapter"]["rank"]),
        alpha=float(config["adapter"]["alpha"]),
        dropout=float(config["adapter"]["dropout"]),
        target_modules=tuple(config["adapter"]["target_modules"]),
    )
    train_config = config["train"]
    calibration_config = config["calibration"]
    if execution_mode == "cpu_smoke_only":
        train_first_n = args.train_first_n or 2
        calibration_samples = args.calibration_samples or 2
        calibration_length = min(
            int(calibration_config["max_sequence_length"]),
            64,
        )
        max_steps = args.max_steps or 1
    else:
        train_first_n = args.train_first_n or int(train_config["first_n"])
        calibration_samples = (
            args.calibration_samples or int(calibration_config["samples"])
        )
        calibration_length = int(calibration_config["max_sequence_length"])
        max_steps = args.max_steps
        if calibration_samples != 256 or calibration_length != 2048:
            raise ValueError(
                "Formal Track A calibration is locked to 256 samples x 2048 tokens"
            )

    data_started = time.monotonic()
    examples = load_metamath_examples(
        args.metamath_json,
        first_n=train_first_n,
    )
    training_dataset = OfficialLoRANullMathDataset(
        examples,
        tokenizer,
        max_length=int(train_config["max_sequence_length"]),
    )
    generator = torch.Generator().manual_seed(seed)
    training_collator = ResponseOnlyCollator(tokenizer.pad_token_id)
    training_batches = DataLoader(
        training_dataset,
        batch_size=int(train_config["per_device_batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=training_collator,
    )
    first_batch = training_collator([training_dataset[0]])
    data_seconds = time.monotonic() - data_started

    calibration_seconds = 0.0
    if method == "lora_null":
        calibration_started = time.monotonic()
        calibration_batches = official_lora_null_nq_character_spans(
            args.nq_parquet,
            tokenizer,
            samples=calibration_samples,
            seed=int(calibration_config["seed"]),
            sequence_length=calibration_length,
        )
        moments = OfficialLoRANullCalibrator(
            model,
            expected_samples=calibration_samples,
        ).collect(calibration_batches, device=device)
        # Preserve the official FP16 calibration arithmetic, then install the
        # decomposition in FP32. This avoids encoding a dense FP16 residual
        # quantization error in an otherwise adapter-only checkpoint.
        model.to(dtype=torch.float32)
        baseline_logits = _logits(model, first_batch, device)
        _, initialization_record = build_track_a_native_adapter(
            model,
            adapter_config,
            calibration_moments=moments,
        )
        initialization = initialization_record.to_dict()
        calibration_seconds = time.monotonic() - calibration_started
        del calibration_batches, moments
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        baseline_logits = _logits(model, first_batch, device)
        model = build_track_a_peft_lora(model, adapter_config)
        initialization = {
            "method": "lora",
            "module_count": sum(
                1 for name, _ in model.named_modules() if name.endswith("lora_A")
            ),
            "exact_function_preserving": True,
            "calibration_required": False,
            "details": {
                "protocol_track": "A",
                "official_commit": TRACK_A_OFFICIAL_COMMIT,
                "implementation": "peft-0.17.1",
                "target_scope": "all_decoder_linears",
                "initialization": "standard_lora",
            },
        }

    initialized_logits = _logits(model, first_batch, device)
    tolerance = (
        {"atol": 2e-4, "rtol": 2e-4}
        if device.type == "cpu"
        else {"atol": 2e-2, "rtol": 2e-2}
    )
    torch.testing.assert_close(initialized_logits, baseline_logits, **tolerance)

    if bool(train_config.get("gradient_checkpointing", False)):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    global_batch = int(train_config["global_batch_size"])
    per_device_batch = int(train_config["per_device_batch_size"])
    if global_batch % per_device_batch != 0:
        raise ValueError(
            "global_batch_size must be divisible by per_device_batch_size"
        )
    accumulation = global_batch // per_device_batch
    if execution_mode == "cpu_smoke_only":
        accumulation = 1
    training_summary = train_epochs(
        model,
        training_batches,
        device=device,
        epochs=int(train_config["epochs"]),
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        warmup_ratio=float(train_config["warmup_ratio"]),
        gradient_accumulation_steps=accumulation,
        precision=str(train_config["precision"]),
        adam_beta1=float(train_config["adam_beta1"]),
        adam_beta2=float(train_config["adam_beta2"]),
        adam_epsilon=float(train_config["adam_epsilon"]),
        max_grad_norm=float(train_config["max_grad_norm"]),
        max_steps=max_steps,
    )
    if execution_mode == "gpu_formal":
        expected_training = {
            "steps": int(train_config["optimizer_steps_per_epoch"]),
            "examples": int(train_config["consumed_examples_per_epoch"]),
            "discarded_micro_batches": int(
                train_config["discarded_examples_per_epoch"]
            ),
        }
        actual_training = {
            "steps": training_summary.steps,
            "examples": training_summary.examples,
            "discarded_micro_batches": (
                training_summary.discarded_micro_batches
            ),
        }
        if actual_training != expected_training:
            raise RuntimeError(
                "Formal Track A training budget drift: "
                f"actual={actual_training}, expected={expected_training}"
            )

    metadata = RunMetadata(
        run_id=config["run"]["name"],
        execution_mode=execution_mode,
        method=method,
        model_id=config["model"]["id"],
        seed=seed,
        commit_sha=git_state["commit_sha"],
    )
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_metadata = {
        **metadata.to_dict(),
        "protocol_track": "A",
        "config_hash": config_hash,
        "protocol_config_hash": protocol_config_hash,
        "comparison_config_hash": comparison_config_hash,
        "source_dirty": git_state["source_dirty"],
    }
    if method == "lora_null":
        save_track_a_native_adapter(
            model,
            str(checkpoint_dir),
            adapter_config=adapter_config,
            metadata=checkpoint_metadata,
        )
    else:
        model.save_pretrained(str(checkpoint_dir), safe_serialization=True)
    checkpoint_integrity = write_directory_integrity(
        str(checkpoint_dir),
        kind="adapter_checkpoint",
    )

    peak_gpu_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    report = {
        "status": "trained" if execution_mode == "gpu_formal" else "passed",
        "protocol_track": "A",
        "protocol_label": "LoRA-Null protocol-aligned Instruct reimplementation",
        "official_commit": TRACK_A_OFFICIAL_COMMIT,
        "metadata": metadata.to_dict(),
        "source_dirty": git_state["source_dirty"],
        "config_hash": config_hash,
        "protocol_config_hash": protocol_config_hash,
        "comparison_config_hash": comparison_config_hash,
        "artifact_retention": config.get("artifact_retention"),
        "admission": {
            "mode": admission.mode,
            "selected_gpu_indices": admission.selected_gpu_indices,
            "idle_gpu_indices": admission.idle_gpu_indices,
            "reason": admission.reason,
        },
        "model": load_record.to_dict(),
        "model_snapshot_integrity": model_integrity_record,
        "environment": environment_record,
        "precision_boundary": {
            "calibration_activation_dtype": str(load_dtype),
            "decomposition_install_dtype": str(next(model.parameters()).dtype),
            "training_parameter_dtype": str(next(model.parameters()).dtype),
            "training_autocast": str(train_config["precision"]),
            "stabilization": (
                "fp16_calibration_then_fp32_adapter_install"
                if method == "lora_null"
                else "official_peft_fp32_install"
            ),
        },
        "dataset": {
            "metamath_rows": train_first_n,
            "training_prompt": "official_lora_null_alpaca_style",
            "response_only_loss": True,
            "nq_calibration_rows": (
                calibration_samples if method == "lora_null" else 0
            ),
            "nq_calibration_sequence_length": (
                calibration_length if method == "lora_null" else 0
            ),
            "nq_sampling": "official_raw_character_spans",
        },
        "initialization": initialization,
        "parameters": count_parameters(model),
        "timing": {
            "data_seconds": data_seconds,
            "model_integrity_validation_seconds": model_integrity_seconds,
            "calibration_seconds": calibration_seconds,
            "training_seconds": training_summary.wall_seconds,
            "total_seconds": time.monotonic() - total_started,
        },
        "resources": {
            "peak_gpu_memory_bytes": peak_gpu_bytes,
            "process_max_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "training": training_summary.to_dict(),
        "checkpoint_integrity": checkpoint_integrity,
        "formal_result_eligible": execution_mode == "gpu_formal",
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA-Null Track A protocol-aligned Instruct SFT runner"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metamath-json", required=True)
    parser.add_argument("--nq-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-smoke-fallback", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-first-n", type=int)
    parser.add_argument("--calibration-samples", type=int)
    parser.add_argument("--method", choices=["lora", "lora_null"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    with project_gpu_lock():
        _run(args)


if __name__ == "__main__":
    main()
