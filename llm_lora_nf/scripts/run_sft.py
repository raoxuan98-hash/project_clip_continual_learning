#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from llm_lora_nf.baselines import (
    build_corda_baseline,
    build_native_adapter,
    build_peft_baseline,
)
from llm_lora_nf.calibration import ActivationCalibrator, build_and_assign_filters
from llm_lora_nf.checkpoint import save_native_adapter
from llm_lora_nf.config_io import (
    adapter_config_from_mapping,
    canonical_config_hash,
    load_yaml_config,
)
from llm_lora_nf.data import (
    CalibrationCollator,
    CalibrationQuestionDataset,
    LazySupervisedChatDataset,
    ResponseOnlyCollator,
)
from llm_lora_nf.dataset_io import load_metamath_examples, sample_nq_questions
from llm_lora_nf.inject import count_parameters
from llm_lora_nf.model_io import load_instruct_model
from llm_lora_nf.resource_guard import inspect_admission, project_gpu_lock
from llm_lora_nf.result_metadata import RunMetadata
from llm_lora_nf.training import set_reproducible_seed, train_epochs


NATIVE_METHODS = {"lora", "lora_nf", "lora_null", "milora"}
PEFT_METHODS = {"dora", "pissa"}


def _git_state() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "llm_lora_nf"],
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified attention-only SFT runner for LoRA baselines and LoRA-NF"
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
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = load_yaml_config(args.config)
    config_hash = canonical_config_hash(config)
    git_state = _git_state()
    requested_mode = config["run"]["execution_mode"]

    with project_gpu_lock():
        admission = inspect_admission(requested=1)
        if requested_mode == "gpu_formal" and admission.mode == "gpu":
            if git_state["source_dirty"]:
                raise RuntimeError("Formal training requires a clean llm_lora_nf worktree")
            selected_gpu = admission.selected_gpu_indices[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
            execution_mode = "gpu_formal"
            device = torch.device("cuda:0")
        elif args.cpu_smoke_fallback:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            execution_mode = "cpu_smoke_only"
            device = torch.device("cpu")
        else:
            raise RuntimeError(
                "No admissible GPU is available; rerun with --cpu-smoke-fallback "
                "only to validate the chain without collecting results"
            )

        seed = int(config["run"]["seed"])
        set_reproducible_seed(seed)
        model, tokenizer, load_record = load_instruct_model(
            requested_id=config["model"]["id"],
            source_id=config["model"].get("source_id", config["model"]["id"]),
            resolved_path=args.model_path,
            checkpoint_type=config["model"]["checkpoint_type"],
            device=str(device),
            local_files_only=True,
        )

    adapter_config = adapter_config_from_mapping(config["adapter"])
    train_config = config["train"]
    calibration_config = config["calibration"]
    if execution_mode == "cpu_smoke_only":
        train_first_n = args.train_first_n or 2
        calibration_samples = args.calibration_samples or 2
        max_steps = args.max_steps or 1
    else:
        train_first_n = args.train_first_n or int(train_config["first_n"])
        calibration_samples = (
            args.calibration_samples or int(calibration_config["samples"])
        )
        max_steps = args.max_steps

    examples = load_metamath_examples(
        args.metamath_json,
        first_n=train_first_n,
    )
    training_dataset = LazySupervisedChatDataset(
        examples,
        tokenizer,
        max_length=int(train_config["max_sequence_length"]),
        enable_thinking=bool(config["model"].get("enable_thinking", False)),
    )
    generator = torch.Generator().manual_seed(seed)
    training_batches = DataLoader(
        training_dataset,
        batch_size=int(train_config["per_device_batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=ResponseOnlyCollator(tokenizer.pad_token_id),
    )
    first_batch = next(iter(training_batches))
    baseline_logits = _logits(model, first_batch, device)

    calibration_batches = None
    moments = None
    if adapter_config.method in {"lora_nf", "lora_null", "corda"}:
        questions = sample_nq_questions(
            args.nq_parquet,
            samples=calibration_samples,
            seed=int(calibration_config["seed"]),
        )
        calibration_dataset = CalibrationQuestionDataset(
            questions,
            tokenizer,
            max_length=int(calibration_config["max_sequence_length"]),
            enable_thinking=bool(config["model"].get("enable_thinking", False)),
        )
        calibration_batch_size = min(
            int(calibration_config["batch_size"]),
            calibration_samples,
        )
        # PEFT 0.17.1 CorDA's official covariance hook squeezes only a
        # singleton batch and then calls Tensor.t(), so batch size > 1 fails
        # on 3D activations. Keep this method-specific constraint explicit.
        if adapter_config.method == "corda":
            calibration_batch_size = 1
        calibration_batches = DataLoader(
            calibration_dataset,
            batch_size=calibration_batch_size,
            shuffle=False,
            collate_fn=CalibrationCollator(tokenizer.pad_token_id),
        )

    initialization: Dict[str, Any]
    filter_results = {}
    if adapter_config.method in {"lora_nf", "lora_null"}:
        moments = ActivationCalibrator(model).collect(
            calibration_batches,
            device=device,
        )
    if adapter_config.method in NATIVE_METHODS:
        _, record = build_native_adapter(
            model,
            adapter_config,
            calibration_moments=moments,
        )
        initialization = record.to_dict()
        if adapter_config.method == "lora_nf":
            filter_results = build_and_assign_filters(
                model,
                moments,
                energy_fraction=adapter_config.filter.energy_fraction,
                leakage=adapter_config.filter.leakage,
                ridge=adapter_config.filter.ridge,
            )
    elif adapter_config.method in PEFT_METHODS:
        model = build_peft_baseline(model, adapter_config, adapter_config.method)
        initialization = {
            "method": adapter_config.method,
            "implementation": "peft-0.17.1",
        }
    elif adapter_config.method == "corda":
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        @torch.no_grad()
        def run_corda_calibration():
            model.eval()
            for batch in calibration_batches:
                model(
                    **{
                        key: value.to(device)
                        for key, value in batch.items()
                    }
                )

        model = build_corda_baseline(
            model,
            adapter_config,
            run_calibration=run_corda_calibration,
            cache_file=str(output_dir / "corda_svd_cache.pt"),
            covariance_file=str(output_dir / "corda_covariance.pt"),
            mode=str(config["adapter"].get("corda_mode", "kpm")),
        )
        initialization = {
            "method": "corda",
            "mode": config["adapter"].get("corda_mode", "kpm"),
            "implementation": "peft-0.17.1",
        }
    else:
        raise ValueError(f"Unsupported adapter method: {adapter_config.method}")

    initialized_logits = _logits(model, first_batch, device)
    tolerance = {"atol": 2e-4, "rtol": 2e-4}
    if device.type == "cuda":
        tolerance = {"atol": 2e-2, "rtol": 2e-2}
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
        raise ValueError("global_batch_size must divide per_device_batch_size")
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
        max_steps=max_steps,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = RunMetadata(
        run_id=config["run"]["name"],
        execution_mode=execution_mode,
        method=adapter_config.method,
        model_id=config["model"]["id"],
        seed=seed,
        commit_sha=git_state["commit_sha"],
    )
    checkpoint_dir = output_dir / "checkpoint"
    if adapter_config.method in NATIVE_METHODS:
        save_native_adapter(
            model,
            str(checkpoint_dir),
            adapter_config=adapter_config,
            metadata={
                **metadata.to_dict(),
                "config_hash": config_hash,
                "source_dirty": git_state["source_dirty"],
            },
        )
    else:
        model.save_pretrained(str(checkpoint_dir), safe_serialization=True)

    report = {
        "status": "trained" if execution_mode == "gpu_formal" else "passed",
        "metadata": metadata.to_dict(),
        "source_dirty": git_state["source_dirty"],
        "config_hash": config_hash,
        "admission": {
            "mode": admission.mode,
            "selected_gpu_indices": admission.selected_gpu_indices,
            "idle_gpu_indices": admission.idle_gpu_indices,
            "reason": admission.reason,
        },
        "model": load_record.to_dict(),
        "dataset": {
            "metamath_rows": train_first_n,
            "nq_calibration_rows": (
                calibration_samples if calibration_batches is not None else 0
            ),
            "response_only_loss": True,
        },
        "initialization": initialization,
        "parameters": count_parameters(model),
        "training": training_summary.to_dict(),
        "filter_summary": {
            group: {
                "tail_dimension": result.tail_dimension,
                "protected_dimension": result.protected_dimension,
                "captured_tail_energy": result.captured_tail_energy,
            }
            for group, result in filter_results.items()
        },
        "formal_result_eligible": execution_mode == "gpu_formal",
    }
    (output_dir / "run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
