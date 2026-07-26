#!/usr/bin/env python3
import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from llm_lora_nf.artifact_retention import validate_artifact_retention
from llm_lora_nf.baselines import (
    build_native_adapter,
    build_peft_baseline,
    initialize_lora_null,
)
from llm_lora_nf.calibration import (
    ActivationCalibrator,
    build_and_assign_filters,
)
from llm_lora_nf.config_io import (
    adapter_config_from_mapping,
    canonical_comparison_config_hash,
    canonical_config_hash,
    load_yaml_config,
)
from llm_lora_nf.continual_adapter import (
    CumulativeAdapterState,
    capture_and_merge_native_task,
    capture_and_merge_peft_task,
    capture_native_protection_state,
    load_cumulative_adapter,
    save_cumulative_adapter,
    update_continual_moment_map,
)
from llm_lora_nf.data import (
    CalibrationCollator,
    CalibrationQuestionDataset,
    LazySupervisedChatDataset,
    ResponseOnlyCollator,
)
from llm_lora_nf.dataset_io import (
    official_lora_null_nq_character_spans,
    sample_nq_questions,
)
from llm_lora_nf.environment import runtime_environment
from llm_lora_nf.inject import adapter_modules
from llm_lora_nf.integrity import (
    validate_directory_integrity,
    write_directory_integrity,
)
from llm_lora_nf.model_io import load_instruct_model
from llm_lora_nf.resource_guard import (
    admitted_torch_device,
    inspect_admission,
    project_gpu_lock,
)
from llm_lora_nf.trace_data import (
    generate_trace_predictions,
    load_trace_split,
    trace_chat_examples,
    write_trace_prediction_artifact,
)
from llm_lora_nf.trace_metrics import (
    aggregate_trace_prediction_artifacts,
    write_trace_metrics,
)
from llm_lora_nf.trace_protocol import (
    TRACE_ALTERNATE_ORDER,
    TRACE_LORA_EPOCHS,
    TRACE_OFFICIAL_ORDER,
    assert_task_order,
    audit_trace_dataset,
    classify_trace_execution,
)
from llm_lora_nf.track_a import OfficialLoRANullAttentionCalibrator
from llm_lora_nf.training import set_reproducible_seed, train_epochs


TRACE_METHODS = {"lora", "dora", "lora_null", "lora_nf"}
NATIVE_TRACE_METHODS = {"lora_null", "lora_nf"}
PEFT_TRACE_METHODS = {"lora", "dora"}


def _git_state() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "llm_lora_nf"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit_sha": commit, "source_dirty": dirty}


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _projection_summaries(results: Mapping[str, Any]) -> Dict[str, Any]:
    summaries = {}
    for group, result in results.items():
        eigenvalues = result.eigenvalues.float()
        summaries[group] = {
            "tail_dimension": int(result.tail_dimension),
            "protected_dimension": int(result.protected_dimension),
            "captured_tail_energy": float(result.captured_tail_energy),
            "minimum_eigenvalue": float(eigenvalues.min().item()),
            "maximum_eigenvalue": float(eigenvalues.max().item()),
            "eigenvalue_sum": float(eigenvalues.sum().item()),
            "eigenvalues_sha256": hashlib.sha256(
                eigenvalues.numpy().tobytes()
            ).hexdigest(),
        }
    return summaries


def _make_nq_calibration(
    *,
    method: str,
    model,
    tokenizer,
    nq_parquet: str,
    config: Mapping[str, Any],
    device: torch.device,
    samples: int,
    execution_mode: str,
):
    calibration = config["calibration"]
    if method == "lora_nf":
        sequence_length = int(calibration["max_sequence_length"])
        if execution_mode == "cpu_smoke_only":
            sequence_length = min(sequence_length, 64)
        questions = sample_nq_questions(
            nq_parquet,
            samples=samples,
            seed=int(calibration["seed"]),
        )
        dataset = CalibrationQuestionDataset(
            questions,
            tokenizer,
            max_length=sequence_length,
            enable_thinking=bool(config["model"].get("enable_thinking", False)),
        )
        batches = DataLoader(
            dataset,
            batch_size=min(int(calibration["batch_size"]), samples),
            shuffle=False,
            collate_fn=CalibrationCollator(tokenizer.pad_token_id),
        )
        moments = ActivationCalibrator(model).collect(
            batches,
            device=device,
        )
        protocol = "instruct_chat_nonpadding_second_moment"
    elif method == "lora_null":
        sequence_length = int(
            calibration.get("lora_null_max_sequence_length", 2048)
        )
        if execution_mode == "cpu_smoke_only":
            sequence_length = min(sequence_length, 64)
        batches = official_lora_null_nq_character_spans(
            nq_parquet,
            tokenizer,
            samples=samples,
            seed=int(calibration["seed"]),
            sequence_length=sequence_length,
        )
        moments = OfficialLoRANullAttentionCalibrator(
            model,
            expected_samples=samples,
        ).collect(batches, device=device)
        protocol = "official_raw_spans_signed_global_max"
    else:
        return None, "none"
    return moments, protocol


def _prepare_adapter(
    model,
    *,
    method: str,
    adapter_config,
    reference_moments,
):
    if method in NATIVE_TRACE_METHODS:
        _, record = build_native_adapter(
            model,
            adapter_config,
            calibration_moments=reference_moments,
        )
        filter_results = {}
        if method == "lora_nf":
            filter_results = build_and_assign_filters(
                model,
                reference_moments,
                energy_fraction=adapter_config.filter.energy_fraction,
                leakage=adapter_config.filter.leakage,
                ridge=adapter_config.filter.ridge,
            )
        state = CumulativeAdapterState.create(
            method=method,
            backend="native_factor_stack",
            adapter_config=adapter_config,
        )
        return model, state, record.to_dict(), filter_results
    if method in PEFT_TRACE_METHODS:
        model = build_peft_baseline(model, adapter_config, method)
        state = CumulativeAdapterState.create(
            method=method,
            backend="peft_state_stack",
            adapter_config=adapter_config,
        )
        return (
            model,
            state,
            {"method": method, "implementation": "peft-0.17.1"},
            {},
        )
    raise ValueError(f"Unsupported TRACE method: {method}")


def _enable_training_runtime(model, enabled: bool) -> None:
    if enabled:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True


def _run(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = load_yaml_config(args.config)
    if args.method is not None:
        config["adapter"]["method"] = args.method
    method = str(config["adapter"]["method"])
    if method not in TRACE_METHODS:
        raise ValueError("TRACE main comparison is LoRA/DoRA/LoRA-Null/LoRA-NF")
    if method in NATIVE_TRACE_METHODS and args.nq_parquet is None:
        raise ValueError(f"{method} requires --nq-parquet calibration data")
    validate_artifact_retention(config["artifact_retention"])
    if args.seed is not None:
        config["run"]["seed"] = int(args.seed)
    seed = int(config["run"]["seed"])
    order = (
        TRACE_OFFICIAL_ORDER
        if args.order == "official"
        else TRACE_ALTERNATE_ORDER
    )
    assert_task_order(order)
    git_state = _git_state()
    requested_mode = str(config["run"]["execution_mode"])
    if requested_mode not in {"gpu_pilot", "gpu_formal"}:
        raise ValueError("TRACE config execution_mode must be gpu_pilot/gpu_formal")
    has_smoke_overrides = any(
        value is not None
        for value in (
            args.max_tasks,
            args.train_first_n,
            args.test_first_n,
            args.max_steps,
            args.max_new_tokens,
            args.calibration_samples,
            True if args.delete_checkpoint_after_smoke else None,
        )
    )

    data_protocol = str(config["trace"]["data_protocol"])
    data_audit = audit_trace_dataset(
        args.trace_data_root,
        protocol=data_protocol,
    )
    if requested_mode == "gpu_formal":
        if git_state["source_dirty"]:
            raise RuntimeError("Formal TRACE training requires a clean worktree")
        if not data_audit["formal_eligible"]:
            raise RuntimeError(
                "TRACE paper_5k lacks an approved exact-source manifest; "
                "formal training is blocked"
            )
        if has_smoke_overrides:
            raise ValueError("Formal TRACE training forbids smoke overrides")

    admission = inspect_admission(requested=1)
    execution_mode = classify_trace_execution(
        requested_mode=requested_mode,
        gpu_admitted=admission.mode == "gpu",
        cpu_smoke_fallback=args.cpu_smoke_fallback,
        has_smoke_overrides=has_smoke_overrides,
    )
    if admission.mode == "gpu":
        selected_gpu = admission.selected_gpu_indices[0]
        device = torch.device(
            admitted_torch_device(
                selected_gpu,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        )
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    max_tasks = len(order)
    if execution_mode == "cpu_smoke_only":
        max_tasks = args.max_tasks or 1
    elif args.max_tasks is not None:
        max_tasks = int(args.max_tasks)
    if not 1 <= max_tasks <= len(order):
        raise ValueError("max_tasks lies outside the TRACE task order")
    active_order = tuple(order[:max_tasks])
    train_first_n = (
        args.train_first_n
        if args.train_first_n is not None
        else (2 if execution_mode == "cpu_smoke_only" else None)
    )
    test_first_n = (
        args.test_first_n
        if args.test_first_n is not None
        else (1 if execution_mode == "cpu_smoke_only" else None)
    )
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else (1 if execution_mode == "cpu_smoke_only" else None)
    )
    calibration_samples = (
        args.calibration_samples
        if args.calibration_samples is not None
        else (
            2
            if execution_mode == "cpu_smoke_only"
            else int(config["calibration"]["samples"])
        )
    )
    generation_max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else (
            8
            if execution_mode == "cpu_smoke_only"
            else int(config["generation"]["max_new_tokens"])
        )
    )
    training_max_sequence_length = int(config["train"]["max_sequence_length"])
    generation_max_prompt_length = int(
        config["generation"]["max_prompt_length"]
    )
    if execution_mode == "cpu_smoke_only":
        training_max_sequence_length = min(training_max_sequence_length, 64)
        generation_max_prompt_length = min(
            generation_max_prompt_length,
            64,
        )

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    predictions_dir = output_dir / "predictions"
    checkpoint_dir = output_dir / "checkpoint"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    config_hash = canonical_config_hash(config)
    protocol_config = copy.deepcopy(config)
    protocol_config["run"].pop("seed", None)
    protocol_config["run"].pop("name", None)
    protocol_config_hash = canonical_config_hash(protocol_config)
    comparison_config_hash = canonical_comparison_config_hash(config)
    data_identity = {
        "protocol": data_audit["protocol"],
        "qualification": data_audit["qualification"],
        "files_sha256": _canonical_hash(data_audit["files"]),
        "source": data_audit.get("source"),
    }
    set_reproducible_seed(seed)
    total_started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    environment_record = runtime_environment(
        device,
        validate_formal=execution_mode == "gpu_formal",
    )
    model_integrity_record = None
    if execution_mode == "gpu_formal":
        model_integrity_record = validate_directory_integrity(
            args.model_path,
            expected_kind="model_snapshot",
        )
    model, tokenizer, load_record = load_instruct_model(
        requested_id=config["model"]["id"],
        source_id=config["model"].get(
            "source_id",
            config["model"]["id"],
        ),
        resolved_path=args.model_path,
        checkpoint_type=config["model"]["checkpoint_type"],
        device=str(device),
        local_files_only=True,
    )
    if (
        execution_mode == "gpu_formal"
        and (
            load_record.snapshot_manifest_sha256 is None
            or load_record.snapshot_integrity_sha256 is None
        )
    ):
        raise RuntimeError(
            "Formal TRACE training requires a hashed local model snapshot"
        )
    adapter_config = adapter_config_from_mapping(config["adapter"])
    calibration_started = time.monotonic()
    reference_moments, calibration_protocol = _make_nq_calibration(
        method=method,
        model=model,
        tokenizer=tokenizer,
        nq_parquet=args.nq_parquet,
        config=config,
        device=device,
        samples=int(calibration_samples),
        execution_mode=execution_mode,
    )
    calibration_seconds = time.monotonic() - calibration_started
    model, cumulative_state, initialization, filter_results = _prepare_adapter(
        model,
        method=method,
        adapter_config=adapter_config,
        reference_moments=reference_moments,
    )
    continual_moments = reference_moments if method == "lora_nf" else None
    state_update_mode = str(config["trace"]["state_update"])
    if method == "lora_nf" and state_update_mode not in {
        "reference_fixed",
        "reference_plus_history",
    }:
        raise ValueError("LoRA-NF TRACE state_update is not preregistered")

    run_identity = {
        "method": method,
        "model_id": config["model"]["id"],
        "model_snapshot_revision": load_record.snapshot_revision,
        "seed": seed,
        "commit_sha": git_state["commit_sha"],
        "config_hash": config_hash,
        "protocol_config_hash": protocol_config_hash,
        "comparison_config_hash": comparison_config_hash,
        "data_identity": data_identity,
        "execution_mode": execution_mode,
        "order": args.order,
        "active_task_order": list(active_order),
        "state_update": state_update_mode if method == "lora_nf" else "none",
    }
    train_config = config["train"]
    generation_config = config["generation"]
    stages = []
    prediction_records = []
    for task_index, task in enumerate(active_order):
        task_seed = seed + 1000 * task_index
        set_reproducible_seed(task_seed)
        if task_index > 0:
            if method == "lora_nf":
                for _, module in adapter_modules(model):
                    module.reset_lora_parameters()
            elif method == "lora_null":
                initialize_lora_null(model, reference_moments)
            else:
                model = build_peft_baseline(model, adapter_config, method)

        train_rows = load_trace_split(
            args.trace_data_root,
            task=task,
            split="train",
        )
        if train_first_n is not None:
            train_rows = train_rows[: int(train_first_n)]
        training_dataset = LazySupervisedChatDataset(
            trace_chat_examples(train_rows),
            tokenizer,
            max_length=training_max_sequence_length,
            enable_thinking=bool(
                config["model"].get("enable_thinking", False)
            ),
        )
        training_generator = torch.Generator().manual_seed(task_seed)
        collator = ResponseOnlyCollator(tokenizer.pad_token_id)
        training_batches = DataLoader(
            training_dataset,
            batch_size=int(train_config["per_device_batch_size"]),
            shuffle=True,
            generator=training_generator,
            collate_fn=collator,
        )
        global_batch = int(train_config["global_batch_size"])
        per_device_batch = int(train_config["per_device_batch_size"])
        if global_batch % per_device_batch:
            raise ValueError("TRACE global batch must divide per-device batch")
        accumulation = global_batch // per_device_batch
        if execution_mode == "cpu_smoke_only":
            accumulation = 1
        _enable_training_runtime(model, True)
        training_summary = train_epochs(
            model,
            training_batches,
            device=device,
            epochs=(
                1
                if execution_mode == "cpu_smoke_only"
                else int(TRACE_LORA_EPOCHS[task])
            ),
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
        _enable_training_runtime(model, False)

        task_moment_record = None
        if method == "lora_nf":
            task_moments = None
            if state_update_mode == "reference_plus_history":
                moment_batches = DataLoader(
                    training_dataset,
                    batch_size=int(config["calibration"]["batch_size"]),
                    shuffle=False,
                    collate_fn=collator,
                )
                moment_started = time.monotonic()
                task_moments = ActivationCalibrator(model).collect(
                    moment_batches,
                    device=device,
                )
                task_moment_record = {
                    "seconds": time.monotonic() - moment_started,
                    "groups": len(task_moments),
                    "observations": {
                        group: state.observations
                        for group, state in task_moments.items()
                    },
                }
            capture_and_merge_native_task(
                model,
                cumulative_state,
                task=task,
                metadata={"task_seed": task_seed},
            )
            if task_moments is not None:
                continual_moments = update_continual_moment_map(
                    continual_moments,
                    task_moments,
                    mode=state_update_mode,
                )
                filter_results = build_and_assign_filters(
                    model,
                    continual_moments,
                    energy_fraction=adapter_config.filter.energy_fraction,
                    leakage=adapter_config.filter.leakage,
                    ridge=adapter_config.filter.ridge,
                )
        elif method == "lora_null":
            capture_and_merge_native_task(
                model,
                cumulative_state,
                task=task,
                metadata={"task_seed": task_seed},
            )
        else:
            model, _ = capture_and_merge_peft_task(
                model,
                cumulative_state,
                task=task,
                metadata={"task_seed": task_seed},
            )

        stage_prediction_records = []
        for evaluation_task in active_order[: task_index + 1]:
            test_rows = load_trace_split(
                args.trace_data_root,
                task=evaluation_task,
                split=str(generation_config["split"]),
            )
            if test_first_n is not None:
                test_rows = test_rows[: int(test_first_n)]
            predictions = generate_trace_predictions(
                model,
                tokenizer,
                test_rows,
                device=device,
                batch_size=int(generation_config["batch_size"]),
                max_prompt_length=generation_max_prompt_length,
                max_new_tokens=int(generation_max_new_tokens),
                enable_thinking=bool(
                    config["model"].get("enable_thinking", False)
                ),
            )
            record = write_trace_prediction_artifact(
                str(
                    predictions_dir
                    / (
                        f"stage_{task_index + 1:02d}_"
                        f"{evaluation_task}.json.gz"
                    )
                ),
                task=evaluation_task,
                stage=task_index + 1,
                task_order=active_order,
                predictions=predictions,
                generation={
                    "split": generation_config["split"],
                    "batch_size": int(generation_config["batch_size"]),
                    "max_prompt_length": generation_max_prompt_length,
                    "max_new_tokens": int(generation_max_new_tokens),
                    "do_sample": False,
                    "num_beams": 1,
                },
                identity=run_identity,
            )
            stage_prediction_records.append(record)
            prediction_records.append(record)
        stages.append(
            {
                "stage": task_index + 1,
                "task": task,
                "task_seed": task_seed,
                "training": training_summary.to_dict(),
                "task_moments": task_moment_record,
                "predictions": stage_prediction_records,
            }
        )

    if method == "lora_nf":
        phase_counts = {
            group: state.phases
            for group, state in continual_moments.items()
        }
        cumulative_state.protection_state = capture_native_protection_state(
            model,
            moment_phases=phase_counts,
            summaries=_projection_summaries(filter_results),
        )
    save_cumulative_adapter(
        cumulative_state,
        str(checkpoint_dir),
        metadata=run_identity,
    )
    checkpoint_integrity = write_directory_integrity(
        str(checkpoint_dir),
        kind="adapter_checkpoint",
    )
    restored_state, restored_manifest = load_cumulative_adapter(
        str(checkpoint_dir)
    )
    if (
        restored_state.tasks != list(active_order)
        or restored_manifest["tasks"] != list(active_order)
    ):
        raise RuntimeError("Final cumulative adapter failed structural replay audit")
    metrics_result = aggregate_trace_prediction_artifacts(
        [record["path"] for record in prediction_records],
        task_order=active_order,
    )
    metrics_path = write_trace_metrics(
        str(output_dir / "trace_metrics.json"),
        metrics_result,
    )
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else 0
    )
    checkpoint_retention = {
        "path": str(checkpoint_dir),
        "deleted_after_smoke_audit": False,
    }
    if args.delete_checkpoint_after_smoke:
        if execution_mode == "gpu_formal":
            raise ValueError("Formal TRACE checkpoints cannot be smoke-deleted")
        if (
            checkpoint_dir.parent != output_dir
            or checkpoint_dir.name != "checkpoint"
        ):
            raise RuntimeError("Refusing unsafe TRACE checkpoint deletion target")
        validate_directory_integrity(
            str(checkpoint_dir),
            expected_kind="adapter_checkpoint",
        )
        shutil.rmtree(checkpoint_dir)
        checkpoint_retention["deleted_after_smoke_audit"] = True
    report = {
        "status": (
            "trained"
            if execution_mode in {"gpu_pilot", "gpu_formal"}
            else "passed"
        ),
        "formal_result_eligible": execution_mode == "gpu_formal",
        "identity": run_identity,
        "source_dirty": git_state["source_dirty"],
        "admission": {
            "mode": admission.mode,
            "selected_gpu_indices": admission.selected_gpu_indices,
            "idle_gpu_indices": admission.idle_gpu_indices,
            "reason": admission.reason,
        },
        "model": load_record.to_dict(),
        "model_integrity": model_integrity_record,
        "environment": environment_record,
        "data_audit": data_audit,
        "calibration": {
            "protocol": calibration_protocol,
            "samples": int(calibration_samples)
            if reference_moments is not None
            else 0,
            "seconds": calibration_seconds,
        },
        "initialization": initialization,
        "stages": stages,
        "checkpoint_integrity": checkpoint_integrity,
        "checkpoint_retention": checkpoint_retention,
        "metrics_path": metrics_path,
        "summary": metrics_result["summary"],
        "artifact_retention": config["artifact_retention"],
        "effective_limits": {
            "tasks": len(active_order),
            "train_first_n": train_first_n,
            "test_first_n": test_first_n,
            "max_steps": max_steps,
            "training_max_sequence_length": training_max_sequence_length,
            "generation_max_prompt_length": generation_max_prompt_length,
            "generation_max_new_tokens": generation_max_new_tokens,
        },
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "wall_seconds": time.monotonic() - total_started,
    }
    report_path = output_dir / "run_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps({"report": str(report_path), **report["summary"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run low-storage TRACE continual attention-LoRA comparison"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-data-root", required=True)
    parser.add_argument("--nq-parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=sorted(TRACE_METHODS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--order", choices=["official", "alternate"], default="official")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--train-first-n", type=int)
    parser.add_argument("--test-first-n", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--calibration-samples", type=int)
    parser.add_argument("--cpu-smoke-fallback", action="store_true")
    parser.add_argument(
        "--delete-checkpoint-after-smoke",
        action="store_true",
        help=(
            "After integrity and structural replay checks, delete the final "
            "adapter from a non-formal chain smoke run."
        ),
    )
    parser.add_argument(
        "--gpu-lock",
        default="/tmp/llm_lora_nf_trace_gpu.lock",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    with project_gpu_lock(parsed.gpu_lock):
        _run(parsed)
