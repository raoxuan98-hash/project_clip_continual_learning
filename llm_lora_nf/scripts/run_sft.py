#!/usr/bin/env python3
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

from llm_lora_nf.baselines import (
    build_corda_baseline,
    build_native_adapter,
    build_peft_baseline,
)
from llm_lora_nf.calibration import ActivationCalibrator, build_and_assign_filters
from llm_lora_nf.calibration_cache import (
    build_calibration_cache_identity,
    load_artifact_cache,
    load_calibration_cache,
    save_artifact_cache,
    save_calibration_cache,
)
from llm_lora_nf.checkpoint import save_native_adapter
from llm_lora_nf.config_io import (
    adapter_config_from_mapping,
    canonical_comparison_config_hash,
    canonical_config_hash,
    load_yaml_config,
)
from llm_lora_nf.data import (
    CalibrationCollator,
    CalibrationQuestionDataset,
    LazySupervisedChatDataset,
    ResponseOnlyCollator,
)
from llm_lora_nf.dataset_io import (
    load_training_examples,
    official_lora_null_nq_character_spans,
    sample_nq_questions,
)
from llm_lora_nf.environment import runtime_environment
from llm_lora_nf.inject import count_parameters
from llm_lora_nf.integrity import (
    validate_directory_integrity,
    write_directory_integrity,
)
from llm_lora_nf.model_io import load_instruct_model
from llm_lora_nf.method_registry import (
    TRACK_B_DIRECT_PEFT_METHODS,
    TRACK_B_NATIVE_METHODS,
)
from llm_lora_nf.protocol_validation import validate_track_b_formal_config
from llm_lora_nf.resource_guard import inspect_admission, project_gpu_lock
from llm_lora_nf.result_metadata import RunMetadata
from llm_lora_nf.track_a import OfficialLoRANullAttentionCalibrator
from llm_lora_nf.training import set_reproducible_seed, train_epochs


NATIVE_METHODS = TRACK_B_NATIVE_METHODS
PEFT_METHODS = TRACK_B_DIRECT_PEFT_METHODS


def _remove_corda_ephemeral_artifacts(
    output_dir: Path,
    *,
    require_eigens: bool,
) -> list[str]:
    removed = []
    for name in ("corda_svd_cache.pt", "corda_covariance.pt"):
        artifact = output_dir / name
        if not artifact.exists():
            continue
        if not artifact.is_file():
            raise ValueError(
                f"CorDA ephemeral artifact is not a regular file: {artifact}"
            )
        artifact.unlink()
        removed.append(name)
    if require_eigens and "corda_svd_cache.pt" not in removed:
        raise FileNotFoundError(
            "CorDA preprocessing did not produce its declared ephemeral "
            "eigens artifact"
        )
    return sorted(removed)


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
    if args.method is not None:
        config["adapter"]["method"] = args.method
        if args.method == "corda":
            config["adapter"].setdefault("corda_mode", "kpm")
    if "method" not in config["adapter"]:
        raise ValueError("adapter.method is required unless --method is provided")
    if args.seed is not None:
        config["run"]["seed"] = args.seed
    if args.run_name is not None:
        config["run"]["name"] = args.run_name
    elif "name" not in config["run"]:
        model_slug = config["model"]["id"].rsplit("/", 1)[-1].lower()
        config["run"]["name"] = (
            f"{model_slug}_{config['adapter']['method']}_seed{config['run']['seed']}"
        )
    config_hash = canonical_config_hash(config)
    protocol_config = copy.deepcopy(config)
    protocol_config["run"].pop("seed", None)
    protocol_config["run"].pop("name", None)
    protocol_config_hash = canonical_config_hash(protocol_config)
    comparison_config_hash = canonical_comparison_config_hash(config)
    git_state = _git_state()
    requested_mode = config["run"]["execution_mode"]

    admission = inspect_admission(requested=1)
    if requested_mode == "gpu_formal" and admission.mode == "gpu":
        validate_track_b_formal_config(config)
        if git_state["source_dirty"]:
            raise RuntimeError("Formal training requires a clean llm_lora_nf worktree")
        if any(
            value is not None
            for value in (
                args.max_steps,
                args.train_first_n,
                args.calibration_samples,
            )
        ):
            raise ValueError(
                "Formal training forbids max-step, row-count, and calibration "
                "overrides"
            )
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
    adapter_config = adapter_config_from_mapping(config["adapter"])
    if (
        execution_mode == "gpu_formal"
        and adapter_config.method in {"lora_nf", "lora_null", "corda"}
        and args.calibration_cache_dir is None
    ):
        raise ValueError(
            "Formal calibration-based methods require the external "
            "content-addressed calibration cache"
        )
    environment_record = runtime_environment(
        device,
        validate_formal=execution_mode == "gpu_formal",
    )

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
    model_load_started = time.monotonic()
    model, tokenizer, load_record = load_instruct_model(
        requested_id=config["model"]["id"],
        source_id=config["model"].get("source_id", config["model"]["id"]),
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
            "Formal training requires hashed provenance and integrity manifests "
            "for the local model snapshot"
        )
    model_load_seconds = time.monotonic() - model_load_started

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

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_started = time.monotonic()
    training_data_path = args.train_json or args.metamath_json
    if training_data_path is None:
        raise ValueError("A training JSON path is required")
    training_dataset_name = str(train_config["dataset"])
    if args.metamath_json is not None and training_dataset_name != "metamathqa":
        raise ValueError(
            "--metamath-json is a legacy alias that may only be used with "
            "train.dataset=metamathqa"
        )
    examples = load_training_examples(
        training_dataset_name,
        training_data_path,
        first_n=train_first_n,
        formal=execution_mode == "gpu_formal",
    )
    training_dataset = LazySupervisedChatDataset(
        examples,
        tokenizer,
        max_length=int(train_config["max_sequence_length"]),
        enable_thinking=bool(config["model"].get("enable_thinking", False)),
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
    baseline_logits = _logits(model, first_batch, device)

    calibration_batches = None
    moments = None
    calibration_seconds = 0.0
    calibration_prepare_seconds = 0.0
    calibration_cache_write_seconds = 0.0
    calibration_protocol = "none"
    calibration_sequence_length = 0
    calibration_cache_identity = None
    corda_cached_eigens = None
    calibration_cache_record: Dict[str, Any] = {
        "enabled": False,
        "status": "disabled",
    }
    if args.calibration_cache_dir is not None and git_state["source_dirty"]:
        calibration_cache_record["status"] = "disabled_dirty_source"
    calibration_prepare_started = time.monotonic()
    if adapter_config.method == "lora_null":
        calibration_sequence_length = int(
            calibration_config.get("lora_null_max_sequence_length", 2048)
        )
        if execution_mode == "cpu_smoke_only":
            calibration_sequence_length = min(calibration_sequence_length, 64)
        elif (
            calibration_samples != 256
            or calibration_sequence_length != 2048
        ):
            raise ValueError(
                "Formal Track B LoRA-Null calibration is locked to "
                "256 samples x 2048 tokens"
            )
        calibration_protocol = (
            "official_raw_character_spans_signed_global_max"
        )
    elif adapter_config.method == "lora_nf":
        calibration_sequence_length = int(
            calibration_config["max_sequence_length"]
        )
        if (
            execution_mode == "gpu_formal"
            and calibration_samples != 256
        ):
            raise ValueError(
                "Formal Track B calibration is locked to 256 NQ samples"
            )
        calibration_protocol = (
            "lora_nf_instruct_chat_nonpadding_second_moment"
        )
    elif adapter_config.method == "corda":
        calibration_sequence_length = int(
            calibration_config.get("corda_max_sequence_length", 2048)
        )
        if execution_mode == "cpu_smoke_only":
            calibration_sequence_length = min(calibration_sequence_length, 64)
        elif (
            calibration_samples != 256
            or calibration_sequence_length != 2048
        ):
            raise ValueError(
                "Formal CorDA KPM calibration is locked to "
                "256 samples x 2048 tokens"
            )
        calibration_protocol = (
            "corda_kpm_official_raw_character_spans_signed_global_max_fp32"
        )
    if (
        adapter_config.method in {"lora_nf", "lora_null", "corda"}
        and args.calibration_cache_dir is not None
        and not git_state["source_dirty"]
    ):
        calibration_cache_identity = build_calibration_cache_identity(
            method=adapter_config.method,
            protocol=calibration_protocol,
            code_commit=git_state["commit_sha"],
            model_path=args.model_path,
            requested_model_id=load_record.requested_id,
            source_model_id=load_record.source_id,
            model_torch_dtype=load_record.torch_dtype,
            nq_parquet=args.nq_parquet,
            nq_revision=str(calibration_config["revision"]),
            samples=calibration_samples,
            seed=int(calibration_config["seed"]),
            sequence_length=calibration_sequence_length,
            target_modules=tuple(adapter_config.target_modules),
            require_model_manifest=execution_mode == "gpu_formal",
            extra=(
                {
                    "rank": adapter_config.rank,
                    "alpha": adapter_config.alpha,
                    "mode": str(
                        config["adapter"].get("corda_mode", "kpm")
                    ),
                    "covariance_dtype": "float32",
                    "peft_source_commit": (
                        "53c25fe4fdd7c6aa4b40db0560815ea570d32303"
                    ),
                }
                if adapter_config.method == "corda"
                else {}
            ),
        )
        if adapter_config.method == "corda":
            (
                corda_cached_eigens,
                calibration_cache_record,
            ) = load_artifact_cache(
                args.calibration_cache_dir,
                calibration_cache_identity,
                artifact_name="corda_eigens.pt",
            )
        else:
            moments, calibration_cache_record = load_calibration_cache(
                args.calibration_cache_dir,
                calibration_cache_identity,
            )
    if moments is None and adapter_config.method == "lora_null":
        calibration_batches = official_lora_null_nq_character_spans(
            args.nq_parquet,
            tokenizer,
            samples=calibration_samples,
            seed=int(calibration_config["seed"]),
            sequence_length=calibration_sequence_length,
        )
    elif (
        adapter_config.method == "corda"
        and corda_cached_eigens is None
    ):
        calibration_batches = official_lora_null_nq_character_spans(
            args.nq_parquet,
            tokenizer,
            samples=calibration_samples,
            seed=int(calibration_config["seed"]),
            sequence_length=calibration_sequence_length,
        )
    elif moments is None and adapter_config.method == "lora_nf":
        questions = sample_nq_questions(
            args.nq_parquet,
            samples=calibration_samples,
            seed=int(calibration_config["seed"]),
        )
        calibration_dataset = CalibrationQuestionDataset(
            questions,
            tokenizer,
            max_length=calibration_sequence_length,
            enable_thinking=bool(config["model"].get("enable_thinking", False)),
        )
        calibration_batch_size = min(
            int(calibration_config["batch_size"]),
            calibration_samples,
        )
        calibration_batches = DataLoader(
            calibration_dataset,
            batch_size=calibration_batch_size,
            shuffle=False,
            collate_fn=CalibrationCollator(tokenizer.pad_token_id),
        )
    calibration_prepare_seconds = (
        time.monotonic() - calibration_prepare_started
        if adapter_config.method in {"lora_nf", "lora_null", "corda"}
        else 0.0
    )

    initialization: Dict[str, Any]
    filter_results = {}
    corda_ephemeral_files_removed = []
    adapter_setup_started = time.monotonic()
    if adapter_config.method == "lora_nf" and moments is None:
        calibration_started = time.monotonic()
        moments = ActivationCalibrator(model).collect(
            calibration_batches,
            device=device,
        )
        calibration_seconds = time.monotonic() - calibration_started
    elif adapter_config.method == "lora_null" and moments is None:
        calibration_started = time.monotonic()
        moments = OfficialLoRANullAttentionCalibrator(
            model,
            expected_samples=calibration_samples,
        ).collect(
            calibration_batches,
            device=device,
        )
        calibration_seconds = time.monotonic() - calibration_started
    if (
        adapter_config.method in {"lora_nf", "lora_null"}
        and calibration_cache_identity is not None
        and calibration_cache_record["status"] == "miss"
    ):
        cache_write_started = time.monotonic()
        calibration_cache_record = save_calibration_cache(
            args.calibration_cache_dir,
            calibration_cache_identity,
            moments,
        )
        calibration_cache_write_seconds = (
            time.monotonic() - cache_write_started
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
        corda_calibration_seconds = 0.0
        corda_temporary_eigens = output_dir / "corda_svd_cache.pt"
        corda_temporary_covariance = output_dir / "corda_covariance.pt"

        @torch.no_grad()
        def run_corda_calibration():
            nonlocal corda_calibration_seconds
            started = time.monotonic()
            model.eval()
            for batch in calibration_batches:
                model(
                    **{
                        key: value.to(device)
                        for key, value in batch.items()
                    }
                )
            corda_calibration_seconds += time.monotonic() - started

        model = build_corda_baseline(
            model,
            adapter_config,
            run_calibration=run_corda_calibration,
            cache_file=str(
                corda_cached_eigens
                if corda_cached_eigens is not None
                else corda_temporary_eigens
            ),
            covariance_file=(
                None
                if corda_cached_eigens is not None
                else str(corda_temporary_covariance)
            ),
            mode=str(config["adapter"].get("corda_mode", "kpm")),
        )
        if (
            calibration_cache_identity is not None
            and calibration_cache_record["status"] == "miss"
        ):
            cache_write_started = time.monotonic()
            calibration_cache_record = save_artifact_cache(
                args.calibration_cache_dir,
                calibration_cache_identity,
                source=str(output_dir / "corda_svd_cache.pt"),
                artifact_name="corda_eigens.pt",
            )
            calibration_cache_write_seconds = (
                time.monotonic() - cache_write_started
            )
        if (
            calibration_cache_identity is not None
            and args.calibration_cache_dir is not None
        ):
            verified_artifact, verified_record = load_artifact_cache(
                args.calibration_cache_dir,
                calibration_cache_identity,
                artifact_name="corda_eigens.pt",
            )
            if (
                verified_artifact is None
                or verified_record.get("key")
                != calibration_cache_record.get("key")
                or verified_record.get("artifact_sha256")
                != calibration_cache_record.get("artifact_sha256")
            ):
                raise ValueError(
                    "CorDA shared cache failed its post-use identity check"
                )
            calibration_cache_record["post_use_validation"] = {
                "status": verified_record["status"],
                "artifact_sha256": verified_record["artifact_sha256"],
            }
        if corda_cached_eigens is None:
            corda_ephemeral_files_removed = (
                _remove_corda_ephemeral_artifacts(
                    output_dir,
                    require_eigens=True,
                )
            )
        calibration_cache_record["ephemeral_files_removed"] = (
            sorted(corda_ephemeral_files_removed)
        )
        initialization = {
            "method": "corda",
            "mode": config["adapter"].get("corda_mode", "kpm"),
            "implementation": "peft-0.17.1",
            "peft_source_commit": (
                "53c25fe4fdd7c6aa4b40db0560815ea570d32303"
            ),
            "official_corda_commit": (
                "baffb03ac090f23305e5fb586a2d3c16df7f12db"
            ),
            "calibration_sampling": "official_raw_character_spans",
            "calibration_normalization": "abs(signed_global_max)",
            "covariance_dtype": "float32",
        }
        calibration_seconds = corda_calibration_seconds
    else:
        raise ValueError(f"Unsupported adapter method: {adapter_config.method}")
    adapter_setup_seconds = time.monotonic() - adapter_setup_started
    if moments is not None:
        del moments
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
                "Formal training budget drift: "
                f"actual={actual_training}, expected={expected_training}"
            )

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
                "protocol_config_hash": protocol_config_hash,
                "comparison_config_hash": comparison_config_hash,
                "source_dirty": git_state["source_dirty"],
            },
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
        "dataset": {
            "training": {
                "dataset": training_dataset_name,
                "repository": train_config.get("repository"),
                "revision": train_config["revision"],
                "file": train_config.get("file"),
                "file_size_bytes": train_config.get("file_size_bytes"),
                "file_sha256": train_config.get("file_sha256"),
                "empty_output_indices": train_config.get(
                    "empty_output_indices",
                    [],
                ),
                "rows": train_first_n,
            },
            "nq_calibration_rows": (
                calibration_samples
                if adapter_config.method in {"lora_nf", "lora_null", "corda"}
                else 0
            ),
            "nq_calibration_protocol": calibration_protocol,
            "nq_calibration_sequence_length": calibration_sequence_length,
            "response_only_loss": True,
        },
        "initialization": initialization,
        "calibration_cache": calibration_cache_record,
        "parameters": count_parameters(model),
        "timing": {
            "model_load_seconds": model_load_seconds,
            "model_integrity_validation_seconds": model_integrity_seconds,
            "data_seconds": data_seconds,
            "calibration_prepare_seconds": calibration_prepare_seconds,
            "calibration_forward_seconds": calibration_seconds,
            "calibration_cache_write_seconds": (
                calibration_cache_write_seconds
            ),
            "adapter_setup_seconds": adapter_setup_seconds,
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
        description="Unified attention-only SFT runner for LoRA baselines and LoRA-NF"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    training_data = parser.add_mutually_exclusive_group(required=True)
    training_data.add_argument("--train-json")
    training_data.add_argument(
        "--metamath-json",
        help="Legacy MetaMathQA-only alias for --train-json.",
    )
    parser.add_argument("--nq-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-smoke-fallback", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--train-first-n", type=int)
    parser.add_argument("--calibration-samples", type=int)
    parser.add_argument(
        "--calibration-cache-dir",
        help=(
            "External content-addressed moment cache; must remain outside Git "
            "artifacts."
        ),
    )
    parser.add_argument(
        "--method",
        choices=[
            "lora",
            "lora_nf",
            "lora_null",
            "milora",
            "dora",
            "pissa",
            "corda",
        ],
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    with project_gpu_lock():
        _run(args)


if __name__ == "__main__":
    main()
