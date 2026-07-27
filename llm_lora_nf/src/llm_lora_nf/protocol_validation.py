from typing import Any, Mapping, Sequence

from .dataset_io import (
    CODEFEEDBACK_SOURCE_ID,
    CODEFEEDBACK_SOURCE_REVISION,
    METAMATH_REVISION,
    NQ_OPEN_REVISION,
    PISSA_CODEFEEDBACK_PYTHON_FILE,
    PISSA_CODEFEEDBACK_PYTHON_EMPTY_OUTPUT_INDICES,
    PISSA_CODEFEEDBACK_PYTHON_ROWS,
    PISSA_CODEFEEDBACK_PYTHON_SHA256,
    PISSA_CODEFEEDBACK_PYTHON_SIZE_BYTES,
    PISSA_DATASET_ID,
    PISSA_DATASET_REVISION,
)
from .evalplus_evidence import EVALPLUS_DATASETS


def _nested(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Locked formal config is missing {path}")
        value = value[key]
    return value


def _expect(config: Mapping[str, Any], path: str, expected: Any) -> None:
    actual = _nested(config, path)
    if actual != expected:
        raise ValueError(
            f"Locked formal field {path} must be {expected!r}, got {actual!r}"
        )


def _expect_float(
    config: Mapping[str, Any],
    path: str,
    expected: float,
    *,
    tolerance: float = 1e-12,
) -> None:
    actual = float(_nested(config, path))
    if abs(actual - expected) > tolerance:
        raise ValueError(
            f"Locked formal field {path} must be {expected!r}, got {actual!r}"
        )


def _validate_common_sft_training(
    config: Mapping[str, Any],
    *,
    target_modules: Sequence[str],
) -> None:
    _expect(config, "artifact_retention.base_snapshots_per_revision", 1)
    _expect(
        config,
        "artifact_retention.final_adapter_checkpoints_per_run",
        1,
    )
    _expect(
        config,
        "artifact_retention.save_intermediate_checkpoints",
        False,
    )
    _expect(config, "artifact_retention.save_optimizer_state", False)
    _expect(config, "artifact_retention.save_scheduler_state", False)
    _expect(
        config,
        "artifact_retention.merged_checkpoint_policy",
        "ephemeral_delete_after_qualified_evaluation",
    )
    _expect(
        config,
        "artifact_retention.max_concurrent_merged_checkpoints",
        1,
    )
    _expect(
        config,
        "artifact_retention.request_cache_scope",
        "shared_content_addressed_per_base_model_protocol",
    )
    _expect(config, "run.execution_mode", "gpu_formal")
    _expect(config, "model.checkpoint_type", "instruct")
    _expect(config, "model.local_files_only", True)
    _expect(config, "model.enable_thinking", False)
    _expect(config, "adapter.rank", 128)
    _expect_float(config, "adapter.alpha", 128.0)
    _expect_float(config, "adapter.dropout", 0.0)
    _expect(config, "adapter.target_modules", list(target_modules))
    _expect(config, "calibration.dataset", "nq_open")
    _expect(config, "calibration.revision", NQ_OPEN_REVISION)
    _expect(config, "calibration.samples", 256)
    _expect(config, "calibration.seed", 233)
    _expect(config, "train.max_sequence_length", 512)
    _expect(config, "train.epochs", 1)
    _expect(config, "train.per_device_batch_size", 1)
    _expect(config, "train.global_batch_size", 128)
    _expect_float(config, "train.adam_beta1", 0.9)
    _expect_float(config, "train.adam_beta2", 0.999)
    _expect_float(config, "train.adam_epsilon", 1e-8)
    _expect_float(config, "train.max_grad_norm", 1.0)
    _expect_float(config, "train.learning_rate", 2e-5)
    _expect_float(config, "train.weight_decay", 0.0)
    _expect(config, "train.scheduler", "cosine")
    _expect_float(config, "train.warmup_ratio", 0.03)
    _expect(config, "train.response_only_loss", True)
    _expect(config, "train.precision", "bf16")
    _expect(config, "train.tf32", True)
    _expect(config, "train.gradient_checkpointing", True)
    _expect(config, "train.reference_trainer_version", "4.47.0")
    _expect(
        config,
        "train.reference_trainer_commit",
        "5d7739f15a6e50de416977fe2cc9cb516d67edda",
    )
    _expect(config, "train.accumulation_remainder_policy", "drop")
    seed = int(_nested(config, "run.seed"))
    if seed not in {42, 43, 44}:
        raise ValueError("Locked formal seeds are 42, 43, and 44")


def _validate_metamath_training(config: Mapping[str, Any]) -> None:
    _expect(config, "train.dataset", "metamathqa")
    _expect(config, "train.revision", METAMATH_REVISION)
    _expect(config, "train.first_n", 100000)
    _expect(config, "train.optimizer_steps_per_epoch", 781)
    _expect(config, "train.consumed_examples_per_epoch", 99968)
    _expect(config, "train.discarded_examples_per_epoch", 32)


def _validate_codefeedback_python_training(config: Mapping[str, Any]) -> None:
    _expect(config, "train.dataset", "pissa_codefeedback_python")
    _expect(config, "train.repository", PISSA_DATASET_ID)
    _expect(config, "train.revision", PISSA_DATASET_REVISION)
    _expect(config, "train.file", PISSA_CODEFEEDBACK_PYTHON_FILE)
    _expect(
        config,
        "train.file_size_bytes",
        PISSA_CODEFEEDBACK_PYTHON_SIZE_BYTES,
    )
    _expect(config, "train.file_sha256", PISSA_CODEFEEDBACK_PYTHON_SHA256)
    _expect(
        config,
        "train.empty_output_indices",
        list(PISSA_CODEFEEDBACK_PYTHON_EMPTY_OUTPUT_INDICES),
    )
    _expect(config, "train.source_dataset", CODEFEEDBACK_SOURCE_ID)
    _expect(config, "train.source_revision", CODEFEEDBACK_SOURCE_REVISION)
    _expect(config, "train.first_n", PISSA_CODEFEEDBACK_PYTHON_ROWS)
    _expect(config, "train.optimizer_steps_per_epoch", 819)
    _expect(config, "train.consumed_examples_per_epoch", 104832)
    _expect(config, "train.discarded_examples_per_epoch", 16)


def _validate_track_b_training_dataset(config: Mapping[str, Any]) -> None:
    dataset = str(_nested(config, "train.dataset"))
    if dataset == "metamathqa":
        _validate_metamath_training(config)
    elif dataset == "pissa_codefeedback_python":
        _validate_codefeedback_python_training(config)
    else:
        raise ValueError(f"Unsupported locked Track B dataset: {dataset}")


def validate_track_b_formal_config(config: Mapping[str, Any]) -> None:
    _validate_common_sft_training(
        config,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    _validate_track_b_training_dataset(config)
    _expect(config, "train.truncation_strategy", "left_preserve_response")
    _expect(config, "train.optimizer", "adamw")
    _expect(config, "calibration.batch_size", 8)
    _expect(config, "calibration.max_sequence_length", 1024)
    _expect(config, "calibration.include_non_padding_tokens", True)
    _expect(
        config,
        "calibration.lora_null_sampling",
        "raw_character_spans",
    )
    _expect(config, "calibration.lora_null_max_sequence_length", 2048)
    _expect(
        config,
        "calibration.lora_null_normalization",
        "abs_of_signed_global_max",
    )
    _expect(config, "calibration.corda_sampling", "raw_character_spans")
    _expect(config, "calibration.corda_max_sequence_length", 2048)
    _expect(
        config,
        "calibration.corda_normalization",
        "abs_of_signed_global_max",
    )
    _expect(config, "calibration.corda_covariance_dtype", "float32")
    _expect_float(config, "adapter.filter.energy_fraction", 0.20)
    _expect_float(config, "adapter.filter.leakage", 0.02)
    _expect_float(config, "adapter.filter.ridge", 0.0001)
    method = str(_nested(config, "adapter.method"))
    if method not in {
        "lora",
        "dora",
        "lora_null",
        "pissa",
        "milora",
        "corda",
        "lora_nf",
    }:
        raise ValueError(f"Unsupported locked Track B method: {method}")
    if method == "corda":
        _expect(config, "adapter.corda_mode", "kpm")


def validate_track_a_formal_config(config: Mapping[str, Any]) -> None:
    _expect(config, "protocol_track", "A")
    _expect(
        config,
        "official_commit",
        "1e6808abb81fe10e50b8172c40ac9a8ab4f11e83",
    )
    _validate_common_sft_training(
        config,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    )
    _validate_metamath_training(config)
    _expect(config, "train.optimizer", "adamw_torch")
    _expect(config, "train.prompt_format", "official_lora_null_alpaca_style")
    _expect(config, "train.truncation_strategy", "official_right")
    _expect(config, "calibration.batch_size", 1)
    _expect(config, "calibration.max_sequence_length", 2048)
    _expect(config, "calibration.sampling", "raw_character_spans")
    _expect(
        config,
        "calibration.per_sample_normalization",
        "abs_of_signed_global_max",
    )
    _expect(config, "calibration.sample_divisor", 256)
    _expect(config, "model.null_calibration_activation_dtype", "float16")
    _expect(config, "model.null_decomposition_install_dtype", "float32")
    _expect(config, "model.training_parameter_dtype", "float32")
    method = str(_nested(config, "adapter.method"))
    if method not in {"lora", "lora_null"}:
        raise ValueError(f"Unsupported locked Track A method: {method}")


def validate_formal_evaluation_config(config: Mapping[str, Any]) -> None:
    _expect(config, "run.execution_mode", "gpu_formal")
    seed = int(_nested(config, "run.seed"))
    if seed not in {42, 43, 44}:
        raise ValueError("Locked formal evaluation seeds are 42, 43, and 44")
    _expect(
        config,
        "evaluator.repository",
        "EleutherAI/lm-evaluation-harness",
    )
    _expect(
        config,
        "evaluator.revision",
        "6d642546f4688648fced259eb3302efd36ece5af",
    )
    _expect(config, "evaluator.version", "v0.4.12")
    _expect(
        config,
        "evaluator.seeds",
        {
            "random": 0,
            "numpy": 1234,
            "torch": 1234,
            "fewshot": 1234,
        },
    )
    _expect(config, "prompt.apply_chat_template", True)
    _expect(config, "prompt.fewshot_as_multiturn", True)
    _expect(config, "prompt.enable_thinking", False)
    _expect(
        config,
        "tasks.downstream",
        ["gsm8k_cot", "minerva_math500"],
    )
    _expect(
        config,
        "tasks.retention",
        ["triviaqa", "nq_open", "webqs"],
    )
    locked_metrics = {
        "gsm8k_cot": "exact_match,flexible-extract",
        "minerva_math500": "math_verify,none",
        "triviaqa": "exact_match,remove_whitespace",
        "nq_open": "exact_match,remove_whitespace",
        "webqs": "exact_match,none",
    }
    _expect(config, "metrics", locked_metrics)
    _expect(config, "runtime.batch_size", "auto")
    _expect(config, "runtime.max_batch_size", 8)
    _expect(config, "runtime.dtype", "float32")
    _expect(config, "runtime.log_samples", True)
    _expect(config, "runtime.cache_requests", True)
    locked_datasets = {
        "openai/gsm8k": {
            "config": "main",
            "revision": "740312add88f781978c0658806c59bc2815b9866",
        },
        "HuggingFaceH4/MATH-500": {
            "config": "default",
            "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        },
        "mandarjoshi/trivia_qa": {
            "config": "rc.nocontext",
            "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
        },
        "google-research-datasets/nq_open": {
            "config": None,
            "revision": NQ_OPEN_REVISION,
        },
        "web_questions": {
            "config": None,
            "revision": "0e473cbe21d1e91ec18da343644498be6a3f5454",
        },
    }
    _expect(config, "datasets", locked_datasets)


def validate_formal_code_evaluation_config(
    config: Mapping[str, Any],
) -> None:
    _expect(config, "run.execution_mode", "gpu_formal")
    seed = int(_nested(config, "run.seed"))
    if seed not in {42, 43, 44}:
        raise ValueError("Locked formal code-evaluation seeds are 42, 43, and 44")
    _expect(config, "evaluator.repository", "evalplus/evalplus")
    _expect(config, "evaluator.version", "v0.3.1")
    _expect(
        config,
        "evaluator.revision",
        "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2",
    )
    _expect(config, "datasets", EVALPLUS_DATASETS)
    _expect(config, "generation.backend", "hf")
    _expect(config, "generation.checkpoint_type", "instruct")
    _expect(config, "generation.apply_chat_template", True)
    _expect(config, "generation.force_base_prompt", False)
    _expect(config, "generation.greedy", True)
    _expect_float(config, "generation.temperature", 0.0)
    _expect(config, "generation.num_samples", 1)
    _expect(config, "generation.batch_size", 1)
    _expect(config, "generation.max_new_tokens", 768)
    _expect(config, "generation.dtype", "float32")
    _expect(config, "generation.attn_implementation", "eager")
    _expect(config, "generation.trust_remote_code", False)
    _expect(config, "generation.resume", False)
    _expect(config, "generation.enable_thinking", False)
    _expect(
        config,
        "generation.instruction_prefix",
        (
            "Please provide a self-contained Python script that solves the "
            "following problem in a markdown code block:"
        ),
    )
    _expect(
        config,
        "generation.response_prefix",
        (
            "Below is a Python script with a self-contained function that "
            "solves the problem and passes corresponding tests:"
        ),
    )
    _expect(config, "execution.runtime", "bubblewrap")
    _expect(config, "execution.unshare_all", True)
    _expect(config, "execution.network", "disabled")
    _expect(config, "execution.parallel_workers", 8)
    _expect(config, "execution.base_only", False)
    _expect(config, "execution.test_details", False)
    _expect_float(config, "execution.min_time_limit_seconds", 1.0)
    _expect_float(config, "execution.ground_truth_time_limit_factor", 4.0)
    _expect(config, "execution.memory_limit_bytes", 4294967296)
    _expect(config, "metrics.samples_per_task", 1)
    _expect(
        config,
        "metrics.primary",
        [
            "humaneval_pass_at_1",
            "humaneval_plus_pass_at_1",
            "mbpp_pass_at_1",
            "mbpp_plus_pass_at_1",
        ],
    )
