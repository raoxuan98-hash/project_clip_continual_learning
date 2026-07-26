#!/usr/bin/env python3
import argparse
import copy
import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from llm_lora_nf.baselines import build_native_adapter
from llm_lora_nf.calibration import ActivationCalibrator, build_and_assign_filters
from llm_lora_nf.checkpoint import load_native_adapter, save_native_adapter
from llm_lora_nf.config_io import (
    adapter_config_from_mapping,
    canonical_config_hash,
    load_yaml_config,
)
from llm_lora_nf.data import (
    ResponseOnlyCollator,
    SupervisedChatDataset,
    synthetic_smoke_examples,
)
from llm_lora_nf.inject import count_parameters
from llm_lora_nf.integrity import (
    validate_directory_integrity,
    write_directory_integrity,
)
from llm_lora_nf.model_io import load_instruct_model
from llm_lora_nf.resource_guard import inspect_admission, project_gpu_lock
from llm_lora_nf.result_metadata import RunMetadata
from llm_lora_nf.training import set_reproducible_seed, train_steps


def _git_state() -> Dict[str, Any]:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--", "llm_lora_nf"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "commit_sha": commit_result.stdout.strip(),
        "source_dirty": bool(status_result.stdout.strip()),
    }


def _model_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value for key, value in batch.items() if key != "labels"}


@torch.no_grad()
def _logits(model: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    return model(**_model_inputs(batch)).logits.detach().cpu()


def _run(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    git_state = _git_state()
    run_config = config["run"]
    smoke_config = config["smoke"]
    model_config = config["model"]
    if run_config["execution_mode"] != "cpu_smoke_only":
        raise ValueError("run_smoke.py only accepts execution_mode=cpu_smoke_only")
    if smoke_config.get("collect_formal_metrics", False):
        raise ValueError("CPU smoke must not collect formal metrics")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty smoke output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    admission = inspect_admission(requested=0)
    device = torch.device("cpu")
    seed = int(run_config["seed"])
    set_reproducible_seed(seed)
    adapter_config = adapter_config_from_mapping(config["adapter"])
    model_integrity = validate_directory_integrity(
        args.model_path,
        expected_kind="model_snapshot",
    )
    model, tokenizer, load_record = load_instruct_model(
        requested_id=model_config["id"],
        source_id=model_config.get("source_id", model_config["id"]),
        resolved_path=args.model_path,
        checkpoint_type=model_config["checkpoint_type"],
        device=str(device),
        local_files_only=True,
    )
    examples = synthetic_smoke_examples(int(smoke_config["synthetic_examples"]))
    dataset = SupervisedChatDataset(
        examples,
        tokenizer,
        max_length=int(smoke_config["max_sequence_length"]),
        enable_thinking=bool(model_config.get("enable_thinking", False)),
    )
    batches = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=ResponseOnlyCollator(tokenizer.pad_token_id),
    )
    first_batch = next(iter(batches))
    baseline_logits = _logits(model, first_batch)

    moments = ActivationCalibrator(model).collect(
        batches,
        device=device,
        max_batches=1,
    )
    _, initialization = build_native_adapter(
        model,
        adapter_config,
        calibration_moments=moments,
    )
    filter_results = build_and_assign_filters(
        model,
        moments,
        energy_fraction=adapter_config.filter.energy_fraction,
        leakage=adapter_config.filter.leakage,
        ridge=adapter_config.filter.ridge,
    )
    initialized_logits = _logits(model, first_batch)
    torch.testing.assert_close(
        initialized_logits,
        baseline_logits,
        atol=1e-5,
        rtol=1e-4,
    )

    training_summary = train_steps(
        model,
        batches,
        device=device,
        max_steps=int(smoke_config["max_steps"]),
        learning_rate=2e-5,
    )
    trained_logits = _logits(model, first_batch)

    metadata = RunMetadata(
        run_id=run_config["name"],
        execution_mode="cpu_smoke_only",
        method=adapter_config.method,
        model_id=model_config["id"],
        seed=seed,
        commit_sha=git_state["commit_sha"],
    )
    save_native_adapter(
        model,
        str(output_dir / "checkpoint"),
        adapter_config=adapter_config,
        metadata={
            **metadata.to_dict(),
            "config_hash": canonical_config_hash(config),
            "source_dirty": git_state["source_dirty"],
        },
    )
    checkpoint_integrity = write_directory_integrity(
        str(output_dir / "checkpoint"),
        kind="adapter_checkpoint",
    )
    del model
    gc.collect()

    reloaded_model, reloaded_tokenizer, _ = load_instruct_model(
        requested_id=model_config["id"],
        source_id=model_config.get("source_id", model_config["id"]),
        resolved_path=args.model_path,
        checkpoint_type=model_config["checkpoint_type"],
        device=str(device),
        local_files_only=True,
    )
    validate_directory_integrity(
        str(output_dir / "checkpoint"),
        expected_kind="adapter_checkpoint",
    )
    load_native_adapter(
        reloaded_model,
        str(output_dir / "checkpoint"),
        adapter_config=adapter_config,
    )
    reloaded_logits = _logits(reloaded_model, first_batch)
    torch.testing.assert_close(
        reloaded_logits,
        trained_logits,
        atol=1e-5,
        rtol=1e-4,
    )

    prompt = reloaded_tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with exactly: smoke-ok"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(model_config.get("enable_thinking", False)),
    )
    generation_inputs = reloaded_tokenizer(prompt, return_tensors="pt")
    from transformers import GenerationConfig

    generation_config = GenerationConfig(
        max_new_tokens=8,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        bos_token_id=reloaded_model.generation_config.bos_token_id,
        eos_token_id=reloaded_model.generation_config.eos_token_id,
        pad_token_id=reloaded_tokenizer.pad_token_id,
    )
    with torch.no_grad():
        output_ids = reloaded_model.generate(
            **generation_inputs,
            generation_config=generation_config,
        )
    generated_text = reloaded_tokenizer.decode(
        output_ids[0, generation_inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    report = {
        "status": "passed",
        "metadata": metadata.to_dict(),
        "source_dirty": git_state["source_dirty"],
        "config_hash": canonical_config_hash(config),
        "admission": {
            "mode": admission.mode,
            "selected_gpu_indices": admission.selected_gpu_indices,
            "idle_gpu_indices": admission.idle_gpu_indices,
            "reason": admission.reason,
        },
        "model_integrity": model_integrity,
        "model": load_record.to_dict(),
        "initialization": initialization.to_dict(),
        "parameters": count_parameters(reloaded_model),
        "calibration": {
            group: {
                "observations": moments[group].observations,
                "tail_dimension": result.tail_dimension,
                "protected_dimension": result.protected_dimension,
                "captured_tail_energy": result.captured_tail_energy,
            }
            for group, result in filter_results.items()
        },
        "checks": {
            "checkpoint_function_preserved": True,
            "one_optimizer_step": True,
            "save_load_logits_equal": True,
            "response_only_labels": True,
            "formal_result_eligible": False,
        },
        "training": training_summary.to_dict(),
        "checkpoint_integrity": checkpoint_integrity,
        "generation_text": generated_text,
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote-only Qwen/Llama smoke run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with project_gpu_lock():
        _run(args)


if __name__ == "__main__":
    main()
