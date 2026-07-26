#!/usr/bin/env python3
"""Execute EvalPlus samples inside a networkless Bubblewrap sandbox."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch

from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.environment import (
    code_evaluation_package_versions,
    runtime_environment,
    validate_code_evaluation_package_versions,
)
from llm_lora_nf.evalplus_evidence import validate_evalplus_data_manifest
from llm_lora_nf.evalplus_results import (
    validate_evalplus_result,
    validate_evalplus_samples,
)
from llm_lora_nf.integrity import (
    validate_directory_integrity,
    write_directory_integrity,
)
from llm_lora_nf.protocol_validation import (
    validate_formal_code_evaluation_config,
)
from llm_lora_nf.resource_guard import project_file_lock


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}")
    return payload


def _git_state(path: Path, *, scope: str = ".") -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", scope],
            cwd=str(path),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit_sha": commit, "source_dirty": dirty}


def _bubblewrap_base(
    *,
    executable: Path,
    evaluator_path: Path,
    output_dir: Path,
    data: Dict[str, Any],
    config: Dict[str, Any],
) -> list[str]:
    base_environment = Path(os.path.realpath(sys.executable)).parents[1]
    virtual_environment = Path(sys.prefix).resolve()
    site_packages = (
        f"/opt/venv/lib/python{sys.version_info.major}."
        f"{sys.version_info.minor}/site-packages"
    )
    humaneval_path = Path(
        data["datasets"]["humaneval"]["jsonl"]["path"]
    ).resolve()
    mbpp_path = Path(data["datasets"]["mbpp"]["jsonl"]["path"]).resolve()
    command = [
        str(executable),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc",
        "/etc",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/tmp/home",
        "--dir",
        "/opt",
        "--ro-bind",
        str(base_environment),
        "/opt/baseenv",
        "--ro-bind",
        str(virtual_environment),
        "/opt/venv",
        "--ro-bind",
        str(evaluator_path),
        "/opt/evalplus",
        "--dir",
        "/datasets",
        "--ro-bind",
        str(humaneval_path),
        "/datasets/HumanEvalPlus.jsonl",
        "--ro-bind",
        str(mbpp_path),
        "/datasets/MbppPlus.jsonl",
        "--bind",
        str(output_dir),
        "/work",
        "--setenv",
        "HOME",
        "/tmp/home",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--setenv",
        "CUDA_VISIBLE_DEVICES",
        "",
        "--setenv",
        "PYTHONPATH",
        f"/opt/evalplus:{site_packages}",
        "--setenv",
        "HF_HUB_OFFLINE",
        "1",
        "--setenv",
        "TRANSFORMERS_OFFLINE",
        "1",
        "--setenv",
        "TOKENIZERS_PARALLELISM",
        "false",
        "--setenv",
        "EVALPLUS_MAX_MEMORY_BYTES",
        str(config["execution"]["memory_limit_bytes"]),
        "--setenv",
        "HUMANEVAL_OVERRIDE_PATH",
        "/datasets/HumanEvalPlus.jsonl",
        "--setenv",
        "MBPP_OVERRIDE_PATH",
        "/datasets/MbppPlus.jsonl",
        "--chdir",
        "/work",
    ]
    return command


def _run(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.config)
    if args.seed is not None:
        config["run"]["seed"] = args.seed
    validate_formal_code_evaluation_config(config)
    config_hash = canonical_config_hash(config)
    protocol = copy.deepcopy(config)
    protocol["run"].pop("seed", None)
    protocol_hash = canonical_config_hash(protocol)

    repo_root = Path(__file__).resolve().parents[2]
    source = _git_state(repo_root, scope="llm_lora_nf")
    evaluator_path = Path(args.evaluator_path).resolve()
    evaluator = _git_state(evaluator_path)
    if (
        source["source_dirty"]
        or evaluator["source_dirty"]
        or evaluator["commit_sha"] != config["evaluator"]["revision"]
    ):
        raise RuntimeError(
            "Formal EvalPlus execution requires clean, exact source revisions"
        )
    data = validate_evalplus_data_manifest(
        args.dataset_manifest,
        config=config,
    )

    generation_dir = Path(args.generation_dir).resolve()
    generation_integrity = validate_directory_integrity(
        str(generation_dir),
        expected_kind="code_generation_output",
    )
    generation = _load_json(
        generation_dir / "code_generation_run.json"
    )
    requested_formal = generation.get("formal_result_eligible", False)
    if requested_formal:
        if (
            generation.get("execution_mode") != "gpu_formal"
            or generation.get("status") != "generated"
            or generation.get("source") != source
            or generation.get("evaluator", {}).get("commit_sha")
            != evaluator["commit_sha"]
            or generation.get("protocol_config_hash") != protocol_hash
            or generation.get("seed") != int(config["run"]["seed"])
            or generation.get("dataset_evidence") != data
        ):
            raise ValueError("Code generation evidence/protocol mismatch")
    elif not args.allow_cpu_smoke:
        raise ValueError(
            "CPU smoke generation cannot enter formal execution"
        )

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ("humaneval", "mbpp"):
        source_samples = generation_dir / f"{dataset}_samples.jsonl"
        destination = output_dir / f"{dataset}_samples.jsonl"
        shutil.copy2(source_samples, destination)
        copied = validate_evalplus_samples(
            destination,
            dataset_path=Path(data["datasets"][dataset]["jsonl"]["path"]),
            allow_subset=not requested_formal,
        )
        if copied["sha256"] != generation["samples"][dataset]["sha256"]:
            raise ValueError(f"Copied sample identity mismatch: {dataset}")

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("Bubblewrap is required for untrusted code execution")
    bwrap_path = Path(bwrap).resolve()
    version = subprocess.run(
        [str(bwrap_path), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    base_command = _bubblewrap_base(
        executable=bwrap_path,
        evaluator_path=evaluator_path,
        output_dir=output_dir,
        data=data,
        config=config,
    )
    environment = runtime_environment(
        torch.device("cpu"),
        validate_formal=requested_formal,
    )
    evalplus_packages = code_evaluation_package_versions()
    validate_code_evaluation_package_versions(evalplus_packages)

    started = time.monotonic()
    commands = {}
    results = {}
    for dataset in ("humaneval", "mbpp"):
        command = [
            *base_command,
            "/opt/baseenv/bin/python",
            "/opt/evalplus/evalplus/evaluate.py",
            f"--dataset={dataset}",
            f"--samples=/work/{dataset}_samples.jsonl",
            f"--parallel={config['execution']['parallel_workers']}",
            (
                "--min-time-limit="
                f"{config['execution']['min_time_limit_seconds']}"
            ),
            (
                "--gt-time-limit-factor="
                f"{config['execution']['ground_truth_time_limit_factor']}"
            ),
        ]
        commands[dataset] = command
        stdout_path = output_dir / f"{dataset}_sandbox.stdout.log"
        stderr_path = output_dir / f"{dataset}_sandbox.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr:
            completed = subprocess.run(
                command,
                stdout=stdout,
                stderr=stderr,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Sandboxed EvalPlus {dataset} failed with "
                f"return code {completed.returncode}"
            )
        samples_path = output_dir / f"{dataset}_samples.jsonl"
        post_samples = validate_evalplus_samples(
            samples_path,
            dataset_path=Path(data["datasets"][dataset]["jsonl"]["path"]),
            allow_subset=not requested_formal,
        )
        if post_samples["sha256"] != generation["samples"][dataset]["sha256"]:
            raise RuntimeError(
                f"Sandbox modified the immutable sample evidence: {dataset}"
            )
        if requested_formal:
            result_path = (
                output_dir / f"{dataset}_samples_eval_results.json"
            )
            results[dataset] = validate_evalplus_result(
                result_path,
                dataset=dataset,
                dataset_path=Path(
                    data["datasets"][dataset]["jsonl"]["path"]
                ),
                expected_samples=1,
            )

    metrics = (
        {
            "humaneval_pass_at_1": results["humaneval"]["base_pass_at_1"],
            "humaneval_plus_pass_at_1": results["humaneval"]["plus_pass_at_1"],
            "mbpp_pass_at_1": results["mbpp"]["base_pass_at_1"],
            "mbpp_plus_pass_at_1": results["mbpp"]["plus_pass_at_1"],
        }
        if requested_formal
        else {}
    )
    report = {
        "format_version": 1,
        "status": "passed",
        "formal_result_eligible": requested_formal,
        "execution_mode": (
            "cpu_sandbox_formal" if requested_formal else "cpu_smoke_only"
        ),
        "source": source,
        "evaluator": {
            **evaluator,
            "repository": config["evaluator"]["repository"],
            "version": config["evaluator"]["version"],
        },
        "config_hash": config_hash,
        "protocol_config_hash": protocol_hash,
        "seed": int(config["run"]["seed"]),
        "generation_dir": str(generation_dir),
        "generation_integrity": generation_integrity,
        "generation_report": generation,
        "dataset_evidence": data,
        "sandbox": {
            "runtime": "bubblewrap",
            "path": str(bwrap_path),
            "version": version,
            "unshare_all": True,
            "network": "disabled",
            "commands": commands,
        },
        "results": results,
        "metrics": metrics,
        "environment": environment,
        "code_evaluation_packages": evalplus_packages,
        "wall_seconds": time.monotonic() - started,
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "code_execution_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    integrity = write_directory_integrity(
        str(output_dir),
        kind="code_execution_output",
    )
    print(
        json.dumps(
            {"report": report, "integrity": integrity},
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run official EvalPlus under a networkless Bubblewrap sandbox"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluator-path", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    with project_file_lock("/tmp/llm_lora_nf_evalplus_sandbox.lock"):
        _run(args)


if __name__ == "__main__":
    main()
