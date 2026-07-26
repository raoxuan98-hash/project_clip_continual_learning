#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.trace_state_selection import select_trace_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> Dict[str, Any]:
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


def _load_completed_run(
    run_dir: Path,
    *,
    expected_state: str,
    expected_commit: str,
) -> Dict[str, Any]:
    validate_directory_integrity(
        str(run_dir),
        expected_kind="trace_run",
    )
    report_path = run_dir / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    identity = report["identity"]
    if (
        identity["state_update"] != expected_state
        or identity["commit_sha"] != expected_commit
    ):
        raise ValueError(f"TRACE state pilot identity mismatch: {run_dir}")
    return report


def _append_optional(command, flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and preregister the two LoRA-NF TRACE state pilots"
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-data-root", required=True)
    parser.add_argument("--nq-parquet", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    repository_root = package_root.parent
    source = _git_state(repository_root)
    if source["source_dirty"]:
        raise RuntimeError("TRACE state selection requires a clean source tree")

    spec_path = Path(args.spec).resolve()
    spec = load_yaml_config(str(spec_path))
    if spec.get("format_version") != 1:
        raise ValueError("Unsupported TRACE state-ablation spec version")
    if spec["selection"].get("default_on_failed_gate") != "reference_fixed":
        raise ValueError("TRACE state pilot must default to reference_fixed")
    if spec["qualification"] != {
        "data_protocol": "treelora_500_pilot",
        "execution_mode": "gpu_chain_smoke_only",
        "formal_result_eligible": False,
        "purpose": "preregister_lora_nf_continual_state_before_full_pilot",
    }:
        raise ValueError("TRACE state-ablation qualification drifted")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selection_path = output_root / "state_selection.json"
    if selection_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite TRACE state selection: {selection_path}"
        )

    runner = package_root / "scripts" / "run_trace_continual.py"
    reports = {}
    report_records = {}
    overrides = spec["pilot_overrides"]
    for state in ("reference_fixed", "reference_plus_history"):
        run_dir = output_root / state
        if run_dir.is_dir() and (run_dir / "artifact_integrity.json").is_file():
            report = _load_completed_run(
                run_dir,
                expected_state=state,
                expected_commit=source["commit_sha"],
            )
        else:
            if run_dir.exists():
                raise FileExistsError(
                    "Incomplete TRACE state pilot requires explicit audit: "
                    f"{run_dir}"
                )
            config_path = (
                spec_path.parent / spec["jobs"][state]["config"]
            ).resolve()
            command = [
                args.python,
                str(runner),
                "--config",
                str(config_path),
                "--model-path",
                str(Path(args.model_path).resolve()),
                "--trace-data-root",
                str(Path(args.trace_data_root).resolve()),
                "--nq-parquet",
                str(Path(args.nq_parquet).resolve()),
                "--output-dir",
                str(run_dir),
                "--seed",
                str(int(spec["seed"])),
                "--order",
                str(spec["order"]),
            ]
            _append_optional(
                command,
                "--train-first-n",
                overrides.get("train_first_n"),
            )
            _append_optional(
                command,
                "--test-first-n",
                overrides.get("test_first_n"),
            )
            _append_optional(
                command,
                "--max-steps",
                overrides.get("max_steps"),
            )
            _append_optional(
                command,
                "--max-new-tokens",
                overrides.get("max_new_tokens"),
            )
            _append_optional(
                command,
                "--calibration-samples",
                overrides.get("calibration_samples"),
            )
            if overrides.get("delete_checkpoint_after_smoke") is not True:
                raise ValueError("TRACE state pilot must delete smoke adapters")
            command.append("--delete-checkpoint-after-smoke")
            subprocess.run(
                command,
                cwd=str(package_root),
                check=True,
            )
            report = _load_completed_run(
                run_dir,
                expected_state=state,
                expected_commit=source["commit_sha"],
            )
        report_path = run_dir / "run_report.json"
        reports[state] = report
        report_records[state] = {
            "path": str(report_path),
            "sha256": _sha256(report_path),
            "output_integrity_manifest_sha256": _sha256(
                run_dir / "artifact_integrity.json"
            ),
        }

    rule = spec["selection"]
    selection = select_trace_state(
        reports["reference_fixed"],
        reports["reference_plus_history"],
        minimum_final_average_gain=float(
            rule["minimum_final_average_gain"]
        ),
        maximum_final_bwt_regression=float(
            rule["maximum_final_bwt_regression"]
        ),
        maximum_wall_time_ratio=float(rule["maximum_wall_time_ratio"]),
    )
    payload = {
        "format_version": 1,
        "formal_result_eligible": False,
        "purpose": spec["qualification"]["purpose"],
        "commit_sha": source["commit_sha"],
        "spec_path": str(spec_path),
        "spec_sha256": _sha256(spec_path),
        "spec_config_sha256": canonical_config_hash(spec),
        "reports": report_records,
        "selection": selection,
    }
    temporary = selection_path.with_name(f".{selection_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(selection_path)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
