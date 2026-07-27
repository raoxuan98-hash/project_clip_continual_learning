#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping

from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.trace_filter_ablation import (
    TRACE_FILTER_SETTINGS,
    aggregate_trace_filter_reports,
)


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


def _read_report(
    run_dir: Path,
    *,
    label: str,
) -> tuple[Mapping[str, Any], Dict[str, Any]]:
    integrity = validate_directory_integrity(
        str(run_dir),
        expected_kind="trace_run",
    )
    report_path = run_dir / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(report["checkpoint_retention"]["path"])
    if checkpoint_path.exists():
        raise ValueError(
            f"TRACE filter smoke adapter still exists for {label}: "
            f"{checkpoint_path}"
        )
    record = {
        "path": str(report_path),
        "sha256": _sha256(report_path),
        "integrity_manifest_sha256": _sha256(
            run_dir / "artifact_integrity.json"
        ),
        "integrity_tree_sha256": integrity["tree_sha256"],
    }
    return report, record


def _validate_filter_config(
    *,
    label: str,
    config: Mapping[str, Any],
    setting: Mapping[str, Any],
) -> None:
    if config["adapter"]["method"] != "lora_nf":
        raise ValueError(f"TRACE filter method config drifted: {label}")
    if config["trace"]["state_update"] != "reference_fixed":
        raise ValueError(f"TRACE filter state config drifted: {label}")
    filter_config = config["adapter"]["filter"]
    if float(filter_config["energy_fraction"]) != float(
        setting["energy_fraction"]
    ):
        raise ValueError(f"TRACE filter energy config drifted: {label}")
    if float(filter_config["leakage"]) != float(setting["leakage"]):
        raise ValueError(f"TRACE filter leakage config drifted: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, aggregate, and select the TRACE filter pilot"
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--lora-run-dir", required=True)
    parser.add_argument("--default-run-dir", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--expected-baseline-commit", required=True)
    parser.add_argument("--expected-candidate-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    repo_root = package_root.parent
    source = _git_state(repo_root)
    if source["source_dirty"]:
        raise RuntimeError(
            "TRACE filter aggregation requires clean LLM source"
        )
    spec_path = Path(args.spec).resolve()
    spec = load_yaml_config(str(spec_path))
    config_root = package_root / "configs" / "continual"

    lora_config = load_yaml_config(
        str(config_root / "trace_qwen3_0p6b_pilot_lora.yaml")
    )
    if lora_config["adapter"]["method"] != "lora":
        raise ValueError("TRACE filter LoRA reference config drifted")
    configs = {
        "e20_r02": load_yaml_config(
            str(
                config_root
                / "trace_qwen3_0p6b_pilot_lora_nf_fixed.yaml"
            )
        )
    }
    for setting in TRACE_FILTER_SETTINGS:
        setting_spec = spec["settings"][setting]
        if setting != "e20_r02":
            config_name = setting_spec.get("config")
            if not config_name:
                raise ValueError(
                    f"TRACE filter candidate lacks config: {setting}"
                )
            configs[setting] = load_yaml_config(
                str(config_root / str(config_name))
            )
        _validate_filter_config(
            label=setting,
            config=configs[setting],
            setting=setting_spec,
        )

    lora_report, lora_record = _read_report(
        Path(args.lora_run_dir).resolve(),
        label="lora",
    )
    default_report, default_record = _read_report(
        Path(args.default_run_dir).resolve(),
        label="e20_r02",
    )
    candidate_root = Path(args.candidate_root).resolve()
    reports = {"e20_r02": default_report}
    report_records = {"lora": lora_record, "e20_r02": default_record}
    for setting in TRACE_FILTER_SETTINGS:
        if setting == "e20_r02":
            continue
        report, record = _read_report(
            candidate_root / setting,
            label=setting,
        )
        reports[setting] = report
        report_records[setting] = record

    expected_run_commits = {
        "lora": args.expected_baseline_commit,
        "e20_r02": args.expected_baseline_commit,
        **{
            setting: args.expected_candidate_commit
            for setting in TRACE_FILTER_SETTINGS
            if setting != "e20_r02"
        },
    }
    expected_config_hashes = {
        "lora": canonical_config_hash(lora_config),
        **{
            setting: canonical_config_hash(config)
            for setting, config in configs.items()
        },
    }
    aggregate = aggregate_trace_filter_reports(
        spec,
        lora_report,
        reports,
        expected_run_commits=expected_run_commits,
        expected_config_hashes=expected_config_hashes,
    )
    payload = {
        "format_version": 1,
        "aggregator_source": source,
        "spec_path": str(spec_path),
        "spec_sha256": _sha256(spec_path),
        "spec_config_sha256": canonical_config_hash(spec),
        "candidate_root": str(candidate_root),
        "expected_run_commits": expected_run_commits,
        "expected_config_hashes": expected_config_hashes,
        "reports": report_records,
        **aggregate,
    }

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite TRACE filter selection: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "reference_lora": payload["reference_lora"],
                "rows": payload["rows"],
                "selection": payload["selection"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
