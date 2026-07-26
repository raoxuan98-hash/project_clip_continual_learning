#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from llm_lora_nf.config_io import canonical_config_hash, load_yaml_config
from llm_lora_nf.integrity import validate_directory_integrity
from llm_lora_nf.trace_method_pilot import (
    TRACE_PILOT_METHODS,
    aggregate_trace_method_reports,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the locked four-method TRACE-500 pilot"
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-run-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    source = _git_state(package_root.parent)
    if source["source_dirty"]:
        raise RuntimeError("TRACE pilot aggregation requires clean source")
    spec_path = Path(args.spec).resolve()
    spec = load_yaml_config(str(spec_path))
    run_root = Path(args.run_root).resolve()
    reports = {}
    report_records = {}
    for method in TRACE_PILOT_METHODS:
        run_dir = run_root / method
        integrity = validate_directory_integrity(
            str(run_dir),
            expected_kind="trace_run",
        )
        report_path = run_dir / "run_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint_path = Path(report["checkpoint_retention"]["path"])
        if checkpoint_path.exists():
            raise ValueError(
                f"TRACE smoke adapter still exists for {method}: "
                f"{checkpoint_path}"
            )
        reports[method] = report
        report_records[method] = {
            "path": str(report_path),
            "sha256": _sha256(report_path),
            "integrity_manifest_sha256": _sha256(
                run_dir / "artifact_integrity.json"
            ),
            "integrity_tree_sha256": integrity["tree_sha256"],
        }
    aggregate = aggregate_trace_method_reports(
        spec,
        reports,
        expected_run_commit=args.expected_run_commit,
    )
    payload = {
        "format_version": 1,
        "aggregator_source": source,
        "spec_path": str(spec_path),
        "spec_sha256": _sha256(spec_path),
        "spec_config_sha256": canonical_config_hash(spec),
        "run_root": str(run_root),
        "reports": report_records,
        **aggregate,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite TRACE summary: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "comparison_config_hash": payload[
                    "comparison_config_hash"
                ],
                "ranking_by_final_average": payload[
                    "ranking_by_final_average"
                ],
                "rows": payload["rows"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
