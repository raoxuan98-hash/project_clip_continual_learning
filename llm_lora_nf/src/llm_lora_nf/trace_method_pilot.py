from typing import Any, Dict, Mapping


TRACE_PILOT_METHODS = ("lora", "dora", "lora_null", "lora_nf")


def aggregate_trace_method_reports(
    spec: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    *,
    expected_run_commit: str,
) -> Dict[str, Any]:
    if tuple(spec.get("jobs", ())) != TRACE_PILOT_METHODS:
        raise ValueError("TRACE method pilot must contain four locked methods")
    if set(reports) != set(TRACE_PILOT_METHODS):
        raise ValueError("TRACE method pilot report set is incomplete")
    qualification = spec.get("qualification")
    if qualification != {
        "data_protocol": "treelora_500_pilot",
        "execution_mode": "gpu_chain_smoke_only",
        "formal_result_eligible": False,
        "purpose": "controlled_four_method_trace500_pilot_after_state_selection",
    }:
        raise ValueError("TRACE method pilot qualification drifted")
    overrides = spec["pilot_overrides"]
    expected_limits = {
        "tasks": 8,
        "train_first_n": overrides["train_first_n"],
        "test_first_n": overrides["test_first_n"],
        "max_steps": overrides["max_steps"],
        "generation_max_new_tokens": overrides["max_new_tokens"],
        "training_max_sequence_length": 1536,
        "generation_max_prompt_length": 1024,
    }
    comparison_hashes = set()
    rows = []
    for method in TRACE_PILOT_METHODS:
        report = reports[method]
        identity = report["identity"]
        if identity["method"] != method:
            raise ValueError(f"TRACE method identity mismatch: {method}")
        if identity["commit_sha"] != expected_run_commit:
            raise ValueError(f"TRACE run commit mismatch: {method}")
        if identity["seed"] != int(spec["seed"]):
            raise ValueError(f"TRACE seed mismatch: {method}")
        if identity["order"] != spec["order"]:
            raise ValueError(f"TRACE order mismatch: {method}")
        if identity["execution_mode"] != qualification["execution_mode"]:
            raise ValueError(f"TRACE execution mode mismatch: {method}")
        if (
            identity["data_identity"]["protocol"]
            != qualification["data_protocol"]
        ):
            raise ValueError(f"TRACE data protocol mismatch: {method}")
        if method == "lora_nf" and (
            identity["state_update"] != spec["locked_lora_nf_state"]
        ):
            raise ValueError("TRACE LoRA-NF state rule mismatch")
        if report["formal_result_eligible"] is not False:
            raise ValueError(f"TRACE pilot claimed formal eligibility: {method}")
        if report["effective_limits"] != expected_limits:
            raise ValueError(f"TRACE effective limits mismatch: {method}")
        retention = report["checkpoint_retention"]
        if retention.get("deleted_after_smoke_audit") is not True:
            raise ValueError(f"TRACE adapter was not deleted: {method}")
        comparison_hashes.add(identity["comparison_config_hash"])
        summary = report["summary"]
        rows.append(
            {
                "method": method,
                "final_average": float(summary["final_average"]),
                "final_bwt_trace_definition": float(
                    summary["final_bwt_trace_definition"]
                ),
                "final_forgetting_standard": float(
                    summary["final_forgetting_standard"]
                ),
                "peak_gpu_memory_bytes": int(
                    report["peak_gpu_memory_bytes"]
                ),
                "wall_seconds": float(report["wall_seconds"]),
            }
        )
    if len(comparison_hashes) != 1:
        raise ValueError("TRACE methods do not share one comparison hash")
    reference = next(row for row in rows if row["method"] == "lora_nf")
    for row in rows:
        row["final_average_minus_lora_nf"] = (
            row["final_average"] - reference["final_average"]
        )
        row["bwt_minus_lora_nf"] = (
            row["final_bwt_trace_definition"]
            - reference["final_bwt_trace_definition"]
        )
        row["forgetting_minus_lora_nf"] = (
            row["final_forgetting_standard"]
            - reference["final_forgetting_standard"]
        )
    ranking = [
        row["method"]
        for row in sorted(
            rows,
            key=lambda item: (-item["final_average"], item["method"]),
        )
    ]
    return {
        "formal_result_eligible": False,
        "qualification": dict(qualification),
        "expected_run_commit": expected_run_commit,
        "comparison_config_hash": comparison_hashes.pop(),
        "rows": rows,
        "ranking_by_final_average": ranking,
    }
