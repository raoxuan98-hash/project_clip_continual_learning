from typing import Any, Dict, Mapping

from .trace_filter_selection import select_trace_filter_setting


TRACE_FILTER_SETTINGS = (
    "e20_r02",
    "e20_r05",
    "e20_r10",
    "e20_r20",
    "e10_r10",
)

TRACE_FILTER_QUALIFICATION = {
    "data_protocol": "treelora_500_pilot",
    "execution_mode": "gpu_chain_smoke_only",
    "formal_result_eligible": False,
    "purpose": (
        "one_preregistered_filter_strength_check_after_negative_pilot"
    ),
}


def _validate_report(
    *,
    label: str,
    report: Mapping[str, Any],
    expected_method: str,
    expected_commit: str,
    expected_config_hash: str,
    expected_seed: int,
    expected_order: str,
    expected_limits: Mapping[str, Any],
) -> Dict[str, Any]:
    identity = report["identity"]
    if identity["method"] != expected_method:
        raise ValueError(f"TRACE filter method mismatch: {label}")
    if identity["commit_sha"] != expected_commit:
        raise ValueError(f"TRACE filter run commit mismatch: {label}")
    if identity["config_hash"] != expected_config_hash:
        raise ValueError(f"TRACE filter config hash mismatch: {label}")
    if identity["seed"] != expected_seed:
        raise ValueError(f"TRACE filter seed mismatch: {label}")
    if identity["order"] != expected_order:
        raise ValueError(f"TRACE filter order mismatch: {label}")
    if identity["execution_mode"] != TRACE_FILTER_QUALIFICATION[
        "execution_mode"
    ]:
        raise ValueError(f"TRACE filter execution mode mismatch: {label}")
    if (
        identity["data_identity"]["protocol"]
        != TRACE_FILTER_QUALIFICATION["data_protocol"]
    ):
        raise ValueError(f"TRACE filter data protocol mismatch: {label}")
    if report["formal_result_eligible"] is not False:
        raise ValueError(f"TRACE filter claimed formal eligibility: {label}")
    if report["effective_limits"] != expected_limits:
        raise ValueError(f"TRACE filter effective limits mismatch: {label}")
    retention = report["checkpoint_retention"]
    if retention.get("deleted_after_smoke_audit") is not True:
        raise ValueError(f"TRACE filter adapter was not deleted: {label}")
    if expected_method == "lora_nf" and (
        identity["state_update"] != "reference_fixed"
    ):
        raise ValueError(f"TRACE filter state rule mismatch: {label}")
    summary = report["summary"]
    return {
        "setting": label,
        "final_average": float(summary["final_average"]),
        "final_bwt_trace_definition": float(
            summary["final_bwt_trace_definition"]
        ),
        "final_forgetting_standard": float(
            summary["final_forgetting_standard"]
        ),
        "peak_gpu_memory_bytes": int(report["peak_gpu_memory_bytes"]),
        "wall_seconds": float(report["wall_seconds"]),
        "commit_sha": identity["commit_sha"],
        "config_hash": identity["config_hash"],
        "comparison_config_hash": identity["comparison_config_hash"],
    }


def aggregate_trace_filter_reports(
    spec: Mapping[str, Any],
    reference_lora: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    expected_run_commits: Mapping[str, str],
    expected_config_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    if tuple(spec.get("settings", ())) != TRACE_FILTER_SETTINGS:
        raise ValueError("TRACE filter setting set or order drifted")
    if set(candidates) != set(TRACE_FILTER_SETTINGS):
        raise ValueError("TRACE filter candidate report set is incomplete")
    if set(expected_run_commits) != {"lora", *TRACE_FILTER_SETTINGS}:
        raise ValueError("TRACE filter expected commit map is incomplete")
    if set(expected_config_hashes) != {"lora", *TRACE_FILTER_SETTINGS}:
        raise ValueError("TRACE filter expected config hash map is incomplete")
    if spec.get("qualification") != TRACE_FILTER_QUALIFICATION:
        raise ValueError("TRACE filter qualification drifted")
    if spec.get("resources") != {
        "max_parallel_gpus": 3,
        "reserve_idle_gpus": 1,
    }:
        raise ValueError("TRACE filter resource policy drifted")

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
    seed = int(spec["seed"])
    order = str(spec["order"])
    reference_row = _validate_report(
        label="lora",
        report=reference_lora,
        expected_method="lora",
        expected_commit=expected_run_commits["lora"],
        expected_config_hash=expected_config_hashes["lora"],
        expected_seed=seed,
        expected_order=order,
        expected_limits=expected_limits,
    )

    rows = []
    selection_candidates = {}
    identities = {}
    for setting in TRACE_FILTER_SETTINGS:
        row = _validate_report(
            label=setting,
            report=candidates[setting],
            expected_method="lora_nf",
            expected_commit=expected_run_commits[setting],
            expected_config_hash=expected_config_hashes[setting],
            expected_seed=seed,
            expected_order=order,
            expected_limits=expected_limits,
        )
        rows.append(row)
        selection_candidates[setting] = row
        identity = candidates[setting]["identity"]
        identities[setting] = {
            "model_id": identity["model_id"],
            "active_task_order": identity["active_task_order"],
            "data_files_sha256": identity["data_identity"]["files_sha256"],
        }

    reference_identity = reference_lora["identity"]
    reference_protocol = {
        "model_id": reference_identity["model_id"],
        "active_task_order": reference_identity["active_task_order"],
        "data_files_sha256": reference_identity["data_identity"][
            "files_sha256"
        ],
    }
    for setting, identity in identities.items():
        if identity != reference_protocol:
            raise ValueError(
                f"TRACE filter comparison identity mismatch: {setting}"
            )

    selection_spec = spec["selection"]
    if selection_spec["reference_method"] != "lora":
        raise ValueError("TRACE filter reference method drifted")
    if selection_spec["tie_break_order"] != [
        "highest_final_average",
        "lowest_forgetting",
        "highest_bwt",
    ]:
        raise ValueError("TRACE filter tie break drifted")
    selection = select_trace_filter_setting(
        reference_row,
        selection_candidates,
        maximum_final_average_gap=float(
            selection_spec["maximum_final_average_gap"]
        ),
        minimum_forgetting_improvement=float(
            selection_spec["minimum_forgetting_improvement"]
        ),
        maximum_final_bwt_regression=float(
            selection_spec["maximum_final_bwt_regression"]
        ),
        default_on_failed_gate=str(
            selection_spec["default_on_failed_gate"]
        ),
    )
    return {
        "formal_result_eligible": False,
        "qualification": dict(TRACE_FILTER_QUALIFICATION),
        "reference_lora": reference_row,
        "rows": rows,
        "selection": selection,
    }
