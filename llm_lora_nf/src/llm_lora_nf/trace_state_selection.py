from typing import Any, Dict, Mapping


TRACE_STATE_MODES = ("reference_fixed", "reference_plus_history")


def _identity(report: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = report.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("TRACE state pilot report lacks identity")
    return identity


def validate_trace_state_pilot_pair(
    fixed: Mapping[str, Any],
    history: Mapping[str, Any],
) -> None:
    reports = {
        "reference_fixed": fixed,
        "reference_plus_history": history,
    }
    for expected_state, report in reports.items():
        identity = _identity(report)
        if report.get("formal_result_eligible") is not False:
            raise ValueError("TRACE state pilot must be non-formal")
        if report.get("status") != "passed":
            raise ValueError("TRACE state pilot report did not pass")
        if identity.get("execution_mode") != "gpu_chain_smoke_only":
            raise ValueError("TRACE state pilot must use isolated GPU smoke mode")
        if identity.get("method") != "lora_nf":
            raise ValueError("TRACE state pilot must compare LoRA-NF states")
        if identity.get("state_update") != expected_state:
            raise ValueError("TRACE state pilot report has the wrong state mode")
        if report.get("source_dirty") is not False:
            raise ValueError("TRACE state pilot requires clean source")
        retention = report.get("checkpoint_retention", {})
        if retention.get("deleted_after_smoke_audit") is not True:
            raise ValueError("TRACE state pilot adapter was not smoke-deleted")

    fixed_identity = _identity(fixed)
    history_identity = _identity(history)
    shared_keys = (
        "model_id",
        "model_snapshot_revision",
        "seed",
        "commit_sha",
        "state_ablation_config_hash",
        "data_identity",
        "order",
        "active_task_order",
    )
    for key in shared_keys:
        if fixed_identity.get(key) != history_identity.get(key):
            raise ValueError(f"TRACE state pilot identity mismatch: {key}")


def select_trace_state(
    fixed: Mapping[str, Any],
    history: Mapping[str, Any],
    *,
    minimum_final_average_gain: float,
    maximum_final_bwt_regression: float,
    maximum_wall_time_ratio: float,
) -> Dict[str, Any]:
    validate_trace_state_pilot_pair(fixed, history)
    if minimum_final_average_gain < 0.0:
        raise ValueError("Minimum TRACE state gain must be non-negative")
    if maximum_final_bwt_regression < 0.0:
        raise ValueError("Maximum TRACE BWT regression must be non-negative")
    if maximum_wall_time_ratio < 1.0:
        raise ValueError("Maximum TRACE wall-time ratio must be at least one")

    fixed_average = float(fixed["summary"]["final_average"])
    history_average = float(history["summary"]["final_average"])
    fixed_bwt = float(fixed["summary"]["final_bwt_trace_definition"])
    history_bwt = float(history["summary"]["final_bwt_trace_definition"])
    fixed_wall = float(fixed["wall_seconds"])
    history_wall = float(history["wall_seconds"])
    if fixed_wall <= 0.0 or history_wall <= 0.0:
        raise ValueError("TRACE state pilot wall time must be positive")

    average_gain = history_average - fixed_average
    bwt_delta = history_bwt - fixed_bwt
    wall_time_ratio = history_wall / fixed_wall
    gates = {
        "minimum_final_average_gain": (
            average_gain >= minimum_final_average_gain
        ),
        "maximum_final_bwt_regression": (
            bwt_delta >= -maximum_final_bwt_regression
        ),
        "maximum_wall_time_ratio": (
            wall_time_ratio <= maximum_wall_time_ratio
        ),
    }
    selected = (
        "reference_plus_history"
        if all(gates.values())
        else "reference_fixed"
    )
    return {
        "selected_state": selected,
        "default_on_failed_gate": "reference_fixed",
        "fixed": {
            "final_average": fixed_average,
            "final_bwt_trace_definition": fixed_bwt,
            "wall_seconds": fixed_wall,
        },
        "history": {
            "final_average": history_average,
            "final_bwt_trace_definition": history_bwt,
            "wall_seconds": history_wall,
        },
        "differences_history_minus_fixed": {
            "final_average": average_gain,
            "final_bwt_trace_definition": bwt_delta,
            "wall_time_ratio": wall_time_ratio,
        },
        "gates": gates,
        "thresholds": {
            "minimum_final_average_gain": minimum_final_average_gain,
            "maximum_final_bwt_regression": maximum_final_bwt_regression,
            "maximum_wall_time_ratio": maximum_wall_time_ratio,
        },
    }
