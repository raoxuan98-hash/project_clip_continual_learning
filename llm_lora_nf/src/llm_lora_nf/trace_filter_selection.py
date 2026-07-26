from typing import Any, Dict, Mapping


def select_trace_filter_setting(
    reference_lora: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    maximum_final_average_gap: float,
    minimum_forgetting_improvement: float,
    maximum_final_bwt_regression: float,
    default_on_failed_gate: str,
) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("TRACE filter selection requires candidates")
    if default_on_failed_gate not in candidates:
        raise ValueError("TRACE filter default is not a candidate")
    if (
        maximum_final_average_gap < 0
        or minimum_forgetting_improvement < 0
        or maximum_final_bwt_regression < 0
    ):
        raise ValueError("TRACE filter selection thresholds must be nonnegative")
    reference_average = float(reference_lora["final_average"])
    reference_forgetting = float(
        reference_lora["final_forgetting_standard"]
    )
    reference_bwt = float(
        reference_lora["final_bwt_trace_definition"]
    )
    average_floor = reference_average - maximum_final_average_gap
    forgetting_ceiling = (
        reference_forgetting - minimum_forgetting_improvement
    )
    bwt_floor = reference_bwt - maximum_final_bwt_regression
    audits = {}
    passing = []
    for setting, summary in candidates.items():
        average = float(summary["final_average"])
        forgetting = float(summary["final_forgetting_standard"])
        bwt = float(summary["final_bwt_trace_definition"])
        gates = {
            "final_average": average >= average_floor,
            "forgetting": forgetting <= forgetting_ceiling,
            "bwt": bwt >= bwt_floor,
        }
        audits[setting] = {
            "summary": {
                "final_average": average,
                "final_forgetting_standard": forgetting,
                "final_bwt_trace_definition": bwt,
            },
            "gates": gates,
            "passed_all": all(gates.values()),
        }
        if all(gates.values()):
            passing.append(setting)
    if passing:
        selected = sorted(
            passing,
            key=lambda setting: (
                -audits[setting]["summary"]["final_average"],
                audits[setting]["summary"]["final_forgetting_standard"],
                -audits[setting]["summary"][
                    "final_bwt_trace_definition"
                ],
                setting,
            ),
        )[0]
    else:
        selected = default_on_failed_gate
    return {
        "selected_setting": selected,
        "default_on_failed_gate": default_on_failed_gate,
        "passed_settings": sorted(passing),
        "thresholds": {
            "final_average_floor": average_floor,
            "forgetting_ceiling": forgetting_ceiling,
            "bwt_floor": bwt_floor,
            "maximum_final_average_gap": maximum_final_average_gap,
            "minimum_forgetting_improvement": (
                minimum_forgetting_improvement
            ),
            "maximum_final_bwt_regression": (
                maximum_final_bwt_regression
            ),
        },
        "candidates": audits,
    }
