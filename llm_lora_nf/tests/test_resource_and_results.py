import pytest

from llm_lora_nf.resource_guard import decide_admission, parse_nvidia_smi_csv
from llm_lora_nf.result_metadata import (
    RunMetadata,
    require_formal_result,
)


def test_gpu_admission_uses_at_most_two_and_reserves_one():
    statuses = parse_nvidia_smi_csv(
        "\n".join(
            [
                "0, 81920, 10, 0",
                "1, 81920, 10, 0",
                "2, 81920, 10, 0",
                "3, 81920, 50000, 90",
            ]
        )
    )
    decision = decide_admission(statuses, requested=2)
    assert decision.mode == "gpu"
    assert decision.selected_gpu_indices == [0, 1]
    assert 2 in decision.idle_gpu_indices


def test_one_idle_gpu_forces_cpu_smoke():
    statuses = parse_nvidia_smi_csv(
        "\n".join(["0, 81920, 10, 0", "1, 81920, 50000, 90"])
    )
    decision = decide_admission(statuses, requested=2)
    assert decision.mode == "cpu_smoke_only"
    assert decision.selected_gpu_indices == []


def test_cpu_smoke_cannot_enter_formal_results():
    metadata = RunMetadata(
        run_id="smoke",
        execution_mode="cpu_smoke_only",
        method="lora_nf",
        model_id="tiny",
        seed=42,
        commit_sha="test",
    )
    assert not metadata.formal_result_eligible
    with pytest.raises(ValueError):
        require_formal_result(metadata)

