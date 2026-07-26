import pytest

from llm_lora_nf.resource_guard import (
    decide_admission,
    parse_nvidia_smi_csv,
    project_file_lock,
)
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


def test_cpu_smoke_requests_no_gpu_and_reserves_all_idle_devices():
    statuses = parse_nvidia_smi_csv(
        "0, 24576, 0, 0\n"
        "1, 24576, 0, 0\n"
        "2, 24576, 0, 0\n"
    )
    decision = decide_admission(statuses, requested=0)
    assert decision.mode == "cpu_smoke_only"
    assert decision.selected_gpu_indices == []
    assert decision.idle_gpu_indices == [0, 1, 2]
    assert 2 in decision.idle_gpu_indices


def test_one_idle_gpu_forces_cpu_smoke():
    statuses = parse_nvidia_smi_csv(
        "\n".join(["0, 81920, 10, 0", "1, 81920, 50000, 90"])
    )
    decision = decide_admission(statuses, requested=2)
    assert decision.mode == "cpu_smoke_only"
    assert decision.selected_gpu_indices == []


def test_project_file_lock_creates_only_the_exact_lock_file(tmp_path):
    lock_path = tmp_path / "evaluation.lock"
    with project_file_lock(str(lock_path)):
        assert lock_path.is_file()
    assert list(tmp_path.iterdir()) == [lock_path]


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
