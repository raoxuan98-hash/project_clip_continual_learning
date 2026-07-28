import pytest
import torch

from llm_lora_nf.resource_guard import (
    _numeric_visible_gpu_indices,
    PARENT_MANAGED_GPU_BATCH_ENV,
    admitted_torch_device,
    decide_admission,
    parse_nvidia_smi_csv,
    project_file_lock,
    project_gpu_lock,
)
from llm_lora_nf.result_metadata import (
    RunMetadata,
    require_formal_result,
    tensor_difference_summary,
)


def test_gpu_admission_uses_at_most_three_and_reserves_one():
    statuses = parse_nvidia_smi_csv(
        "\n".join(
            [
                "0, 81920, 10, 0",
                "1, 81920, 10, 0",
                "2, 81920, 10, 0",
                "3, 81920, 10, 0",
                "4, 81920, 50000, 90",
            ]
        )
    )
    decision = decide_admission(statuses, requested=3)
    assert decision.mode == "gpu"
    assert decision.selected_gpu_indices == [0, 1, 2]


def test_gpu_admission_respects_explicit_eligible_devices_and_global_reserve():
    statuses = parse_nvidia_smi_csv(
        "\n".join(
            [
                "0, 81920, 10, 0",
                "1, 81920, 10, 0",
                "2, 81920, 10, 0",
                "3, 81920, 10, 0",
            ]
        )
    )
    decision = decide_admission(
        statuses,
        requested=2,
        eligible_gpu_indices=[3, 1],
    )
    assert decision.selected_gpu_indices == [3, 1]
    assert decision.idle_gpu_indices == [0, 1, 2, 3]


def test_visible_gpu_parser_accepts_only_unique_physical_indices():
    assert _numeric_visible_gpu_indices(None) is None
    assert _numeric_visible_gpu_indices("") == []
    assert _numeric_visible_gpu_indices("5, 2") == [5, 2]
    with pytest.raises(RuntimeError, match="physical numeric"):
        _numeric_visible_gpu_indices("GPU-uuid")
    with pytest.raises(ValueError, match="duplicate"):
        _numeric_visible_gpu_indices("2,2")


def test_gpu_admission_rejects_more_than_three():
    with pytest.raises(ValueError, match="between 0 and 3"):
        decide_admission([], requested=4)


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


def test_admitted_torch_device_resolves_existing_visibility():
    assert admitted_torch_device(
        5,
        cuda_visible_devices=None,
    ) == "cuda:5"
    assert admitted_torch_device(
        5,
        cuda_visible_devices="5,2",
    ) == "cuda:0"
    assert admitted_torch_device(
        2,
        cuda_visible_devices="5,2",
    ) == "cuda:1"


def test_admitted_torch_device_rejects_incompatible_visibility():
    with pytest.raises(RuntimeError, match="outside CUDA_VISIBLE_DEVICES"):
        admitted_torch_device(0, cuda_visible_devices="5,2")
    with pytest.raises(RuntimeError, match="empty"):
        admitted_torch_device(0, cuda_visible_devices="")


def test_project_file_lock_creates_only_the_exact_lock_file(tmp_path):
    lock_path = tmp_path / "evaluation.lock"
    with project_file_lock(str(lock_path)):
        assert lock_path.is_file()
    assert list(tmp_path.iterdir()) == [lock_path]


def test_parent_managed_gpu_batch_bypasses_duplicate_child_lock(
    tmp_path,
    monkeypatch,
):
    lock_path = tmp_path / "gpu.lock"
    monkeypatch.setenv(PARENT_MANAGED_GPU_BATCH_ENV, "1")
    with project_gpu_lock(str(lock_path)):
        assert not lock_path.exists()


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


def test_tensor_difference_summary_records_initialization_drift():
    reference = torch.tensor([0.0, 2.0])
    candidate = torch.tensor([1.0, 0.0])
    summary = tensor_difference_summary(reference, candidate)
    assert summary["element_count"] == 2
    assert summary["max_absolute_error"] == pytest.approx(2.0)
    assert summary["mean_absolute_error"] == pytest.approx(1.5)
    assert summary["rmse"] == pytest.approx(2.5**0.5)
    assert summary["reference_rms"] == pytest.approx(2.0**0.5)
    assert summary["relative_rmse"] == pytest.approx(1.25**0.5)

    with pytest.raises(ValueError, match="finite"):
        tensor_difference_summary(reference, torch.tensor([float("nan"), 0.0]))
