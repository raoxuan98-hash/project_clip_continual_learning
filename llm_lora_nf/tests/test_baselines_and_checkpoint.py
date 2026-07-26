import copy

import torch

from llm_lora_nf.baselines import (
    _scaled_factors_from_projected_weight,
    build_native_adapter,
)
from llm_lora_nf.calibration import ActivationCalibrator, build_and_assign_filters
from llm_lora_nf.checkpoint import load_native_adapter, save_native_adapter
from llm_lora_nf.config import AdapterConfig
from llm_lora_nf.inject import adapter_modules, merge_native_adapters
from llm_lora_nf.layers import FilteredLoRALinear
from llm_lora_nf.method_registry import (
    TRACK_B_DIRECT_PEFT_METHODS,
    TRACK_B_METHODS,
    TRACK_B_NATIVE_METHODS,
    TRACK_B_PEFT_METHODS,
)
from test_injection_and_calibration import TinyModel


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }


def test_controlled_lora_baseline_routes_to_pinned_peft():
    assert "lora" in TRACK_B_DIRECT_PEFT_METHODS
    assert "lora" in TRACK_B_PEFT_METHODS
    assert "lora" not in TRACK_B_NATIVE_METHODS
    assert TRACK_B_NATIVE_METHODS == {"lora_nf", "lora_null", "milora"}
    assert TRACK_B_METHODS == {
        "lora",
        "dora",
        "lora_null",
        "pissa",
        "milora",
        "corda",
        "lora_nf",
    }


def test_lora_null_initialization_preserves_checkpoint_function():
    torch.manual_seed(41)
    model = TinyModel()
    batch = _batch()
    baseline = model(**batch)
    moments = ActivationCalibrator(model).collect(
        [batch],
        device=torch.device("cpu"),
    )
    config = AdapterConfig(method="lora_null", rank=2, alpha=2, dropout=0.0)
    _, record = build_native_adapter(
        model,
        config,
        calibration_moments=moments,
    )
    torch.testing.assert_close(model(**batch), baseline, atol=1e-5, rtol=1e-4)
    assert record.calibration_required
    assert record.exact_function_preserving


def test_milora_fp32_svd_initialization_preserves_checkpoint_function():
    torch.manual_seed(42)
    model = TinyModel()
    batch = _batch()
    baseline = model(**batch)
    config = AdapterConfig(method="milora", rank=2, alpha=2, dropout=0.0)
    _, record = build_native_adapter(model, config)
    torch.testing.assert_close(model(**batch), baseline, atol=1e-5, rtol=1e-4)
    assert record.details["svd_dtype"] == "float32"
    assert record.details["svd_devices"] == ["cpu"]


def test_native_checkpoint_roundtrip_preserves_logits_and_filter(tmp_path):
    torch.manual_seed(43)
    source = TinyModel()
    original_state = copy.deepcopy(source.state_dict())
    batch = _batch()
    moments = ActivationCalibrator(source).collect(
        [batch],
        device=torch.device("cpu"),
    )
    config = AdapterConfig(method="lora_nf", rank=2, alpha=2, dropout=0.0)
    build_native_adapter(source, config, calibration_moments=moments)
    build_and_assign_filters(
        source,
        moments,
        energy_fraction=0.20,
        leakage=0.02,
        ridge=1e-4,
    )
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if "lora_B.weight" in name:
                parameter.normal_(0.0, 0.1)
    expected = source(**batch)
    save_native_adapter(
        source,
        str(tmp_path),
        adapter_config=config,
        metadata={"test": True},
    )

    target = TinyModel()
    target.load_state_dict(original_state)
    manifest = load_native_adapter(
        target,
        str(tmp_path),
        adapter_config=config,
    )
    torch.testing.assert_close(target(**batch), expected, atol=1e-6, rtol=1e-5)
    assert manifest["metadata"]["test"] is True
    modules = dict(adapter_modules(target))
    assert modules["self_attn.q_proj"].filter is modules["self_attn.k_proj"].filter
    assert modules["self_attn.q_proj"].filter is modules["self_attn.v_proj"].filter


def test_lora_null_checkpoint_roundtrip_reapplies_base_offset(tmp_path):
    torch.manual_seed(47)
    source = TinyModel()
    original_state = copy.deepcopy(source.state_dict())
    batch = _batch()
    moments = ActivationCalibrator(source).collect(
        [batch],
        device=torch.device("cpu"),
    )
    config = AdapterConfig(method="lora_null", rank=2, alpha=2, dropout=0.0)
    build_native_adapter(source, config, calibration_moments=moments)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if "lora_B.weight" in name:
                parameter.add_(0.01)
    expected = source(**batch)
    save_native_adapter(
        source,
        str(tmp_path),
        adapter_config=config,
        metadata={},
    )

    target = TinyModel()
    target.load_state_dict(original_state)
    load_native_adapter(target, str(tmp_path), adapter_config=config)
    torch.testing.assert_close(target(**batch), expected, atol=1e-6, rtol=1e-5)


def test_low_rank_projected_svd_matches_dense_projection():
    torch.manual_seed(53)
    weight = torch.randn(11, 9, dtype=torch.float64)
    basis, _ = torch.linalg.qr(torch.randn(9, 3, dtype=torch.float64))
    lora_A, lora_B = _scaled_factors_from_projected_weight(
        weight,
        basis,
        scaling=1.0,
    )
    expected = weight @ basis @ basis.transpose(0, 1)
    torch.testing.assert_close(lora_B @ lora_A, expected, atol=1e-10, rtol=1e-10)


def test_merge_native_adapters_restores_standard_linear_modules():
    torch.manual_seed(59)
    model = TinyModel()
    batch = _batch()
    moments = ActivationCalibrator(model).collect(
        [batch],
        device=torch.device("cpu"),
    )
    config = AdapterConfig(method="lora_nf", rank=2, alpha=2, dropout=0.0)
    build_native_adapter(model, config, calibration_moments=moments)
    build_and_assign_filters(
        model,
        moments,
        energy_fraction=0.20,
        leakage=0.02,
        ridge=1e-4,
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B.weight" in name:
                parameter.normal_(0.0, 0.1)
    expected = model(**batch)
    merged = merge_native_adapters(model)
    assert len(merged) == 4
    assert not any(isinstance(module, FilteredLoRALinear) for module in model.modules())
    torch.testing.assert_close(model(**batch), expected, atol=1e-6, rtol=1e-5)
