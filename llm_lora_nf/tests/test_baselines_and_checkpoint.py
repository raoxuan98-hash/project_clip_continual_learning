import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llm_lora_nf.baselines import (
    _scaled_factors_from_projected_weight,
    build_corda_baseline,
    build_native_adapter,
    build_peft_baseline,
)
from llm_lora_nf.calibration import ActivationCalibrator, build_and_assign_filters
from llm_lora_nf.checkpoint import load_native_adapter, save_native_adapter
from llm_lora_nf.config import AdapterConfig
from llm_lora_nf.inject import (
    adapter_modules,
    audit_trainable_parameter_scope,
    merge_native_adapters,
)
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


def _tiny_llama():
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )


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


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@torch.no_grad()
def test_direct_peft_baselines_are_fp32_attention_only_and_function_preserving(
    method,
):
    torch.manual_seed(37)
    model = _tiny_llama()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    baseline = model(input_ids=input_ids).logits
    config = AdapterConfig(method=method, rank=2, alpha=2, dropout=0.0)
    model = build_peft_baseline(model, config, method)
    audit = audit_trainable_parameter_scope(
        model,
        target_modules=config.target_modules,
    )
    assert audit["observed_target_modules"] == [
        "k_proj",
        "o_proj",
        "q_proj",
        "v_proj",
    ]
    assert audit["trainable_dtype_elements"] == {
        "torch.float32": audit["trainable_parameter_count"]
    }
    torch.testing.assert_close(
        model(input_ids=input_ids).logits,
        baseline,
        atol=2e-4,
        rtol=2e-4,
    )


@torch.no_grad()
def test_corda_baseline_is_fp32_attention_only_and_function_preserving(
    tmp_path,
):
    torch.manual_seed(39)
    model = _tiny_llama()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    baseline = model(input_ids=input_ids).logits
    config = AdapterConfig(method="corda", rank=2, alpha=2, dropout=0.0)

    def run_calibration():
        model(input_ids=input_ids)

    model = build_corda_baseline(
        model,
        config,
        run_calibration=run_calibration,
        cache_file=str(tmp_path / "corda_eigens.pt"),
        covariance_file=str(tmp_path / "corda_covariance.pt"),
        mode="kpm",
    )
    audit = audit_trainable_parameter_scope(
        model,
        target_modules=config.target_modules,
    )
    assert audit["observed_target_modules"] == [
        "k_proj",
        "o_proj",
        "q_proj",
        "v_proj",
    ]
    assert audit["trainable_dtype_elements"] == {
        "torch.float32": audit["trainable_parameter_count"]
    }
    torch.testing.assert_close(
        model(input_ids=input_ids).logits,
        baseline,
        atol=2e-4,
        rtol=2e-4,
    )


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
