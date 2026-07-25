import copy

import torch

from llm_lora_nf.baselines import build_native_adapter
from llm_lora_nf.calibration import ActivationCalibrator, build_and_assign_filters
from llm_lora_nf.checkpoint import load_native_adapter, save_native_adapter
from llm_lora_nf.config import AdapterConfig
from test_injection_and_calibration import TinyModel


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
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
