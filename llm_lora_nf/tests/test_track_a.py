import copy

import torch
import torch.nn as nn

from llm_lora_nf.inject import merge_native_adapters
from llm_lora_nf.layers import FilteredLoRALinear
from llm_lora_nf.track_a import (
    OfficialLoRANullAttentionCalibrator,
    OfficialLoRANullCalibrator,
    OfficialLoRANullMomentAccumulator,
    TrackAAdapterConfig,
    build_track_a_native_adapter,
    load_track_a_native_adapter,
    save_track_a_native_adapter,
    track_a_adapter_modules,
)
from test_injection_and_calibration import TinyModel


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }


def test_official_moment_uses_abs_of_signed_max_and_sample_divisor():
    accumulator = OfficialLoRANullMomentAccumulator(
        input_dim=2,
        expected_samples=1,
    )
    activations = torch.tensor([[[-10.0, 2.0], [-4.0, 1.0]]])
    accumulator.update(activations)
    state = accumulator.finalize()
    normalized = activations.squeeze(0) / 2.0
    expected = normalized.transpose(0, 1) @ normalized
    torch.testing.assert_close(state.matrix, expected)
    assert state.observations == 1


def test_track_b_official_calibrator_is_attention_only_and_shares_qkv():
    model = TinyModel()
    moments = OfficialLoRANullAttentionCalibrator(
        model,
        expected_samples=1,
    ).collect([_batch()], device=torch.device("cpu"))
    assert set(moments) == {
        "self_attn.qkv_shared",
        "self_attn.o_proj",
    }
    assert all(state.observations == 1 for state in moments.values())


def test_track_a_wraps_all_seven_decoder_linears_but_not_lm_head():
    model = TinyModel()
    config = TrackAAdapterConfig(method="lora", rank=2, alpha=2)
    wrapped, record = build_track_a_native_adapter(model, config)
    assert len(wrapped) == 7
    assert {name.rsplit(".", 1)[-1] for name in wrapped} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert isinstance(model.self_attn.q_proj, FilteredLoRALinear)
    assert isinstance(model.mlp.down_proj, FilteredLoRALinear)
    assert isinstance(model.lm_head, nn.Linear)
    assert record.details["protocol_track"] == "A"
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all("lora_A" in name or "lora_B" in name for name in trainable)


def test_track_a_lora_null_initialization_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(67)
    source = TinyModel()
    original_state = copy.deepcopy(source.state_dict())
    batch = _batch()
    baseline = source(**batch)
    moments = OfficialLoRANullCalibrator(
        source,
        expected_samples=1,
    ).collect([batch], device=torch.device("cpu"))
    config = TrackAAdapterConfig(method="lora_null", rank=2, alpha=2)
    _, record = build_track_a_native_adapter(
        source,
        config,
        calibration_moments=moments,
    )
    assert record.module_count == 7
    torch.testing.assert_close(source(**batch), baseline, atol=1e-5, rtol=1e-4)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if "lora_B.weight" in name:
                parameter.add_(0.01)
    expected = source(**batch)
    save_track_a_native_adapter(
        source,
        str(tmp_path),
        adapter_config=config,
        metadata={"test": True},
    )

    target = TinyModel()
    target.load_state_dict(original_state)
    manifest = load_track_a_native_adapter(
        target,
        str(tmp_path),
        adapter_config=config,
    )
    torch.testing.assert_close(target(**batch), expected, atol=1e-6, rtol=1e-5)
    assert manifest["protocol_track"] == "A"
    assert manifest["metadata"]["test"] is True
    assert len(dict(track_a_adapter_modules(target))) == 7

    expected_merged = target(**batch)
    merged = merge_native_adapters(target)
    assert len(merged) == 7
    assert not any(
        isinstance(module, FilteredLoRALinear) for module in target.modules()
    )
    torch.testing.assert_close(
        target(**batch),
        expected_merged,
        atol=1e-6,
        rtol=1e-5,
    )
