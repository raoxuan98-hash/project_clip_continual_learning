import copy

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

from llm_lora_nf.baselines import build_peft_baseline
from llm_lora_nf.calibration import SecondMomentState
from llm_lora_nf.config import AdapterConfig
from llm_lora_nf.continual_adapter import (
    CumulativeAdapterState,
    capture_and_merge_native_task,
    capture_and_merge_peft_task,
    load_cumulative_adapter,
    replay_native_factor_stack,
    replay_peft_state_stack,
    save_cumulative_adapter,
    update_continual_moment_map,
)
from llm_lora_nf.filters import HardLeakyFilter
from llm_lora_nf.inject import (
    adapter_modules,
    audit_trainable_parameter_scope,
    inject_attention_adapters,
)


class TinyAttention(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, values):
        mixed = (
            self.q_proj(values)
            + self.k_proj(values)
            + self.v_proj(values)
        ) / 3.0
        return self.o_proj(mixed)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()

    def forward(self, values):
        return self.attn(values)


@torch.no_grad()
def _set_task_parameters(model, *, seed, filtered):
    generator = torch.Generator().manual_seed(seed)
    for _, module in adapter_modules(model):
        module.lora_A.weight.copy_(
            torch.randn(
                module.lora_A.weight.shape,
                generator=generator,
                dtype=module.lora_A.weight.dtype,
            )
            * 0.2
        )
        module.lora_B.weight.copy_(
            torch.randn(
                module.lora_B.weight.shape,
                generator=generator,
                dtype=module.lora_B.weight.dtype,
            )
            * 0.2
        )
        if filtered:
            basis = torch.linalg.qr(
                torch.randn(module.in_features, 2, generator=generator)
            ).Q
            module.set_filter(
                HardLeakyFilter(
                    module.in_features,
                    protected_basis=basis,
                    leakage=0.1,
                )
            )


@torch.no_grad()
def test_native_factor_stack_reconstructs_multiple_filtered_tasks():
    torch.manual_seed(3)
    source = TinyModel()
    original_state = copy.deepcopy(source.state_dict())
    config = AdapterConfig(method="lora_nf", rank=2, alpha=2)
    inject_attention_adapters(source, config)
    state = CumulativeAdapterState.create(
        method="lora_nf",
        backend="native_factor_stack",
        adapter_config=config,
    )

    _set_task_parameters(source, seed=11, filtered=True)
    capture_and_merge_native_task(source, state, task="task-1")
    _set_task_parameters(source, seed=12, filtered=True)
    capture_and_merge_native_task(source, state, task="task-2")

    values = torch.randn(2, 4, 6)
    expected = source(values)
    rebuilt = TinyModel()
    rebuilt.load_state_dict(original_state)
    replay_native_factor_stack(rebuilt, state, adapter_config=config)
    torch.testing.assert_close(rebuilt(values), expected, atol=1e-6, rtol=1e-5)
    assert state.tasks == ["task-1", "task-2"]


@torch.no_grad()
def test_native_factor_stack_replays_lora_null_base_offset():
    torch.manual_seed(7)
    source = TinyModel()
    original_state = copy.deepcopy(source.state_dict())
    config = AdapterConfig(method="lora_null", rank=2, alpha=2)
    inject_attention_adapters(source, config)
    state = CumulativeAdapterState.create(
        method="lora_null",
        backend="native_factor_stack",
        adapter_config=config,
    )
    generator = torch.Generator().manual_seed(19)
    for _, module in adapter_modules(source):
        initial_A = torch.randn(2, 6, generator=generator) * 0.1
        initial_B = torch.randn(6, 2, generator=generator) * 0.1
        module.set_decomposition_initialization(initial_A, initial_B)
        module.lora_A.weight.add_(0.05)
        module.lora_B.weight.sub_(0.03)
    capture_and_merge_native_task(source, state, task="task-1")

    values = torch.randn(2, 3, 6)
    rebuilt = TinyModel()
    rebuilt.load_state_dict(original_state)
    replay_native_factor_stack(rebuilt, state, adapter_config=config)
    torch.testing.assert_close(
        rebuilt(values), source(values), atol=1e-6, rtol=1e-5
    )


def test_cumulative_adapter_checkpoint_is_one_replayable_artifact(tmp_path):
    model = TinyModel()
    config = AdapterConfig(method="lora", rank=2, alpha=2)
    inject_attention_adapters(model, config)
    state = CumulativeAdapterState.create(
        method="lora",
        backend="native_factor_stack",
        adapter_config=config,
    )
    _set_task_parameters(model, seed=21, filtered=False)
    capture_and_merge_native_task(model, state, task="task-1")
    weights, manifest = save_cumulative_adapter(
        state,
        str(tmp_path),
        metadata={"base_model": "tiny-instruct"},
    )
    assert weights.is_file()
    assert manifest.is_file()
    restored, restored_manifest = load_cumulative_adapter(str(tmp_path))
    assert restored.tasks == ["task-1"]
    assert restored_manifest["checkpoint_semantics"].startswith(
        "one_final_replayable_adapter"
    )


def test_continual_moment_update_is_phase_equal_not_observation_weighted():
    old = {
        "group": SecondMomentState(
            matrix=torch.eye(2),
            observations=1000,
            phases=2,
        )
    }
    new = {
        "group": SecondMomentState(
            matrix=4.0 * torch.eye(2),
            observations=10,
            phases=1,
        )
    }
    updated = update_continual_moment_map(
        old,
        new,
        mode="reference_plus_history",
    )
    torch.testing.assert_close(updated["group"].matrix, 2.0 * torch.eye(2))
    assert updated["group"].phases == 3
    assert updated["group"].observations == 1010

    fixed = update_continual_moment_map(
        old,
        new,
        mode="reference_fixed",
    )
    torch.testing.assert_close(fixed["group"].matrix, torch.eye(2))
    assert fixed["group"].phases == 2
    with pytest.raises(ValueError, match="group set"):
        update_continual_moment_map(
            old,
            {"other": new["group"]},
            mode="reference_plus_history",
        )


@pytest.mark.parametrize("method", ["lora", "dora"])
@torch.no_grad()
def test_peft_state_stack_reconstructs_multiple_task_merges(method):
    model_config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(29)
    source = LlamaForCausalLM(model_config)
    original_state = copy.deepcopy(source.state_dict())
    config = AdapterConfig(method=method, rank=2, alpha=2, dropout=0.0)
    state = CumulativeAdapterState.create(
        method=method,
        backend="peft_state_stack",
        adapter_config=config,
    )
    for task_index, task_seed in enumerate((31, 37), start=1):
        source = build_peft_baseline(source, config, method)
        generator = torch.Generator().manual_seed(task_seed)
        for parameter in source.parameters():
            if parameter.requires_grad:
                parameter.add_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                    )
                    * 0.01
                )
        source, branch = capture_and_merge_peft_task(
            source,
            state,
            task=f"task-{task_index}",
        )
        assert not hasattr(source, "peft_config")
        if method == "dora":
            assert any(
                "magnitude" in name for name in branch["modules"]
            )
    assert state.tasks == ["task-1", "task-2"]

    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = source(input_ids=input_ids).logits
    rebuilt = LlamaForCausalLM(model_config)
    rebuilt.load_state_dict(original_state)
    rebuilt = replay_peft_state_stack(
        rebuilt,
        state,
        adapter_config=config,
    )
    torch.testing.assert_close(
        rebuilt(input_ids=input_ids).logits,
        expected,
        atol=1e-6,
        rtol=1e-5,
    )
    assert not hasattr(rebuilt, "peft_config")


def test_peft_lora_explicitly_promotes_bf16_adapter_parameters_to_fp32():
    model_config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(model_config).to(dtype=torch.bfloat16)
    config = AdapterConfig(method="lora", rank=2, alpha=2, dropout=0.0)
    model = build_peft_baseline(model, config, "lora")
    audit = audit_trainable_parameter_scope(
        model,
        target_modules=config.target_modules,
    )
    assert audit["trainable_dtype_elements"] == {
        "torch.float32": audit["trainable_parameter_count"]
    }
