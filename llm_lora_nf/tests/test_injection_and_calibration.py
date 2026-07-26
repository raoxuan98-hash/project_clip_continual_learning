import torch
import torch.nn as nn

from llm_lora_nf.calibration import ActivationCalibrator, build_and_assign_filters
from llm_lora_nf.config import AdapterConfig
from llm_lora_nf.inject import adapter_modules, inject_attention_adapters
from llm_lora_nf.layers import FilteredLoRALinear


class TinyAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        mixed = (
            self.q_proj(hidden_states)
            + self.k_proj(hidden_states)
            + self.v_proj(hidden_states)
        ) / 3.0
        return self.o_proj(mixed)


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return self.down_proj(torch.relu(self.gate_proj(hidden_states) + self.up_proj(hidden_states)))


class TinyModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.self_attn = TinyAttention(hidden_size)
        self.mlp = TinyMLP(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.embed_tokens(input_ids)
        hidden = self.self_attn(hidden)
        hidden = self.mlp(hidden)
        return self.lm_head(hidden)


def test_injection_is_attention_only():
    model = TinyModel()
    wrapped = inject_attention_adapters(
        model,
        AdapterConfig(method="lora_nf", rank=2, alpha=2),
    )
    assert set(name.rsplit(".", 1)[-1] for name in wrapped) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }
    assert isinstance(model.self_attn.q_proj, FilteredLoRALinear)
    assert isinstance(model.mlp.gate_proj, nn.Linear)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all("lora_A" in name or "lora_B" in name for name in trainable_names)


def test_calibration_shares_qkv_and_masks_padding():
    torch.manual_seed(11)
    model = TinyModel()
    inject_attention_adapters(
        model,
        AdapterConfig(method="lora_nf", rank=2, alpha=2),
    )
    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        }
    ]
    moments = ActivationCalibrator(model).collect(
        batches, device=torch.device("cpu")
    )
    assert len(moments) == 2  # q/k/v shared + o_proj
    assert all(state.observations == 5 for state in moments.values())

    results = build_and_assign_filters(
        model,
        moments,
        energy_fraction=0.20,
        leakage=0.02,
        ridge=1e-4,
    )
    assert len(results) == 2
    modules = dict(adapter_modules(model))
    q_basis = modules["self_attn.q_proj"].filter.protected_basis
    torch.testing.assert_close(
        q_basis, modules["self_attn.k_proj"].filter.protected_basis
    )
    torch.testing.assert_close(
        q_basis, modules["self_attn.v_proj"].filter.protected_basis
    )
    assert modules["self_attn.q_proj"].filter is modules["self_attn.k_proj"].filter
    assert modules["self_attn.q_proj"].filter is modules["self_attn.v_proj"].filter
    assert (
        modules["self_attn.q_proj"].filter.protected_basis.data_ptr()
        == modules["self_attn.v_proj"].filter.protected_basis.data_ptr()
    )
    assert modules["self_attn.o_proj"].filter is not modules["self_attn.q_proj"].filter
