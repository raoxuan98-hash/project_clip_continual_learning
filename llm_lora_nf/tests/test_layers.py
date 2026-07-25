import copy

import torch
import torch.nn as nn

from llm_lora_nf.filters import HardLeakyFilter
from llm_lora_nf.layers import FilteredLoRALinear


def _make_layer() -> FilteredLoRALinear:
    torch.manual_seed(3)
    base = nn.Linear(9, 7, bias=True)
    layer = FilteredLoRALinear(base, rank=3, alpha=3, dropout=0.0)
    with torch.no_grad():
        layer.lora_B.weight.normal_(0.0, 0.1)
    q, _ = torch.linalg.qr(torch.randn(9, 2))
    layer.set_filter(HardLeakyFilter(9, q, leakage=0.02))
    return layer


def test_layer_matches_explicit_dense_weight():
    layer = _make_layer()
    x = torch.randn(2, 4, 9)
    dense_weight = (
        layer.base_layer.weight
        + layer.scaling
        * layer.lora_B.weight
        @ layer.lora_A.weight
        @ layer.filter.dense()
    )
    expected = torch.nn.functional.linear(x, dense_weight, layer.base_layer.bias)
    torch.testing.assert_close(layer(x), expected, atol=1e-6, rtol=1e-5)


def test_layer_gradient_matches_dense_filter_branch():
    layer = _make_layer()
    reference = copy.deepcopy(layer)
    x = torch.randn(2, 3, 9)

    loss = layer(x).square().mean()
    loss.backward()

    filtered = x @ reference.filter.dense()
    output = reference.base_layer(x) + reference.scaling * reference.lora_B(
        reference.lora_A(filtered)
    )
    output.square().mean().backward()

    torch.testing.assert_close(
        layer.lora_A.weight.grad,
        reference.lora_A.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        layer.lora_B.weight.grad,
        reference.lora_B.weight.grad,
        atol=1e-6,
        rtol=1e-5,
    )


def test_merge_and_unmerge_preserve_output():
    layer = _make_layer().eval()
    x = torch.randn(4, 9)
    before = layer(x)
    layer.merge()
    merged = layer(x)
    torch.testing.assert_close(before, merged, atol=1e-6, rtol=1e-5)
    layer.unmerge()
    after = layer(x)
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)

