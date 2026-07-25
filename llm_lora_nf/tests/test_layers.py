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


def test_optimizer_step_matches_dense_filter_reference():
    layer = _make_layer()
    reference = copy.deepcopy(layer)
    x = torch.randn(2, 3, 9)
    optimizer = torch.optim.SGD(
        [layer.lora_A.weight, layer.lora_B.weight],
        lr=0.03,
    )
    reference_optimizer = torch.optim.SGD(
        [reference.lora_A.weight, reference.lora_B.weight],
        lr=0.03,
    )

    layer(x).square().mean().backward()
    optimizer.step()

    filtered = x @ reference.filter.dense()
    output = reference.base_layer(x) + reference.scaling * reference.lora_B(
        reference.lora_A(filtered)
    )
    output.square().mean().backward()
    reference_optimizer.step()

    torch.testing.assert_close(layer.lora_A.weight, reference.lora_A.weight)
    torch.testing.assert_close(layer.lora_B.weight, reference.lora_B.weight)


def test_identity_filter_and_leakage_one_reduce_to_standard_lora():
    torch.manual_seed(19)
    base = nn.Linear(6, 5)
    filtered = FilteredLoRALinear(
        copy.deepcopy(base),
        rank=2,
        alpha=2,
        dropout=0.0,
    )
    with torch.no_grad():
        filtered.lora_B.weight.normal_()
    x = torch.randn(3, 6)
    expected = filtered.base_layer(x) + filtered.scaling * filtered.lora_B(
        filtered.lora_A(x)
    )
    torch.testing.assert_close(filtered(x), expected)

    basis, _ = torch.linalg.qr(torch.randn(6, 3))
    filtered.set_filter(HardLeakyFilter(6, basis, leakage=1.0))
    torch.testing.assert_close(filtered(x), expected)


def test_protected_subspace_is_scaled_by_leakage():
    basis, _ = torch.linalg.qr(torch.randn(7, 2))
    module = HardLeakyFilter(7, basis, leakage=0.1)
    coefficients = torch.randn(4, 2)
    x = coefficients @ basis.transpose(0, 1)
    torch.testing.assert_close(module(x), 0.1 * x, atol=1e-6, rtol=1e-5)


def test_decomposition_initialization_preserves_original_function():
    torch.manual_seed(29)
    base = nn.Linear(8, 6)
    original = copy.deepcopy(base)
    layer = FilteredLoRALinear(
        base,
        rank=3,
        alpha=3,
        dropout=0.0,
        use_filter=False,
    )
    lora_A = torch.randn(3, 8) * 0.1
    lora_B = torch.randn(6, 3) * 0.1
    layer.set_decomposition_initialization(lora_A, lora_B)
    x = torch.randn(5, 8)
    torch.testing.assert_close(layer(x), original(x), atol=1e-6, rtol=1e-5)
    assert layer.has_base_offset
