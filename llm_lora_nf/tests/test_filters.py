import pytest
import torch

from llm_lora_nf.filters import HardLeakyFilter, build_hard_leaky_filter


def _orthonormal_basis(dimension: int, rank: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(dimension, rank))
    return q


def test_filter_supports_standard_module_dtype_conversion():
    basis = _orthonormal_basis(8, 3)
    filter_module = HardLeakyFilter(8, basis, leakage=0.02)
    converted = filter_module.to(dtype=torch.float64)
    assert converted.protected_basis.dtype == torch.float64
    assert converted.leakage.dtype == torch.float64


def test_factorized_filter_matches_dense_forward_and_gradient():
    torch.manual_seed(42)
    dimension = 12
    basis = _orthonormal_basis(dimension, 4)
    filter_module = HardLeakyFilter(dimension, basis, leakage=0.02)

    x_factorized = torch.randn(3, 5, dimension, requires_grad=True)
    x_dense = x_factorized.detach().clone().requires_grad_(True)
    weight = torch.randn(dimension, 7)

    loss_factorized = (filter_module(x_factorized) @ weight).square().mean()
    loss_dense = (x_dense @ filter_module.dense() @ weight).square().mean()
    loss_factorized.backward()
    loss_dense.backward()

    torch.testing.assert_close(loss_factorized, loss_dense, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        x_factorized.grad, x_dense.grad, atol=1e-6, rtol=1e-5
    )


def test_protected_response_is_scaled_by_leakage():
    torch.manual_seed(7)
    dimension = 10
    basis = _orthonormal_basis(dimension, 3)
    filter_module = HardLeakyFilter(dimension, basis, leakage=0.1)
    coefficients = torch.randn(4, 3)
    protected_x = coefficients @ basis.transpose(0, 1)
    torch.testing.assert_close(
        filter_module(protected_x),
        0.1 * protected_x,
        atol=1e-5,
        rtol=1e-4,
    )


def test_rho_one_is_identity():
    basis = _orthonormal_basis(8, 4)
    filter_module = HardLeakyFilter(8, basis, leakage=1.0)
    x = torch.randn(2, 8)
    torch.testing.assert_close(filter_module(x), x)


def test_shared_filter_reuses_exact_qkv_projection_three_times():
    torch.manual_seed(41)
    basis = _orthonormal_basis(8, 3)
    filter_module = HardLeakyFilter(8, basis, leakage=0.02)
    filter_module.configure_runtime_reuse(3)
    x = torch.randn(2, 5, 8, requires_grad=True)
    first = filter_module(x)
    second = filter_module(x)
    third = filter_module(x)
    assert first is second
    assert first is third
    fourth = filter_module(x)
    assert fourth is not first
    torch.testing.assert_close(fourth, first)


def test_shared_qkv_projection_preserves_summed_input_gradient():
    torch.manual_seed(43)
    basis = _orthonormal_basis(9, 4)
    shared = HardLeakyFilter(9, basis, leakage=0.02)
    shared.configure_runtime_reuse(3)
    independent = HardLeakyFilter(9, basis, leakage=0.02)
    x_shared = torch.randn(2, 4, 9, requires_grad=True)
    x_independent = x_shared.detach().clone().requires_grad_(True)
    weights = [torch.randn(9, 7) for _ in range(3)]

    shared_loss = sum(
        (shared(x_shared) @ weight).square().mean() for weight in weights
    )
    independent_loss = sum(
        (independent(x_independent) @ weight).square().mean()
        for weight in weights
    )
    shared_loss.backward()
    independent_loss.backward()

    torch.testing.assert_close(
        shared_loss,
        independent_loss,
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        x_shared.grad,
        x_independent.grad,
        atol=1e-6,
        rtol=1e-5,
    )


def test_filter_builder_uses_inclusive_energy_threshold():
    moment = torch.diag(torch.tensor([1.0, 2.0, 3.0, 10.0]))
    result = build_hard_leaky_filter(
        moment,
        energy_fraction=0.20,
        leakage=0.02,
        ridge=0.0,
    )
    # 1 + 2 < 20% of 16, but 1 + 2 + 3 >= 20%, hence m=3.
    assert result.tail_dimension == 3
    assert result.protected_dimension == 1


def test_filter_builder_rejects_corrupt_second_moments():
    nonfinite = torch.eye(3)
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_hard_leaky_filter(nonfinite, ridge=0.0)

    indefinite = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="non-PSD"):
        build_hard_leaky_filter(indefinite, ridge=0.0)


def test_filter_builder_clamps_only_roundoff_scale_negative_eigenvalues():
    nearly_psd = torch.diag(torch.tensor([-1e-8, 1.0]))
    result = build_hard_leaky_filter(
        nearly_psd,
        energy_fraction=0.20,
        ridge=0.0,
    )
    assert torch.all(result.eigenvalues >= 0)
