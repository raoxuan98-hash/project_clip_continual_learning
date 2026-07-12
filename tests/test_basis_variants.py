"""
验证 basis 参数化变体和 LoRA-Null 初始化在 SGPBaseLoRA/DoRA 下的正确性。
运行：python test_basis_variants.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from src.models.lora_sgp import (
    SGPBaseLoRA,
    SGPBaseDoRA,
    FixedProjection,
    compute_tail_basis,
)


def make_cov(d_in, k):
    """构造一个前 k 个方向为低能量的协方差矩阵。"""
    cov = torch.eye(d_in)
    cov[:k, :k] *= 1e-6
    return cov


def make_module(lora_class, mode, d_in=768, d_out=768, r=4, k=16):
    linear = nn.Linear(d_in, d_out, bias=False)
    with torch.no_grad():
        linear.weight.normal_(0, 0.02)
    proj = FixedProjection(torch.eye(d_in))
    return lora_class(linear, r, proj, projection_param_mode=mode, basis_rank=k)


def test_mode(lora_class, mode, name):
    print(f"\n[{name}]")
    d_in, d_out, r, k = 768, 768, 4, 16
    module = make_module(lora_class, mode, d_in, d_out, r, k)

    # 模拟 Task 1 训练：basis_ready=False 时应走 full 模式
    with torch.no_grad():
        module.A.normal_(0, 0.01)
        module.B.normal_(0, 0.01)

    x = torch.randn(8, d_in)
    out_before = module(x)

    # merge，然后检查 basis_ready=False 时 forward 仍正确
    module.merge_lora_weights()
    out_after_merge = module(x)

    err = (out_before - out_after_merge).abs().max().item()
    print(f"  forward before/after merge (full fallback): {err:.6e}")
    assert err < 1e-3, f"{name}: merge changed output in full fallback!"

    # 模拟 Task 2：设置 U_h 并 basis_ready=True
    cov = make_cov(d_in, k)
    U_h = compute_tail_basis(cov, k)
    module.U_h = U_h.to(device=module.A.device, dtype=module.A.dtype)
    module.basis_ready = torch.tensor(True)

    # 给 basis 参数一些非零值
    with torch.no_grad():
        if mode == "fixed_basis":
            module.B_basis.normal_(0, 0.01)
        elif mode == "core_basis":
            module.C.normal_(0, 0.01)
            module.B.normal_(0, 0.01)

    # 检查 delta 的秩（在 merge 之前）
    with torch.no_grad():
        if mode == "fixed_basis":
            delta = module.B_basis @ module.U_h.T
            rank = torch.linalg.matrix_rank(delta).item()
            print(f"  effective delta rank: {rank} (expected <= {k})")
            assert rank <= k
        elif mode == "core_basis":
            delta = module.B @ module.C @ module.U_h.T
            rank = torch.linalg.matrix_rank(delta).item()
            print(f"  effective delta rank: {rank} (expected <= {r})")
            assert rank <= r

    out_basis_before = module(x)
    module.merge_lora_weights()
    out_basis_after = module(x)

    err = (out_basis_before - out_basis_after).abs().max().item()
    print(f"  forward before/after merge (basis mode):    {err:.6e}")
    assert err < 1e-3, f"{name}: merge changed output in basis mode!"

    print(f"  PASSED")


def test_init_for_mode(lora_class, mode, name):
    print(f"\n[{name} - init]")
    d_in, d_out, r, k = 768, 768, 4, 16
    module = make_module(lora_class, mode, d_in, d_out, r, k)

    # merge 后模拟 Task 2 初始状态
    module.merge_lora_weights()
    W_before = module.get_effective_weight().clone()

    cov = make_cov(d_in, k)
    U_h = compute_tail_basis(cov, k)
    module.U_h = U_h.to(device=module.A.device, dtype=module.A.dtype)
    module.basis_ready = torch.tensor(True)

    from src.models.lora_sgp import compute_proj_init
    A_init, B_init, W_residual = compute_proj_init(W_before, cov, r, window="tail")

    module.set_effective_weight(W_residual)
    module.set_lora_init(A_init, B_init)

    x = torch.randn(8, d_in)
    out_before = module(x)

    # 恢复 W_before 并直接前向
    with torch.no_grad():
        if isinstance(module, SGPBaseDoRA):
            module.set_effective_weight(W_before)
            module.B.zero_()
            if mode == "core_basis":
                module.C.zero_()
            elif mode == "fixed_basis":
                module.B_basis.zero_()
        else:
            module.linear.weight.data.copy_(W_before)
            module.B.zero_()
            if mode == "core_basis":
                module.C.zero_()
            elif mode == "fixed_basis":
                module.B_basis.zero_()

    out_reference = module(x)

    err = (out_before - out_reference).abs().max().item()
    print(f"  init consistency error: {err:.6e}")
    assert err < 1e-3, f"{name}: init changed function!"
    print(f"  PASSED")


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 70)
    print("Testing SGPBaseLoRA")
    print("=" * 70)
    test_mode(SGPBaseLoRA, "fixed_basis", "SGPBaseLoRA_fixed_basis")
    test_mode(SGPBaseLoRA, "core_basis", "SGPBaseLoRA_core_basis")
    test_init_for_mode(SGPBaseLoRA, "fixed_basis", "SGPBaseLoRA_fixed_basis")
    test_init_for_mode(SGPBaseLoRA, "core_basis", "SGPBaseLoRA_core_basis")

    print("\n" + "=" * 70)
    print("Testing SGPBaseDoRA")
    print("=" * 70)
    test_mode(SGPBaseDoRA, "fixed_basis", "SGPBaseDoRA_fixed_basis")
    test_mode(SGPBaseDoRA, "core_basis", "SGPBaseDoRA_core_basis")
    test_init_for_mode(SGPBaseDoRA, "fixed_basis", "SGPBaseDoRA_fixed_basis")
    test_init_for_mode(SGPBaseDoRA, "core_basis", "SGPBaseDoRA_core_basis")

    print("\nAll tests passed!")
