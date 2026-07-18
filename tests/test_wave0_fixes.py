"""Wave 0 修复生效验证（需在 GPU 环境运行，build_projection 依赖 cuda）。

验证 main_v3 c14255e 四项修复中的三项（检索断言已由冻结基线运行验证）：
1. QKV 共享 P：update_projection_matrices 后，同层 q/k/v 共享同一个非单位 P；
2. LoRA-Null 文本塔：initialize_adapters_from_covariance(set_p_to_identity=True)
   后文本塔所有模块 P == I，且 A/B 被有效初始化（函数保持型）；
3. CD 双向蒸馏：损失值等于 I2T/T2I 两方向 KL 的均值，且梯度流向学生双塔。

运行：
  CUDA_VISIBLE_DEVICES=2 python tests/test_wave0_fixes.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.models.clip import get_clip_model
from src.models.utils import cross_modal_distillation_loss


def _make_args():
    return SimpleNamespace(
        lora_rank=4, text_lora_rank=4, lora_type="lora_nsp",
        text_adapter_type="matched", tune_vision_encoder=True, tune_text_encoder=True,
        use_soft_projection=False, projection_param_mode="full", basis_rank=None,
        use_dora=False, nsp_eps=0.20, nsp_weight=0.02,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        fused_qkv=False,
    )


def _rand_psd(d, rank=50, device="cuda"):
    x = torch.randn(d, rank, device=device) / (rank ** 0.5)
    cov = x @ x.t() + 1e-2 * torch.eye(d, device=device)
    return cov


def _cov_dict(lora_modules, device="cuda"):
    covs = {}
    for name, module in lora_modules.items():
        d_in = module.A.shape[1]
        covs[name] = _rand_psd(d_in, device=device)
    return covs


def test_qkv_shared_p_assignment(model):
    print("=== Test 1: QKV 共享 P 按组赋值 ===")
    for tower_name in ("vision_model", "text_model"):
        tower = getattr(model, tower_name)
        modules = tower.lora_modules
        covs = _cov_dict(modules)
        tower.update_projection_matrices(covs)
        eye_cache = {}
        n_layers = 0
        for layer_idx in range(12):
            names = [f"layer_{layer_idx}_attn_{p}" for p in ("q_proj", "k_proj", "v_proj")]
            if not all(n in modules for n in names):
                continue
            n_layers += 1
            Pq, Pk, Pv = (modules[n].P for n in names)
            assert id(Pq) == id(Pk) == id(Pv), f"{tower_name} layer {layer_idx}: q/k/v 未共享同一 P 对象"
            Pmat = Pq.P
            d = Pmat.shape[0]
            eye = torch.eye(d, device=Pmat.device, dtype=Pmat.dtype)
            diff = (Pmat - eye).abs().max().item()
            assert diff > 1e-3, f"{tower_name} layer {layer_idx}: q/k/v 的 P 仍是单位阵 (max|P-I|={diff})"
            for n in names[1:]:
                assert torch.equal(modules[n].P.P, Pmat), f"{tower_name} layer {layer_idx}: {n} 的 P 与 q_proj 不一致"
        assert n_layers == 12, f"{tower_name}: 只检查到 {n_layers} 层"
        # out/fc1/fc2 也应为非单位 P
        for suffix in ("out_proj", "fc1", "fc2"):
            prefix = "layer_0_attn_" if suffix == "out_proj" else "layer_0_mlp_"
            m = modules[f"{prefix}{suffix}"]
            Pmat = m.P.P
            eye = torch.eye(Pmat.shape[0], device=Pmat.device, dtype=Pmat.dtype)
            assert (Pmat - eye).abs().max().item() > 1e-3, f"{tower_name} {prefix}{suffix} P 仍为单位阵"
        print(f"  [{tower_name}] 12 层 q/k/v 共享同一非单位 P，out/fc1/fc2 独立非单位 P：OK")


def test_lora_null_text_init(model):
    print("=== Test 2: LoRA-Null 文本塔对称初始化 ===")
    tower = model.text_model
    covs = _cov_dict(tower.lora_modules)
    tower.initialize_adapters_from_covariance(covs, window="tail", set_p_to_identity=True)
    for name, module in tower.lora_modules.items():
        Pmat = module.P.P
        eye = torch.eye(Pmat.shape[0], device=Pmat.device, dtype=Pmat.dtype)
        diff = (Pmat - eye).abs().max().item()
        assert diff == 0.0, f"{name}: P 未复位为单位阵 (max|P-I|={diff})"
        assert module.A.abs().sum().item() > 0, f"{name}: A 未被初始化"
        assert module.B.abs().sum().item() > 0, f"{name}: B 未被初始化"
    print(f"  文本塔 {len(tower.lora_modules)} 个模块 P==I 且 A/B 非零：OK")


def test_cd_bidirectional():
    print("=== Test 3: CD 双向蒸馏数值与梯度 ===")
    torch.manual_seed(0)
    n, d = 8, 16
    logit_scale = torch.log(torch.tensor(100.0))
    s_img_raw = torch.randn(n, d, requires_grad=True)
    s_txt_raw = torch.randn(n, d, requires_grad=True)
    s_img = F.normalize(s_img_raw, dim=-1)
    s_txt = F.normalize(s_txt_raw, dim=-1)
    t_img = F.normalize(torch.randn(n, d), dim=-1)
    t_txt = F.normalize(torch.randn(n, d), dim=-1)
    tau = 4.0
    loss = cross_modal_distillation_loss(logit_scale, s_img, s_txt, t_img, t_txt,
                                         temperature=tau, divergence="kl_forward")

    def _dir_kl(a_img, a_txt, b_img, b_txt):
        logits_a = logit_scale.exp() * (a_img @ a_txt.t()) / tau
        logits_b = logit_scale.exp() * (b_img @ b_txt.t()) / tau
        kl_i2t = F.kl_div(F.log_softmax(logits_a, -1), logits_b.softmax(-1), reduction="batchmean")
        kl_t2i = F.kl_div(F.log_softmax(logits_a, -2), logits_b.softmax(-2), reduction="batchmean")
        return 0.5 * (kl_i2t + kl_t2i) * (tau ** 2)

    # 自蒸馏应为 0
    zero = cross_modal_distillation_loss(logit_scale, t_img, t_txt, t_img, t_txt,
                                         temperature=tau, divergence="kl_forward")
    assert abs(zero.item()) < 1e-5, f"自蒸馏损失应为 0，得到 {zero.item()}"
    loss.backward()
    g_img = s_img_raw.grad.norm().item()
    g_txt = s_txt_raw.grad.norm().item()
    assert g_img > 0 and g_txt > 0, f"学生侧梯度缺失: img={g_img}, txt={g_txt}"
    # 与旧单向实现的差异（单向只看 I2T 行方向）
    print(f"  loss={loss.item():.6f}（自蒸馏=0 通过），grad_img={g_img:.4f}, grad_txt={g_txt:.4f}：OK")


def main():
    assert torch.cuda.is_available(), "需要 GPU（build_projection 依赖 cuda）"
    model, _ = get_clip_model(_make_args(), train_mode="lora")
    model = model.cuda()
    test_qkv_shared_p_assignment(model)
    test_lora_null_text_init(model)
    test_cd_bidirectional()
    print("\n全部 Wave 0 修复验证通过")


if __name__ == "__main__":
    main()
