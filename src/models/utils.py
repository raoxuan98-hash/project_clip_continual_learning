import torch
import torch.nn.functional as F

class EMASmooth:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

    def get(self):
        return self.value if self.value is not None else 0.0


def feature_distillation_loss(
    teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
    teacher_feat = F.normalize(teacher_feat, dim=-1)
    student_feat = F.normalize(student_feat, dim=-1)
    cosine_sim = (teacher_feat * student_feat).sum(dim=-1)
    return (1 - cosine_sim).mean()


def _logits_and_probs_bidir(logit_scale, img_feat, text_feat, temperature):
    """双向 softmax 分布：I2T（行方向）与 T2I（列方向）。

    旧的单向实现只约束 image→text 分布，会让 text→image 方向的检索几何
    在训练中无约束漂移；双向分布与 CLIP 原始对称对比目标同构。
    """
    logits = logit_scale.exp() * (img_feat @ text_feat.t()) / temperature
    log_probs_i2t = F.log_softmax(logits, dim=-1)
    probs_i2t = logits.softmax(dim=-1)
    log_probs_t2i = F.log_softmax(logits, dim=-2)
    probs_t2i = logits.softmax(dim=-2)
    return log_probs_i2t, probs_i2t, log_probs_t2i, probs_t2i


def _directional_divergence(s_log_probs, s_probs, t_log_probs, t_probs,
                            temperature, divergence):
    """单方向上的蒸馏散度。kl_forward / kl_reverse / js 带 τ² 因子。"""
    if divergence == "kl_forward":
        # D_KL(teacher || student)
        return F.kl_div(input=s_log_probs, target=t_probs,
                        reduction="batchmean") * (temperature ** 2)
    if divergence == "kl_reverse":
        # D_KL(student || teacher)。显式展开：F.kl_div 不向 target 回传梯度，
        # 旧写法（input=t_log_probs, target=s_probs）对学生不产生梯度。
        return ((s_probs * (s_log_probs - t_log_probs)).sum(dim=-1).mean()
                * (temperature ** 2))
    if divergence == "js":
        m_probs = 0.5 * (t_probs + s_probs)
        m_log_probs = torch.log(m_probs.clamp_min(1e-12))
        kl_t_m = F.kl_div(input=m_log_probs, target=t_probs,
                          reduction="batchmean")
        kl_s_m = (s_probs * (s_log_probs - m_log_probs)).sum(dim=-1).mean()
        return 0.5 * (kl_t_m + kl_s_m) * (temperature ** 2)
    if divergence == "mse":
        return F.mse_loss(s_probs, t_probs, reduction="mean")
    if divergence == "cosine":
        return 1.0 - F.cosine_similarity(
            s_probs.view(s_probs.size(0), -1),
            t_probs.view(t_probs.size(0), -1),
            dim=-1).mean()
    if divergence == "l1":
        return (s_probs - t_probs).abs().mean()
    raise ValueError(f"Unsupported divergence: {divergence}")


def cross_modal_distillation_loss(logit_scale: torch.Tensor,
                                  student_img_feat: torch.Tensor,
                                  student_text_feat: torch.Tensor,
                                  teacher_img_feat: torch.Tensor,
                                  teacher_text_feat: torch.Tensor,
                                  temperature: float = 2.0,
                                  divergence: str = "kl_forward",
                                  direction: str = "bidir"):
    """
    Cross-modal distillation between teacher and student CLIP-like models.

    Args:
        divergence: one of
            - kl_forward:  D_KL(teacher || student)  (standard forward KL)
            - kl_reverse:  D_KL(student || teacher)
            - js:          Jensen-Shannon divergence
            - mse:         Mean squared error on probabilities
            - cosine:      1 - cosine similarity on probabilities
            - l1:          L1 distance on probabilities
        direction: "bidir" = 0.5*(I2T + T2I)（默认，c14255e 修复后行为）；
                   "i2t_only" = 仅 image->text 单向、不加 0.5 缩放（v2 旧行为，
                   用于 CD 方向 A/B 对照）。
    """
    s_lp_i2t, s_p_i2t, s_lp_t2i, s_p_t2i = _logits_and_probs_bidir(
        logit_scale, student_img_feat, student_text_feat, temperature)

    with torch.no_grad():
        t_lp_i2t, t_p_i2t, t_lp_t2i, t_p_t2i = _logits_and_probs_bidir(
            logit_scale, teacher_img_feat, teacher_text_feat, temperature)

    loss_i2t = _directional_divergence(
        s_lp_i2t, s_p_i2t, t_lp_i2t, t_p_i2t, temperature, divergence)
    if direction == "i2t_only":
        return loss_i2t
    if direction != "bidir":
        raise ValueError(f"Unsupported cd direction: {direction}")
    loss_t2i = _directional_divergence(
        s_lp_t2i, s_p_t2i, t_lp_t2i, t_p_t2i, temperature, divergence)
    return 0.5 * (loss_i2t + loss_t2i)
