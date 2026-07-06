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


def _logits_and_probs(logit_scale, img_feat, text_feat, temperature):
    logits = logit_scale.exp() * (img_feat @ text_feat.t())
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return logits, log_probs, probs


def cross_modal_distillation_loss(logit_scale: torch.Tensor,
                                  student_img_feat: torch.Tensor,
                                  student_text_feat: torch.Tensor,
                                  teacher_img_feat: torch.Tensor,
                                  teacher_text_feat: torch.Tensor,
                                  temperature: float = 2.0,
                                  divergence: str = "kl_forward"):
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
    """
    _, s_log_probs, s_probs = _logits_and_probs(
        logit_scale, student_img_feat, student_text_feat, temperature)

    with torch.no_grad():
        _, t_log_probs, t_probs = _logits_and_probs(
            logit_scale, teacher_img_feat, teacher_text_feat, temperature)

    if divergence == "kl_forward":
        loss = F.kl_div(input=s_log_probs, target=t_probs,
                        reduction="batchmean") * (temperature ** 2)

    elif divergence == "kl_reverse":
        # D_KL(student || teacher); keep target=s_probs so gradients flow to student.
        loss = F.kl_div(input=t_log_probs, target=s_probs,
                        reduction="batchmean") * (temperature ** 2)

    elif divergence == "js":
        # JS(P=teacher, Q=student) = 0.5 * KL(teacher || M) + 0.5 * KL(student || M)
        # with M = 0.5 * (teacher + student).  Student probabilities must stay in the
        # computational graph for gradients to propagate.
        m_probs = 0.5 * (t_probs + s_probs)
        m_log_probs = torch.log(m_probs.clamp_min(1e-12))
        kl_t_m = F.kl_div(input=m_log_probs, target=t_probs,
                          reduction="batchmean")
        kl_s_m = F.kl_div(input=m_log_probs, target=s_probs,
                          reduction="batchmean")
        loss = 0.5 * (kl_t_m + kl_s_m) * (temperature ** 2)

    elif divergence == "mse":
        loss = F.mse_loss(s_probs, t_probs, reduction="mean")

    elif divergence == "cosine":
        # cosine similarity on flattened probability distributions
        loss = 1.0 - F.cosine_similarity(
            s_probs.view(s_probs.size(0), -1),
            t_probs.view(t_probs.size(0), -1),
            dim=-1).mean()

    elif divergence == "l1":
        loss = (s_probs - t_probs).abs().mean()

    else:
        raise ValueError(f"Unsupported divergence: {divergence}")

    return loss
