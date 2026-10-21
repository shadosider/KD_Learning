
import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
):
    # 学生对真实标签的分类损失
    ce_loss = F.cross_entropy(student_logits, labels)

    # 温度缩放后的软概率分布
    student_log_probs = F.log_softmax(
        student_logits / temperature, dim=1
    )
    teacher_probs = F.softmax(
        teacher_logits / temperature, dim=1
    )

    # KL(Teacher || Student)
    kd_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature ** 2)

    total_loss = (1 - alpha) * ce_loss + alpha * kd_loss

    return total_loss, ce_loss.detach(), kd_loss.detach()
