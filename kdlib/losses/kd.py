import torch
import torch.nn.functional as F

from .base import register


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
):
    """Hinton 2015 蒸馏损失。

    返回 (total, ce, kd),后两项已 detach,仅供日志观测。
    """
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


@register("kd")
def make_kd_loss_fn(teacher, temperature: float = 4.0, alpha: float = 0.5):
    """builder:构造蒸馏 loss_fn,软目标来自冻结的 teacher。"""

    def loss_fn(model, images, labels):
        # Teacher 只提供软目标,不参与梯度更新
        with torch.no_grad():
            teacher_logits = teacher(images)

        student_logits = model(images)

        loss, _, _ = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            temperature=temperature,
            alpha=alpha,
        )
        return loss, student_logits

    return loss_fn
