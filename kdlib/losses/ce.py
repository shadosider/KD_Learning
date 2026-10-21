import torch.nn.functional as F

from .base import register


def ce_loss_fn(model, images, labels):
    """硬标签分类损失,返回 (loss, logits),便于训练循环统计准确率。"""
    logits = model(images)
    return F.cross_entropy(logits, labels), logits


@register("ce")
def make_ce_loss_fn():
    """builder:构造硬标签 CE loss_fn(无额外配置)。"""
    return ce_loss_fn
