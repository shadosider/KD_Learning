from .base import REGISTRY, get_loss_fn, register
from .ce import ce_loss_fn, make_ce_loss_fn
from .kd import distillation_loss, make_kd_loss_fn

__all__ = [
    "REGISTRY",
    "register",
    "get_loss_fn",
    "ce_loss_fn",
    "make_ce_loss_fn",
    "distillation_loss",
    "make_kd_loss_fn",
]
