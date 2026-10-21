"""损失函数注册表。

契约(每个注册项返回的 loss_fn):
    loss_fn(model, images, labels) -> (loss, logits)

注册表中的每一项是 *builder*,接收 **config,返回满足上述契约的 loss_fn。
统一 runner 通过 ``get_loss_fn(name, **config)`` 按名字构造损失,
后续 KD 变体(FitNet / AT / PKT / RKD / DKD)只需在各自模块里 ``@register`` 即可接入。
"""
from typing import Callable

REGISTRY: dict[str, Callable] = {}


def register(name: str) -> Callable:
    """装饰器:把 builder 函数登记到 REGISTRY[name]。"""

    def decorator(builder: Callable) -> Callable:
        REGISTRY[name] = builder
        return builder

    return decorator


def get_loss_fn(name: str, **config) -> Callable:
    """按名字从注册表构造 loss_fn,config 透传给对应 builder。"""
    if name not in REGISTRY:
        raise KeyError(f"未知损失 {name!r},可用: {sorted(REGISTRY)}")
    return REGISTRY[name](**config)
