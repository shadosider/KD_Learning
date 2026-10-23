"""集中管理实验超参数。

所有训练脚本从这里读取参数,避免各脚本里散落硬编码。
改动这里的数值即可全局生效,例如把 KD 训练轮数从 5 提到 20:

    CONFIG.kd_epochs -> 20
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- 通用训练设置 ----
    seed: int = 42
    batch_size: int = 128
    learning_rate: float = 1e-3

    # ---- 各阶段训练轮数 ----
    teacher_epochs: int = 10
    student_epochs: int = 10
    kd_epochs: int = 10
    ablation_epochs: int = 10

    # ---- 蒸馏损失超参(Hinton) ----
    kd_temperature: float = 4.0
    kd_alpha: float = 0.5

    # ---- 消融实验扫描网格 ----
    ablation_temperatures: tuple = (1, 2, 4, 8, 20)
    ablation_alphas: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)


CONFIG = Config()
