"""
MNIST 知识蒸馏：交互式观察温度 T 对 Softmax 概率分布的影响

运行：
    uv run python mnist_temperature_slider.py
或：
    python mnist_temperature_slider.py

依赖：
    pip install numpy matplotlib

说明：
    这里使用一组“示例教师 logits”模拟 MNIST 图片经过教师网络后的输出，
    不需要下载 MNIST 数据集或训练模型。想使用真实模型时，可替换 teacher_logits。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 示例：一张真实标签为数字 3 的 MNIST 图片经过教师模型后得到的 10 个 logits。
# 数字 3 的 logit 最高；数字 8 次之，模拟教师认为它有一点像 8。
digits = np.arange(10)
teacher_logits = np.array([
    -1.0, -0.5, 0.2, 5.0, 0.0,
     0.8, -0.8, -0.2, 2.2, -0.7
], dtype=float)


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """计算 softmax(logits / T)，减去最大值以提高数值稳定性。"""
    scaled = logits / temperature
    scaled = scaled - np.max(scaled)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum()


def main():
    initial_temperature = 1.0
    initial_probs = softmax_with_temperature(teacher_logits, initial_temperature)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.27)

    bars = ax.bar(digits, initial_probs, width=0.68)
    ax.set_xticks(digits)
    ax.set_xticklabels([str(d) for d in digits])
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)
    ax.set_title("MNIST Knowledge Distillation: Effect of Temperature T")
    ax.grid(axis="y", alpha=0.25)

    info = ax.text(
        0.02, 0.96, "", transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9)
    )

    slider_ax = fig.add_axes([0.18, 0.11, 0.64, 0.045])
    temperature_slider = Slider(
        ax=slider_ax,
        label="Temperature T",
        valmin=0.2,
        valmax=8.0,
        valinit=initial_temperature,
        valstep=0.1,
    )

    def update(_):
        temperature = temperature_slider.val
        probs = softmax_with_temperature(teacher_logits, temperature)

        for bar, probability in zip(bars, probs):
            bar.set_height(probability)

        info.set_text(
            f"T = {temperature:.1f}\n"
            f"Top predicted class: {int(np.argmax(probs))}\n"
            f"P(digit 3) = {probs[3]:.3f}    "
            f"P(digit 8) = {probs[8]:.3f}"
        )
        fig.canvas.draw_idle()

    temperature_slider.on_changed(update)
    update(None)
    plt.show()


if __name__ == "__main__":
    main()

