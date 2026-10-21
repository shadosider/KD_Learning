import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from kdlib.data import get_mnist_loaders
from kdlib.metrics import count_parameters, measure_latency
from kdlib.models import create_student, create_teacher
from kdlib.paths import (
    PROJECT_ROOT,
    checkpoint_path,
    deep_suffix,
    kd_checkpoint,
    teacher_checkpoint,
)
from kdlib.training import evaluate, get_device, load_checkpoint, set_seed


def build_models(deep):
    """按 deep 模式返回 (显示名, 权重文件, 构造器) 列表。"""
    teacher_label = "Teacher (deep)" if deep else "Teacher (512)"
    kd_label = "Student KD (64, deep T)" if deep else "Student KD (64)"
    return [
        (teacher_label, teacher_checkpoint(deep), lambda: create_teacher(deep=deep)),
        ("Student (64)", "student_baseline.pt", create_student),
        (kd_label, kd_checkpoint(deep), create_student),
    ]

# 类别色(浅色模式),固定顺序:blue / orange / aqua
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

# 图表配色(浅色表面)
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"


def collect_metrics(device, models):
    _, test_loader = get_mnist_loaders(batch_size=128)

    results = {}
    for name, ckpt, factory in models:
        model = factory().to(device)
        load_checkpoint(model, ckpt, device)

        ms_per_batch, ms_per_image = measure_latency(
            model, test_loader, device, warmup_batches=5, max_batches=60
        )

        results[name] = {
            "accuracy": round(evaluate(model, test_loader, device), 4),
            "params": count_parameters(model),
            "checkpoint_bytes": checkpoint_path(ckpt).stat().st_size,
            "latency_ms_per_batch": round(ms_per_batch, 4),
            "latency_ms_per_image": round(ms_per_image, 4),
        }
    return results


def make_figure(results, figure_path, models):
    names = [name for name, _, _ in models]
    y = list(range(len(names)))[::-1]  # 顶部为 Teacher

    panels = [
        (
            "Test accuracy (%)",
            [results[n]["accuracy"] * 100 for n in names],
            lambda v: f"{v:.2f}%",
            100.0,
        ),
        (
            "Parameters",
            [results[n]["params"] for n in names],
            lambda v: f"{v:,}",
            None,
        ),
        (
            "Latency (ms / batch)",
            [results[n]["latency_ms_per_batch"] for n in names],
            lambda v: f"{v:.3f}",
            None,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(
        "Knowledge Distillation on MNIST",
        color=PRIMARY_INK,
        fontsize=13,
        x=0.05,
        ha="left",
    )

    for ax, (title, values, fmt, fixed_xmax) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.barh(y, values, color=COLORS, height=0.55, zorder=3)

        for yi, v in zip(y, values):
            ax.text(
                v, yi, f"  {fmt(v)}",
                va="center", ha="left", color=SECONDARY_INK, fontsize=10,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(names, color=PRIMARY_INK, fontsize=10)
        ax.set_title(title, color=PRIMARY_INK, fontsize=11, loc="left")

        xmax = fixed_xmax if fixed_xmax is not None else max(values) * 1.3
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(MUTED_INK)
        ax.tick_params(colors=MUTED_INK, length=0)
        ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("Figure saved to:", figure_path)


def main():
    parser = argparse.ArgumentParser(description="对照可视化(准确率/参数量/延迟)")
    parser.add_argument("--deep", action="store_true", help="读取深模式产物(_deep 后缀)")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    models = build_models(args.deep)
    results = collect_metrics(device, models)

    # 保存数值结果,便于复现与查阅
    results_path = PROJECT_ROOT / "outputs" / f"results{deep_suffix(args.deep)}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n结果汇总:")
    print("模型            | 测试准确率 | 参数量   | 权重文件   | 每批延迟")
    for name, ckpt, _ in models:
        r = results[name]
        print(
            f"{name:<15} | {r['accuracy'] * 100:.2f}%     | "
            f"{r['params']:>7,} | {r['checkpoint_bytes'] / 1024:>7.1f}KB | "
            f"{r['latency_ms_per_batch']:.4f} ms"
        )

    figure_path = PROJECT_ROOT / "outputs" / "figures" / f"results_comparison{deep_suffix(args.deep)}.png"
    make_figure(results, figure_path, models)
    print("Results saved to:", results_path)


if __name__ == "__main__":
    main()
