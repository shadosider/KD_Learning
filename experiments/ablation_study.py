import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from torch.optim import Adam

from kdlib.data import get_mnist_loaders
from kdlib.models import create_student, create_teacher
from kdlib.paths import PROJECT_ROOT
from kdlib.training import (
    get_device,
    load_checkpoint,
    make_kd_loss_fn,
    run_training,
    set_seed,
)

TEMPERATURES = [1, 2, 4, 8, 20]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
EPOCHS = 5

# 5 条温度线的类别色(浅色模式,固定顺序:blue/orange/aqua/yellow/magenta)
LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# 图表配色(浅色表面)
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"

# 顺序蓝色 ramp(heatmap,浅->深)
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def run_sweep(device, train_loader, test_loader, teacher):
    """扫描 (temperature, alpha) 组合,返回 (matrix, records)。"""
    matrix = [[0.0] * len(ALPHAS) for _ in TEMPERATURES]
    records = []
    total = len(TEMPERATURES) * len(ALPHAS)
    done = 0

    for i, T in enumerate(TEMPERATURES):
        for j, a in enumerate(ALPHAS):
            # 每个组合都从同一初始权重出发,数据顺序也一致
            set_seed(42)
            student = create_student().to(device)
            load_checkpoint(student, "student_init.pt", device)

            optimizer = Adam(student.parameters(), lr=1e-3)
            loss_fn = make_kd_loss_fn(teacher, temperature=T, alpha=a)

            acc = run_training(
                student,
                train_loader,
                test_loader,
                loss_fn,
                optimizer,
                epochs=EPOCHS,
                device=device,
                verbose=False,
            )
            acc = round(acc, 4)
            matrix[i][j] = acc
            records.append({"temperature": T, "alpha": a, "accuracy": acc})

            done += 1
            print(f"[{done:>2}/{total}] T={T:>2}  α={a:.1f}  test_acc={acc:.4%}")

    return matrix, records


def text_color_for(rgba):
    """根据单元格颜色的相对亮度,返回白字或深字。"""
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#ffffff" if lum < 0.55 else PRIMARY_INK


def plot_ablation(matrix, baseline_acc, figure_path):
    matrix_pct = [[v * 100 for v in row] for row in matrix]
    vmin = min(min(r) for r in matrix_pct)
    vmax = max(max(r) for r in matrix_pct)
    if vmax - vmin < 1e-6:
        vmax = vmin + 0.01

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

    fig, (ax_h, ax_l) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    fig.patch.set_facecolor(SURFACE)

    # ---- 左:heatmap ----
    im = ax_h.imshow(matrix_pct, cmap=cmap, norm=norm, aspect="auto")
    ax_h.set_xticks(range(len(ALPHAS)))
    ax_h.set_xticklabels([f"{a:.1f}" for a in ALPHAS], color=PRIMARY_INK)
    ax_h.set_yticks(range(len(TEMPERATURES)))
    ax_h.set_yticklabels([str(t) for t in TEMPERATURES], color=PRIMARY_INK)
    ax_h.set_xlabel("α (KD weight)", color=SECONDARY_INK)
    ax_h.set_ylabel("Temperature T", color=SECONDARY_INK)
    ax_h.set_title("KD student accuracy (%)", color=PRIMARY_INK, loc="left")
    ax_h.tick_params(colors=MUTED_INK, length=0)

    for i in range(len(TEMPERATURES)):
        for j in range(len(ALPHAS)):
            v = matrix_pct[i][j]
            ax_h.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                color=text_color_for(cmap(norm(v))), fontsize=9,
            )

    cb = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    cb.set_label("accuracy (%)", color=SECONDARY_INK)
    cb.ax.tick_params(colors=MUTED_INK)

    # ---- 右:accuracy vs α ----
    for k, T in enumerate(TEMPERATURES):
        ys = [matrix[k][j] * 100 for j in range(len(ALPHAS))]
        ax_l.plot(
            ALPHAS, ys, marker="o", markersize=6, linewidth=2,
            color=LINE_COLORS[k], label=f"T={T}",
        )

    if baseline_acc is not None:
        ax_l.axhline(
            baseline_acc * 100, color=MUTED_INK, linestyle="--", linewidth=1.2,
        )
        ax_l.text(
            0.02, baseline_acc * 100, f"  baseline (α=0) {baseline_acc * 100:.2f}%",
            va="bottom", ha="left", color=SECONDARY_INK, fontsize=9,
        )

    ax_l.set_xlabel("α (KD weight)", color=SECONDARY_INK)
    ax_l.set_ylabel("Test accuracy (%)", color=SECONDARY_INK)
    ax_l.set_title("Accuracy vs α", color=PRIMARY_INK, loc="left")
    ax_l.set_xlim(min(ALPHAS) - 0.05, max(ALPHAS) + 0.05)
    ax_l.grid(color=GRIDLINE, linewidth=0.8)
    ax_l.set_axisbelow(True)
    ax_l.tick_params(colors=MUTED_INK, length=0)
    for spine in ("top", "right"):
        ax_l.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax_l.spines[spine].set_color(MUTED_INK)

    ax_l.legend(title="temperature", frameon=False, fontsize=9)

    fig.suptitle(
        "Knowledge Distillation ablation: temperature × α",
        color=PRIMARY_INK, fontsize=13, x=0.05, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("Figure saved to:", figure_path)


def load_baseline_accuracy():
    """从之前的结果里读取基线学生(α=0)准确率,作为参考线。"""
    results_path = PROJECT_ROOT / "outputs" / "results.json"
    if results_path.exists():
        data = json.loads(results_path.read_text())
        for key, val in data.items():
            if "Student (64)" in key and "KD" not in key:
                return val["accuracy"]
    return None


def main():
    set_seed(42)
    device = get_device()

    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    # 加载冻结的 teacher
    teacher = create_teacher().to(device)
    load_checkpoint(teacher, "teacher.pt", device)
    teacher.eval()

    matrix, records = run_sweep(device, train_loader, test_loader, teacher)

    # 保存原始数据
    results_path = PROJECT_ROOT / "outputs" / "ablation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    # 汇总最好/最差组合
    best = max(records, key=lambda r: r["accuracy"])
    worst = min(records, key=lambda r: r["accuracy"])
    print(f"\n最好: T={best['temperature']}, α={best['alpha']} -> {best['accuracy']:.4%}")
    print(f"最差: T={worst['temperature']}, α={worst['alpha']} -> {worst['accuracy']:.4%}")

    baseline_acc = load_baseline_accuracy()
    if baseline_acc is not None:
        print(f"基线(α=0, 无蒸馏): {baseline_acc:.4%}")

    figure_path = PROJECT_ROOT / "outputs" / "figures" / "ablation_heatmap.png"
    plot_ablation(matrix, baseline_acc, figure_path)
    print("Results saved to:", results_path)


if __name__ == "__main__":
    main()
