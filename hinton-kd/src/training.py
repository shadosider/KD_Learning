import random

import torch
import torch.nn.functional as F

from .losses import distillation_loss
from .paths import checkpoint_path


def set_seed(seed: int = 42):
    """固定随机种子,便于复现实验。"""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


def ce_loss_fn(model, images, labels):
    """硬标签分类损失,返回 (loss, logits),便于训练循环统计准确率。"""
    logits = model(images)
    return F.cross_entropy(logits, labels), logits


def make_kd_loss_fn(teacher, temperature: float = 4.0, alpha: float = 0.5):
    """构造蒸馏损失函数:软目标来自冻结的 teacher,返回 (loss, student_logits)。"""

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


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = loss_fn(model, images, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    return total_correct / total_samples


def run_training(model, train_loader, test_loader, loss_fn, optimizer, epochs, device, verbose=True):
    """训练并逐 epoch 评估,打印损失与准确率,返回最终测试准确率。"""
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        test_acc = evaluate(model, test_loader, device)

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"loss={train_loss:.4f} "
                f"train_acc={train_acc:.4%} "
                f"test_acc={test_acc:.4%}"
            )

    return test_acc


def save_checkpoint(model, name: str):
    path = checkpoint_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved to: {path}")
    return path


def load_checkpoint(model, name: str, device):
    path = checkpoint_path(name)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    print(f"Loaded weights from: {path}")
    return path
