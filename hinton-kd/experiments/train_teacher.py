
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from src.data import get_mnist_loaders
from src.models import create_teacher


def main():
    # 固定随机种子，便于复现实验
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    train_loader, test_loader = get_mnist_loaders(
        batch_size=128
    )

    model = create_teacher().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 清除上一批次的梯度
            optimizer.zero_grad()

            # 前向传播
            logits = model(images)

            # 计算损失
            loss = criterion(logits, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # 在测试集上评估，不计算梯度
        model.eval()
        test_correct = 0
        test_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                predictions = logits.argmax(dim=1)

                test_correct += (
                    predictions == labels
                ).sum().item()
                test_samples += labels.size(0)

        test_acc = test_correct / test_samples

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"loss={train_loss:.4f} "
            f"train_acc={train_acc:.4%} "
            f"test_acc={test_acc:.4%}"
        )

    # 保存模型参数
    checkpoint_dir = Path("../outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "teacher.pt"
    torch.save(model.state_dict(), checkpoint_path)

    print(f"Teacher saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
