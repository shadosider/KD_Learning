from torch.optim import Adam

from kdlib.data import get_mnist_loaders
from kdlib.models import create_student
from kdlib.training import (
    ce_loss_fn,
    get_device,
    load_checkpoint,
    run_training,
    save_checkpoint,
    set_seed,
)


def main():
    set_seed(42)
    device = get_device()

    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    model = create_student().to(device)

    # 导入统一的初始权重,保证与蒸馏学生从同一起点出发
    load_checkpoint(model, "student_init.pt", device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=ce_loss_fn,
        optimizer=optimizer,
        epochs=5,
        device=device,
    )

    save_checkpoint(model, "student_baseline.pt")


if __name__ == "__main__":
    main()
