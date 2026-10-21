import argparse

from torch.optim import Adam

from kdlib.data import get_mnist_loaders
from kdlib.losses import ce_loss_fn
from kdlib.models import create_teacher
from kdlib.paths import teacher_checkpoint
from kdlib.training import (
    get_device,
    run_training,
    save_checkpoint,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="训练 teacher")
    parser.add_argument("--deep", action="store_true", help="使用深网络(512-256-128 + dropout)")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    model = create_teacher(deep=args.deep).to(device)
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

    save_checkpoint(model, teacher_checkpoint(args.deep))


if __name__ == "__main__":
    main()
