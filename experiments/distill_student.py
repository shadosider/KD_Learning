from torch.optim import Adam

from kdlib.data import get_mnist_loaders
from kdlib.models import create_teacher, create_student
from kdlib.training import (
    get_device,
    load_checkpoint,
    make_kd_loss_fn,
    run_training,
    save_checkpoint,
    set_seed,
)


def main():
    set_seed(42)
    device = get_device()

    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    # 加载已经训练好的 Teacher,只提供软目标,不更新参数
    teacher = create_teacher().to(device)
    load_checkpoint(teacher, "teacher.pt", device)
    teacher.eval()

    # 创建 Student,并从统一的初始权重出发
    student = create_student().to(device)
    load_checkpoint(student, "student_init.pt", device)

    optimizer = Adam(student.parameters(), lr=1e-3)

    loss_fn = make_kd_loss_fn(teacher, temperature=4.0, alpha=0.5)

    run_training(
        model=student,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=5,
        device=device,
    )

    save_checkpoint(student, "student_kd.pt")


if __name__ == "__main__":
    main()
