import argparse

from torch.optim import Adam

from kdlib.config import CONFIG
from kdlib.data import get_mnist_loaders
from kdlib.losses import make_kd_loss_fn
from kdlib.models import create_teacher, create_student
from kdlib.paths import kd_checkpoint, teacher_checkpoint
from kdlib.training import (
    get_device,
    load_checkpoint,
    run_training,
    save_checkpoint,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description="知识蒸馏训练学生")
    parser.add_argument("--deep", action="store_true", help="使用深 teacher 蒸馏,写 _deep 后缀产物")
    args = parser.parse_args()

    set_seed(CONFIG.seed)
    device = get_device()

    train_loader, test_loader = get_mnist_loaders(batch_size=CONFIG.batch_size)

    # 加载已经训练好的 Teacher,只提供软目标,不更新参数
    teacher = create_teacher(deep=args.deep).to(device)
    load_checkpoint(teacher, teacher_checkpoint(args.deep), device)
    teacher.eval()

    # 创建 Student,并从统一的初始权重出发
    student = create_student().to(device)
    load_checkpoint(student, "student_init.pt", device)

    optimizer = Adam(student.parameters(), lr=CONFIG.learning_rate)

    loss_fn = make_kd_loss_fn(
        teacher, temperature=CONFIG.kd_temperature, alpha=CONFIG.kd_alpha
    )

    run_training(
        model=student,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=CONFIG.kd_epochs,
        device=device,
    )

    save_checkpoint(student, kd_checkpoint(args.deep))


if __name__ == "__main__":
    main()
