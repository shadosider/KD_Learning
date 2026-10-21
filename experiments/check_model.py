
import torch

from kdlib.data import get_mnist_loaders
from kdlib.models import create_teacher, create_student


def main():
    train_loader, _ = get_mnist_loaders()
    images, labels = next(iter(train_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = create_teacher().to(device)
    student = create_student().to(device)
    images = images.to(device)

    with torch.no_grad():
        teacher_logits = teacher(images)
        student_logits = student(images)

    print("Device:", device)
    print("Input shape:", images.shape)
    print("Teacher output:", teacher_logits.shape)
    print("Student output:", student_logits.shape)
    print("Teacher parameters:", sum(p.numel() for p in teacher.parameters()))
    print("Student parameters:", sum(p.numel() for p in student.parameters()))
    print("Labels:", labels[:10].tolist())


if __name__ == "__main__":
    main()
