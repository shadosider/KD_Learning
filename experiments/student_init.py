from kdlib.models import create_student
from kdlib.training import save_checkpoint, set_seed


def main():
    # 固定随机种子,保证所有学生模型从同一初始权重出发
    set_seed(42)

    student = create_student()

    path = save_checkpoint(student, "student_init.pt")
    print(f"Initial student weights saved to: {path}")
    print(f"Parameter count: {sum(p.numel() for p in student.parameters())}")


if __name__ == "__main__":
    main()
