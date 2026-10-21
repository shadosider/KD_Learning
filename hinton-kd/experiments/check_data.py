
import torch

from src.data import get_mnist_loaders


def main():
    train_loader, test_loader = get_mnist_loaders()

    print("训练集样本数:", len(train_loader.dataset))
    print("测试集样本数:", len(test_loader.dataset))
    print("训练批次数:", len(train_loader))
    print("测试批次数:", len(test_loader))

    images, labels = next(iter(train_loader))

    print("图像张量形状:", images.shape)
    print("标签张量形状:", labels.shape)
    print("图像数据类型:", images.dtype)
    print("标签示例:", labels[:10].tolist())
    print("图像取值范围:", images.min().item(), images.max().item())

    if torch.cuda.is_available():
        images = images.to("cuda")
        labels = labels.to("cuda")
        print("GPU 图像张量设备:", images.device)
        print("GPU 标签张量设备:", labels.device)


if __name__ == "__main__":
    main()
