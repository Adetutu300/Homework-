import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2


# 定义三层线性神经网络
class ThreeLayerLinear(nn.Module):
    def __init__(self):
        super().__init__()
        # 每一层的输出会作为下一层的输入，不添加激活函数
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        # 每张图片是 1×28×28，将它展平成 784 个数，保留批次维度
        x = torch.flatten(x, 1)
        x = self.layers(x)
        return x


if __name__ == "__main__":
    # 设置训练参数，想调整实验时可以直接修改这里
    epochs = 20
    batch_size = 256
    learning_rate = 0.001
    torch.manual_seed(42)  # 固定随机种子，方便重复实验
    torch.set_num_threads(4)  # 使用 4 个 CPU 线程
    torch.use_deterministic_algorithms(True)  # 相同环境下使用可复现的计算方式

    # 加载 MNIST 数据集
    transform = v2.Compose([
        v2.ToImage(),  # 将图片转换成张量
        v2.ToDtype(torch.float32, scale=True),  # 将像素从 0～255 缩放到 0～1
        v2.Normalize((0.1307,), (0.3081,)),  # 减去均值，再除以标准差
    ])
    train_data = MNIST("data", train=True, download=True, transform=transform)
    test_data = MNIST("data", train=False, download=True, transform=transform)

    # 训练时打乱图片顺序；测试时不需要打乱
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False,
                             num_workers=0)

    # 创建模型、损失函数和优化器
    model = ThreeLayerLinear()
    # 交叉熵直接接收模型输出的 10 个分数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 训练模型
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)
            loss.backward()  # 反向传播
            optimizer.step()

            # 按图片数量累计损失
            total_loss += loss.item() * labels.size(0)

        average_loss = total_loss / len(train_data)
        print(f"第 {epoch + 1}/{epochs} 轮，平均损失：{average_loss:.4f}", flush=True)

    # 保存权重，再加载到一个新模型中
    output_dir = Path("outputs/tutorial")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "mnist_three_linear.pt"
    torch.save(model.state_dict(), model_path)

    model = ThreeLayerLinear()
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"测试准确率：{accuracy:.2%}（{correct}/{total}）", flush=True)

    # 保存实验的配置和结果
    result = {
        "architecture": [784, 256, 128, 10],
        "activations": None,
        "dataset": "MNIST",
        "train_samples": len(train_data),
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "SGD",
        "learning_rate": learning_rate,
        "momentum": 0.9,
        "seed": 42,
        "test_samples": total,
        "test_correct": correct,
        "test_accuracy": accuracy,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
