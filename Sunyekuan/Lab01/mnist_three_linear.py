"""使用三个全连接线性层完成 MNIST 手写数字分类。"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class ThreeLinearNet(nn.Module):
    """三层线性层网络；两个隐藏层之间使用 ReLU 引入非线性。"""

    def __init__(self):
        super().__init__()
        # 28×28 灰度图展平后有 784 个像素，最终输出 10 个类别的原始分数。
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """将一批图像映射为每个数字类别的 logits。"""
        return self.layers(images)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[int, int, np.float32]:
    """统计验证集或测试集的正确数、总样本数和平均交叉熵；不更新模型参数。"""
    model.eval()
    correct = 0  # 预测类别与真实标签相同的样本数。
    total = 0  # 已评估的样本总数。
    loss_sum = np.float32(0)  # 所有样本的交叉熵之和，使用单精度累加。
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for images, labels in loader:
            # images 为 float32 图像张量；labels 为整数类别标签。
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
            loss_sum += np.float32(criterion(logits, labels).item())
    return correct, total, loss_sum / np.float32(total)


def main() -> None:
    """训练、逐轮保存权重，按验证集选择最佳权重，最后评估测试集。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mnist_15epochs"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs 和 batch-size 必须为正整数，num-workers 不能为负数")
    # 一个目录对应一次完整实验，避免覆盖之前保存的逐轮权重和结果。
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir 已有文件，请为新实验指定一个空目录")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 固定数据打乱及权重初始化的随机种子，便于复现实验。
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ToTensor 将像素缩放到 [0, 1]，并生成 torch.float32 张量。
    transform = transforms.ToTensor()
    full_train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    # 从原始训练集固定划出 5,000 张验证图像，用于比较每轮权重。
    # 55,000 张训练图像参与反向传播；10,000 张测试图像只用于最终评估。
    train_set, val_set = random_split(
        full_train_set, [55000, 5000], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = ThreeLinearNet().to(device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss()
    # 网络参数和输入均为 float32；优化器在参数张量上进行单精度更新。
    optimizer = torch.optim.Adam(model.parameters(), lr=np.float32("0.001"))
    # 每完成 5 轮，将下一轮学习率乘以 0.5。
    # 默认 15 轮分别使用 0.001、0.0005、0.00025 三个单精度学习率。
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=np.float32("0.5"))
    best_correct = -1  # 当前最佳验证集正确预测数，优先选择正确数最多的权重。
    best_loss = np.float32("inf")  # 验证正确数相同时，选择交叉熵更低的权重。
    best_epoch = 0  # 最佳权重对应的训练轮次，从 1 开始计数。
    history = []  # 各轮的学习率、训练损失及验证结果。
    print(f"device={device}, train={len(train_set)}, validation={len(val_set)}, test={len(test_set)}", flush=True)
    # 保存划分索引，便于核对训练集、验证集及复现实验。
    torch.save({"train": train_set.indices, "validation": val_set.indices},
               args.output_dir / "split_indices.pt")

    for epoch in range(args.epochs):
        learning_rate = np.float32(optimizer.param_groups[0]["lr"])
        model.train()
        loss_sum = np.float32(0)  # 当前轮次各批次损失的加权和。
        sample_count = 0  # 当前轮次参与训练的样本数。
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_count = labels.numel()
            loss_sum += np.float32(loss.item()) * np.float32(batch_count)
            sample_count += batch_count
        average_loss = loss_sum / np.float32(sample_count)
        val_correct, val_total, val_loss = evaluate(model, val_loader, device)
        val_accuracy = np.float32(val_correct) / np.float32(val_total)
        # 保存本轮完整模型参数；统一放到 CPU，方便在无 GPU 的环境加载。
        epoch_path = args.output_dir / f"epoch_{epoch + 1:02d}.pt"
        torch.save({name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()}, epoch_path)
        is_best = val_correct > best_correct or (val_correct == best_correct and val_loss < best_loss)
        if is_best:
            best_correct, best_loss, best_epoch = val_correct, val_loss, epoch + 1
            shutil.copyfile(epoch_path, args.output_dir / "best_weights.pt")
        # 小数以字符串落盘，计算过程保持 np.float32，不引入双精度统计计算。
        history.append({"epoch": epoch + 1, "learning_rate": str(learning_rate),
                        "train_loss": str(average_loss), "validation_loss": str(val_loss),
                        "validation_correct": val_correct, "validation_total": val_total,
                        "validation_accuracy": str(val_accuracy), "selected_at_epoch": is_best})
        with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch={epoch + 1}, lr={learning_rate:.6f}, train_loss={average_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4%}, best={is_best}", flush=True)
        scheduler.step()

    # 加载根据验证集选出的最佳权重，再进行一次最终测试集评估。
    model.load_state_dict(torch.load(args.output_dir / "best_weights.pt", map_location=device, weights_only=True))
    correct, total, test_loss = evaluate(model, test_loader, device)
    accuracy = np.float32(correct) / np.float32(total)
    summary = {"seed": 42, "epochs": args.epochs, "batch_size": args.batch_size,
               "architecture": [784, 256, 128, 10], "torch_version": torch.__version__,
               "device": str(device), "train_total": len(train_set), "validation_total": len(val_set),
               "selection": "validation_correct; validation_loss breaks ties",
               "best_epoch": best_epoch, "best_validation_correct": best_correct,
               "best_validation_loss": str(best_loss), "test_correct": correct,
               "test_total": total, "test_accuracy": str(accuracy), "test_loss": str(test_loss)}
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"best_epoch={best_epoch}, test_correct={correct}, test_total={total}, "
          f"test_accuracy={accuracy:.4%}", flush=True)


if __name__ == "__main__":
    main()
