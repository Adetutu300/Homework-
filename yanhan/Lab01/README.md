# MNIST 三层线性神经网络

本作业包括 [W3 的反向传播公式推导](推导反向传播公式.md) 和 MNIST 三层线性网络训练。

## 做法

1. 加载 MNIST：训练集 60,000 张，测试集 10,000 张。像素先缩放到 0～1，再用均值 0.1307、标准差 0.3081 标准化。
2. 将每张 28×28 图片展平成 784 个数，输入 `784 → 256 → 128 → 10` 的三个 `Linear` 层，不使用激活函数。三个线性层组合后仍是线性模型。
3. 使用交叉熵损失和 SGD 优化器，学习率为 0.001，动量为 0.9。批大小为 256，在 CPU 上训练 20 轮，随机种子为 42。
4. 每批数据依次执行清除梯度、前向传播、计算损失、反向传播和更新参数。
5. 保存并重新加载模型权重，在测试集上统计预测正确的图片数量，计算准确率。

## 最终结果

测试准确率为 **91.98%（9,198 / 10,000）**。

模型权重保存在 `outputs/tutorial/mnist_three_linear.pt`，实验配置和准确率保存在 `outputs/tutorial/metrics.json`。

## 运行方法

使用 Python 3.12，在 `yanhan/Lab01` 目录打开 PowerShell，执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_mnist.py
```

首次运行会自动下载 MNIST 到 `data` 文件夹。程序在 CPU 上训练，完成后输出测试准确率并保存模型权重和结果。训练参数可直接在 `train_mnist.py` 中修改。
