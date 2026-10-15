# MNIST 手写数字识别：读取数据 -> 训练线性分类器 -> 测试 -> 保存参数。
# 学习时重点关注：张量形状、参数与梯度、训练和评估各自的职责。

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch

# 固定 PyTorch 随机数起点，方便在同一环境中重复初始化和打乱数据。
torch.manual_seed(42)

# MNIST 灰度图：PIL 图片 -> float32 张量；像素 0~255 缩放到 0~1。
# 单张图片形状为 [通道, 高, 宽]，即 [1, 28, 28]。
to_tensor = transforms.ToTensor()

# 保留第 0 维的批次，把每张图展开：[B, 1, 28, 28] -> [B, 784]。
flatten = nn.Flatten(start_dim=1)

# 每张图的 784 个像素 -> 数字 0~9 的 10 个原始分数（不是概率）。
# 可训练参数：weight 为 [10, 784]，bias 为 [10]。
linear = nn.Linear(in_features=28**2, out_features=10)

# 接收原始分数和整数标签，无需先做 Softmax；默认返回当前批次平均损失。
loss_fn = nn.CrossEntropyLoss()

# SGD 更新规则：参数 -= 学习率 * 梯度；这里管理 linear 的权重和偏置。
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# train=True 读取官方的 60,000 张训练图；ToTensor 在取样本时自动执行。
# ./data 相对于运行程序时的工作目录；download=False 使用已下载的数据。
train_data = datasets.MNIST(
    root="./data", train=True, download=False, transform=to_tensor
)

# DataLoader 将样本组成批次；shuffle 打乱顺序时仍保持图片和标签对应。
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

training_step = 0

print("----------Training starts----------\n")

# 完整遍历训练集一次称为 1 个 epoch；60,000 张图按每批 64 张共 938 批。
for images, labels in train_loader:
    # backward 默认累加梯度，因此每批开始先清空上批的梯度记录。
    optimizer.zero_grad()
    # images 为 [B, 1, 28, 28]，labels 为 [B]；最后一批的 B 可能不足 64。
    flat_images = flatten(images)
    scores = linear(flat_images)
    # dim=1 沿类别维找最大分数的索引，得到每张图的预测数字。
    predictions = scores.argmax(dim=1)
    # 损失使用保留全部类别分数的 scores，而不是 argmax 后的类别编号。
    loss = loss_fn(scores, labels)
    # 反向传播：计算损失对各参数的梯度，存入参数的 .grad。
    loss.backward()
    # 根据已计算的梯度更新参数；step 本身不会清空梯度。
    optimizer.step()

    training_step += 1
    # 每 100 批打印一次；item() 取出普通数值，loss 是这次更新前算出的损失。
    if training_step % 100 == 0:
        print("batch:", training_step, "loss:", loss.item())

print("Training completed with " + str(training_step) + " steps.")

# train=False 读取独立的 10,000 张测试图，用于检查未参与训练的样本。
test_data = datasets.MNIST(
    root="./data", train=False, download=False, transform=to_tensor
)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

# 切换评估模式；它与关闭梯度记录是两件事，当前线性层计算行为不变。
linear.eval()
correct_ans = 0
total = 0
print("\n----------Testing starts----------\n")
# 评估只做前向计算，不记录新的自动求导关系；循环中不进行参数更新。
with torch.no_grad():
    for images, labels in test_loader:
        flat_images = flatten(images)
        scores = linear(flat_images)
        predictions = scores.argmax(dim=1)
        # 逐张比较：True 按 1、False 按 0 求和，累计预测正确的图片数量。
        correct_ans += (predictions == labels).sum().item()
        # 累计实际批大小，正确计入最后不足 64 张的一批。
        total += labels.shape[0]

print("Cases tested: " + str(total))
print("Cases passed: " + str(correct_ans))
print("Accuracy: " + str(correct_ans / total) + "\n")

# 保存此线性层的权重和偏置；恢复时先创建相同结构，再 load_state_dict。
state = linear.state_dict()
torch.save(state, "./mnist_linear_training.pt")
print("Parameters saved to ./mnist_linear_training.pt")
