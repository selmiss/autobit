import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization.layers.bnn import *


class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 128 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class BnnCIFAR10Model(nn.Module):

    def __init__(self):
        super(BnnCIFAR10Model, self).__init__()
        self.conv1 = BNNConv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = BNNConv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = BNNConv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = BNNLinear(128 * 4 * 4, 512)
        self.fc2 = BNNLinear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 128 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = CIFAR10Model()
    # 打印模型的输入和输出张量格式
    input_example = torch.randn(1, 3, 32, 32)  # 示例输入张量（1个样本，3个通道，32x32像素）
    output_example = model(input_example)
    print("输入张量格式:", input_example.shape)
    print("输出张量格式:", output_example.shape)
    print(output_example)
    # 打印模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("模型的参数量:", total_params)
    print(model)
