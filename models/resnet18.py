import torch
import torch.nn as nn
import torchvision.models as models
from quantization.layers.bnn import BNNConv2d, BNNConv1d, BNNLinear
class ResNet18Cifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Cifar10, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.resnet18.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.resnet18(x)


class BnnResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(BnnResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        
        self.resnet18.layer1[0].conv1 = BNNConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.layer1[0].conv2 = BNNConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.layer1[1].conv1 = BNNConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.layer1[1].conv2 = BNNConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.resnet18.layer2[0].conv1 = BNNConv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.resnet18.layer2[0].conv2 = BNNConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.layer2[1].conv1 = BNNConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.layer2[1].conv2 = BNNConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.resnet18.layer3[0].conv1 = BNNConv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.resnet18.layer3[0].conv2 = BNNConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.layer3[1].conv1 = BNNConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.layer3[1].conv2 = BNNConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.resnet18.layer4[0].conv1 = BNNConv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.resnet18.layer4[0].conv2 = BNNConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.layer4[1].conv1 = BNNConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.layer4[1].conv2 = BNNConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.resnet18.fc = BNNLinear(512, num_classes)
    def forward(self, x):
        return self.resnet18(x)
