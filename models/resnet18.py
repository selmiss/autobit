import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Cifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Cifar10, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.resnet18.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.resnet18(x)

