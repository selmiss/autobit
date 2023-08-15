from dataloader.cifar10 import Cifar10DataLoader
from models.selmiss import CIFAR10Model, BnnCIFAR10Model
from models.resnet18 import ResNet18Cifar10, BnnResNet18
from models.birealnet import birealnet18
from evaluation.evaluation import classification_val
import torch
from evaluation.tools import log_parameters

data_loader = Cifar10DataLoader().build_loader("../datasets/cifar10")
model_ori = CIFAR10Model()
model_bnn = BnnCIFAR10Model()
model_res18 = ResNet18Cifar10()
model_bnn18 = BnnResNet18()

# print(model_bnn18)
res18_path = "../checkpoints/cifar10_bnn18/96 loss: 15.159.pth"

ckp_path = "../checkpoints/cifar10_trainer/50 loss: 0.301.pth"
ckp_path_quantize = "../checkpoints/cifar10_quantization/50 loss: 0.001.pth"
ckp_path_bnn = "../checkpoints/cifar10_bnn/30 loss: 27.256.pth"
# log_parameters(model_ori, ckp_path)

classification_val(data_loader, model_bnn18, res18_path, torch.device("cuda"))
