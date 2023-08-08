from dataloader.cifar10 import Cifar10DataLoader
from models.selmiss import CIFAR10Model, BnnCIFAR10Model
from evaluation.evaluation import classification_val
import torch
from evaluation.tools import log_parameters

data_loader = Cifar10DataLoader().build_loader("../datasets/cifar10")
model_ori = CIFAR10Model()
model_bnn = BnnCIFAR10Model()

ckp_path = "../checkpoints/cifar10_trainer/50 loss: 0.301.pth"
ckp_path_quantize = "../checkpoints/cifar10_quantization/20 loss: 0.039.pth"
log_parameters(model_ori, ckp_path)

classification_val(data_loader, model_ori, ckp_path, torch.device("cuda"))