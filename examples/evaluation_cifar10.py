from dataloader.cifar10 import Cifar10DataLoader
from models.selmiss import CIFAR10Model, BnnCIFAR10Model
from evaluation.evaluation import classification_val
import torch

data_loader = Cifar10DataLoader().build_loader("../datasets/cifar10")
model_ori = CIFAR10Model()
model_bnn = BnnCIFAR10Model()

ckp_path = "../checkpoints/cifar10_trainer/[10] loss: 153.739"
# torch.save(model_ori.state_dict(), ckp_path)
# exit(0)
ckp_path_ori = "../checkpoints/demo1.pth"
classification_val(data_loader, model_ori, ckp_path, torch.device("cuda"))