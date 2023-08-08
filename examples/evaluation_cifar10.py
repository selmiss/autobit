from dataloader.cifar10 import Cifar10DataLoader
from models.selmiss import CIFAR10Model, BnnCIFAR10Model
from evaluation.evaluation import classification_val
import torch

data_loader = Cifar10DataLoader().build_loader("../datasets/cifar10")
model_ori = CIFAR10Model()
model_bnn = BnnCIFAR10Model()

ckp_path = "../checkpoints/cifar10_trainer/"

classification_val(data_loader, model_ori, ckp_path, torch.device("cuda"))