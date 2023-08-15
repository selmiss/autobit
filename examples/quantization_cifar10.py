from dataloader.cifar10 import Cifar10DataLoader
from models.selmiss import CIFAR10Model, BnnCIFAR10Model
from evaluation.evaluation import classification_val
import torch

model = CIFAR10Model()
data_loader = Cifar10DataLoader().build_loader("../datasets/cifar10")
ckp_path = "../checkpoints/cifar10_trainer/50 loss: 0.301.pth"
device = torch.device("cpu")
model.to(device)
model.qconfig = torch.quantization.default_qconfig
model = torch.quantization.prepare(model)
model = torch.quantization.convert(model)
torch.save(model.state_dict(), "../checkpoints/a.pth")
classification_val(data_loader, model, "../checkpoints/a.pth", torch.device("cpu"))