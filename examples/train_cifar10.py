from models.selmiss import CIFAR10Model, BnnCIFAR10Model
from dataloader.cifar10 import Cifar10DataLoader
from trainer.classification_trainer import ClassificationTrainer

model = CIFAR10Model()
dataloader = Cifar10DataLoader(train=True).build_loader("../datasets/cifar10")
load_path = "../checkpoints/cifar10_trainer/50 loss: 0.301.pth"
save_path = "../checkpoints/cifar10_quantization"
trainer = ClassificationTrainer(model, dataloader, save_path, ckp_load_path=load_path)
trainer.train()
