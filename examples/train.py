from models.selmiss import CIFAR10Model, BnnCIFAR10Model
from dataloader.cifar10 import Cifar10DataLoader
from trainer.classification_trainer import ClassificationTrainer

model = CIFAR10Model()
dataloader = Cifar10DataLoader(train=True).build_loader("../datasets/cifar10")
save_path = "../checkpoints/cifar10_trainer"
trainer = ClassificationTrainer(model, dataloader, save_path)
trainer.train()
