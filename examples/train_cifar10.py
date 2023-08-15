from models.selmiss import CIFAR10Model, BnnCIFAR10Model
from dataloader.cifar10 import Cifar10DataLoader
from trainer.classification_trainer import ClassificationTrainer
from models.resnet18 import ResNet18Cifar10, BnnResNet18

model = CIFAR10Model()
model_bnn = BnnCIFAR10Model()
model_res18 = ResNet18Cifar10()
model_bnn18 = BnnResNet18()
dataloader = Cifar10DataLoader(train=True).build_loader("../datasets/cifar10")
load_path = "../checkpoints/cifar10_resnet18/75 loss: 0.000.pth"
save_path = "../checkpoints/cifar10_bnn18"
trainer = ClassificationTrainer(model_bnn18, dataloader, save_path, ckp_load_path=load_path)
trainer.epoch = 100
trainer.train()
