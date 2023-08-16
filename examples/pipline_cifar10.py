import sys
sys.path.append("../")
from models.resnet18 import ResNet18Cifar10
from dataloader.cifar10 import Cifar10DataLoader
from trainer.classification_trainer import ClassificationTrainer
from evaluation.evaluation import classification_val
from quantization.processor.replacer import Replacer, Approach

model = ResNet18Cifar10()
data_loader = Cifar10DataLoader().build_loader("../datasets/cifar10")
tmp_path = "../checkpoints/"

# Train from raw
tmp_path = ClassificationTrainer(model, data_loader, device="cuda").train()
# Evaluation before quantizing
classification_val(data_loader, model, tmp_path, device="cuda")
# Quantizing the model
Replacer(approach=Approach.BNN).layer_replacer(model)
# Train again
tmp_path = ClassificationTrainer(model, data_loader, device="cuda").train()
# Evaluation after quantizing
classification_val(data_loader, model, tmp_path, device="cuda")

