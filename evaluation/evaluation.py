import torch
from tqdm import tqdm
from models.selmiss import CIFAR10Model
from dataloader.cifar10 import Cifar10DataLoader
from tabulate import tabulate
from tools import start_measure, end_measure, log_measures


def classification_val(data_loader, model, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    start_measures = start_measure(cpu_peak=False)
    with torch.no_grad():  # 禁用梯度计算
        for items, labels in tqdm(data_loader):
            items, labels = items.to(device), labels.to(device)
            outputs = model(items)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    result = dict()
    accuracy = 100 * correct / total
    total_params = sum(p.numel() for p in model.parameters())
    end_measures = end_measure(start_measures, cpu_peak=False)
    result["accuracy"] = accuracy
    result["total_params"] = total_params
    result.update(end_measures)
    header_row = list(result.keys())
    value_row = list(result.values())
    table = tabulate([header_row, value_row], tablefmt="grid")
    print(table)
    return result


if __name__ == "__main__":
    data_loader = Cifar10DataLoader().build_loader("../datasets/cifar10")
    model = CIFAR10Model()
    classification_val(data_loader, model)
