import torch
from tqdm import tqdm
from models.selmiss import CIFAR10Model, BnnCIFAR10Model
from dataloader.cifar10 import Cifar10DataLoader
from tabulate import tabulate
from tools import start_measure, end_measure, log_measures
import os


def classification_val(data_loader, model, checkpoint, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # 设置模型为评估模式
    model.load_state_dict(torch.load(checkpoint))

    # Compute the flops and params
    first_data = next(iter(data_loader))
    input_tmp, _ = first_data
    flops, total_params = cul_complexity(model, input_tmp)
    # model_summary(model, input_tmp)

    # Inference to get accuracy
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
    end_measures = end_measure(start_measures, cpu_peak=False)

    result["accuracy"] = str(accuracy) + "%"
    result["total_params"] = total_params
    result["flops"] = flops
    result["model_size"] = format_size(os.path.getsize(checkpoint))
    result.update(end_measures)
    header_row = list(result.keys())
    value_row = list(result.values())
    table = tabulate([header_row, value_row], tablefmt="grid")
    print(table)
    return result


def cul_complexity(model, input, device="cuda"):
    from thop import profile
    from thop import clever_format
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params


def model_summary(model, input, device="cpu"):
    from torchsummary import summary
    summary(model.to(device), input_size=(1, 32, 32), batch_size=-1)


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


if __name__ == "__main__":
    data_loader = Cifar10DataLoader().build_loader("../datasets/cifar10")
    model_ori = CIFAR10Model()
    model_bnn = BnnCIFAR10Model()
    ckp_path = "../checkpoints/demo1_bnn.pth"
    ckp_path_ori = "../checkpoints/demo1.pth"
    # for name, param in model_ori.named_parameters():
    #     if param.requires_grad:  # 只打印需要梯度更新的参数
    #         print(f"Parameter Name: {name}")
    #         print(f"Parameter Shape: {param.shape}")
    #         print(f"Parameter Data Type: {param.dtype}")
    #         print(f"Parameter Values:")
    #         print(param)
    #         print("-------------------------------")
    # exit(0)
    classification_val(data_loader, model_ori, ckp_path_ori)

