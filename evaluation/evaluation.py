import torch
from tqdm import tqdm
from tabulate import tabulate
from .monitors import start_measure, end_measure, format_size, compute_complexity
import os


def classification_val(data_loader, model, checkpoint, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint), strict=True)
    model.eval()

    # Compute the flops and params
    first_data = next(iter(data_loader))
    input_tmp, _ = first_data
    flops, total_params = compute_complexity(model, input_tmp.to(device))

    # Inference to get accuracy
    correct = 0
    total = 0
    start_measures = start_measure(cpu_peak=True)
    with torch.no_grad():  # 禁用梯度计算
        for items, labels in tqdm(data_loader):
            items, labels = items.to(device), labels.to(device)
            outputs = model(items)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    result = dict()
    accuracy = 100 * correct / total
    end_measures = end_measure(start_measures, cpu_peak=True)

    # Format the evaluation results
    result["accuracy"] = str(accuracy) + "%"
    result["params"] = total_params
    result["flops"] = flops
    result["model_size"] = format_size(os.path.getsize(checkpoint))
    result.update(end_measures)
    header_row = list(result.keys())
    value_row = list(result.values())
    table = tabulate([header_row, value_row], tablefmt="grid")
    print(table)
    return result

