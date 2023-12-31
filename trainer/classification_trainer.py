import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.base import DataLoader
from tqdm import tqdm
from .base import Trainer


class ClassificationTrainer(Trainer):
    model: nn.Module = None
    data_loader: DataLoader = None
    ckp_load_path: str = ""
    ckp_save_path: str = ""
    ckp_tmp_path: str = ""
    epoch: int = 50
    lr: float = 0.001
    momentum: float = 0.9
    log_period = 150
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self, 
        model, 
        data_loader, 
        ckp_save_path: str = "../checkpoints/tmp_ckp",
        epoch: int = 5,
        ckp_load_path: str = None,
        device: torch.device = None
    ):
        self.model = model
        self.data_loader = data_loader
        self.ckp_load_path = ckp_load_path
        self.ckp_save_path = ckp_save_path
        self.epoch = epoch
        if device is not None:
            self.device = device
        self.model.to(self.device)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        cnt = 0

        if self.ckp_load_path is not None:
            self.model.load_state_dict(torch.load(self.ckp_load_path))
        if self.ckp_save_path is not None:
            if not os.path.exists(self.ckp_save_path):
                os.mkdir(self.ckp_save_path)
        for epoch in range(self.epoch):
            cnt += 1
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            for data in tqdm(self.data_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # 
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
            accuracy = correct_predictions / total_samples * 100
            log_message = f'{epoch + 1}-{running_loss / self.log_period:.6f}-{accuracy:.3f}%'
            self.ckp_tmp_path = os.path.join(self.ckp_save_path, log_message + ".pth")
            torch.save(self.model.state_dict(), self.ckp_tmp_path)
            print(log_message)
            running_loss = 0.0
        return self.ckp_tmp_path

