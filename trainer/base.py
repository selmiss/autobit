from abc import ABC, abstractmethod
import torch.nn as nn
from dataloader.base import DataLoader


class Trainer(ABC):

    model: nn.Module = None
    data_loader: DataLoader = None
    ckp_load_path: str = ""
    ckp_save_path: str = ""

    @abstractmethod
    def train(self):
        pass
