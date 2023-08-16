from typing import Any, Tuple
from dataloader.base import DataLoader, DataStructure
import torch
import torchvision
import torchvision.transforms as transforms


class CifarDataStructure(DataStructure):
    pass


class Cifar10DataLoader(DataLoader):

    def __init__(self, batch_size: int = 32, train: bool = False, shuffle: bool = False):
        # Define preprocess approach
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.batch_size = batch_size
        self.train = train
        self.shuffle = shuffle

    def build_loader(self, data_path: str, **kwargs: Any):
        """Load data for local path."""
        # Load cifar10 dataset.
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=self.train, download=True, transform=self.transform)
        # Create dataloader to load data.
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return data_loader

    def download_data(self, target_path: str, **kwargs: Any):
        # Download dataset to specific path
        torchvision.datasets.CIFAR10(root=target_path, train=self.train, download=True, transform=self.transform)

