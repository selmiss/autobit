from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Optional, Sequence


class DataStructure(ABC):
    pass


class DataLoader(ABC):
    """Dataloader interface."""
    @abstractmethod
    def build_loader(self, data_path: str, **kwargs: Any):
        """Build dataloader for evaluation."""
        pass

    @abstractmethod
    def download_data(self, target_path: str, **kwargs: Any):
        """Load data for local path."""
        pass
