from abc import ABC, abstractmethod
from pathlib import Path
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['PyTorchSchedulerRepository']

class PyTorchSchedulerRepository(ABC):
    def __init__(self):
        super().__init__()

    def get_file_name(self, name) -> str:
        return f'{name}.pth'

    def get_file_path(self, name) -> Path:
        return Path(self.run.dir) / self.get_file_name(name)

    @abstractmethod
    async def get(self, name: str) -> _LRScheduler: ...

    @abstractmethod
    async def save(self, scheduler: _LRScheduler, name: str): ...