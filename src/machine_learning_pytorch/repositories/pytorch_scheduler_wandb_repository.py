from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Generic
from machine_learning import TModel, TInput, TTarget
from wandb.wandb_run import Run
from ..training.pytorch_trainer import PyTorchTrainer, TPyTorchModel
import torch
from .pytorch_scheduler_repository import PyTorchSchedulerRepository
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['PyTorchTrainerWandBRepository', 'SchedulerFactory']

SchedulerFactory = Callable[[], _LRScheduler]

class PyTorchSchedulerWandBRepository(PyTorchSchedulerRepository):
    def __init__(self, run: Run, scheduler_factory: SchedulerFactory):
        super().__init__()

        if run is None:
            raise TypeError('run')

        if scheduler_factory is None:
            raise TypeError('scheduler_factory')

        self.run: Run = run
        self.scheduler_factory: SchedulerFactory = scheduler_factory

    def get_file_name(self, name) -> str:
        return f'{name}.pth'

    def get_file_path(self, name) -> Path:
        return Path(self.run.dir) / self.get_file_name(name)

    async def get(self, name: str) -> _LRScheduler:
        try:
            weight_file = self.run.restore(self.get_file_name(name))

            state_dict: Dict[str, Any] = torch.load(weight_file)

            scheduler = self.scheduler_factory()

            scheduler.load_state_dict(state_dict)

            return scheduler
        except:
            return None

    async def save(self, scheduler: _LRScheduler, name: str):
        file_path: Path = self.get_file_path(name)

        torch.save(scheduler.state_dict(), str(file_path))

        self.run.save(str(file_path))