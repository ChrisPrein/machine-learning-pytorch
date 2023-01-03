from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic
from machine_learning import TModel, TInput, TTarget
from machine_learning.repositories import TrainerRepository
from wandb.wandb_run import Run
from ..training.pytorch_trainer import PyTorchTrainer, TPyTorchModel
import torch

__all__ = ['PyTorchTrainerWandBRepository']

class PyTorchTrainerWandBRepository(TrainerRepository[PyTorchTrainer[TInput, TTarget, TPyTorchModel]]):
    def __init__(self, run: Run):
        super().__init__()

        if run is None:
            raise TypeError('run')

        self.run: Run = run

    def get_file_name(self, name) -> str:
        return f'{name}.pth'

    def get_file_path(self, name) -> Path:
        return Path(self.run.dir) / self.get_file_name(name)

    @abstractmethod
    async def get(self, trainer: PyTorchTrainer[TInput, TTarget, TPyTorchModel], name: str) -> PyTorchTrainer[TInput, TTarget, TPyTorchModel]:
        try:
            weight_file = self.run.restore(self.get_file_name(name))

            state_dict: Dict[str, Any] = torch.load(weight_file)

            trainer.optimizer.load_state_dict(state_dict)

            return trainer
        except:
            return None

    @abstractmethod
    async def save(self, trainer: PyTorchTrainer[TInput, TTarget, TPyTorchModel], name: str):
        file_path: Path = self.get_file_path(name)

        torch.save(trainer.optimizer.state_dict(), str(file_path))

        self.run.save(str(file_path))