from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Generic
from machine_learning import TModel, TInput, TTarget
from machine_learning.repositories import TrainerRepository
from wandb.wandb_run import Run
from ..training.pytorch_trainer import PyTorchTrainer, TPyTorchModel
import torch

__all__ = ['PyTorchTrainerWandBRepository', 'TrainerFactory']

TrainerFactory = Callable[[], PyTorchTrainer[TInput, TTarget, TPyTorchModel]]

class PyTorchTrainerWandBRepository(TrainerRepository[PyTorchTrainer[TInput, TTarget, TPyTorchModel]]):
    def __init__(self, run: Run, trainer_factory: TrainerFactory[TInput, TTarget, TPyTorchModel]):
        super().__init__()

        if run is None:
            raise TypeError('run')

        if trainer_factory is None:
            raise TypeError('trainer_factory')

        self.run: Run = run
        self.trainer_factory: TrainerFactory[TInput, TTarget, TPyTorchModel] = trainer_factory

    def get_file_name(self, name) -> str:
        return f'{name}.pth'

    def get_file_path(self, name) -> Path:
        return Path(self.run.dir) / self.get_file_name(name)

    async def get(self, name: str) -> PyTorchTrainer[TInput, TTarget, TPyTorchModel]:
        try:
            weight_file = self.run.restore(self.get_file_name(name))

            state_dict: Dict[str, Any] = torch.load(weight_file)

            trainer = self.trainer_factory()

            trainer.optimizer.load_state_dict(state_dict)

            return trainer
        except:
            return None

    async def save(self, trainer: PyTorchTrainer[TInput, TTarget, TPyTorchModel], name: str):
        file_path: Path = self.get_file_path(name)

        torch.save(trainer.optimizer.state_dict(), str(file_path))

        self.run.save(str(file_path))