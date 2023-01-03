from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generic
from machine_learning import TModel, TInput, TTarget
from machine_learning.repositories import ModelRepository
from wandb.wandb_run import Run
from ..modeling.pytorch_model import PyTorchModel
import torch

__all__ = ['PyTorchModelWandBRepository']

class PyTorchModelWandBRepository(ModelRepository[PyTorchModel[TInput, TTarget]], ABC):
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
    async def get(self, model: PyTorchModel[TInput, TTarget], name: str) -> PyTorchModel[TInput, TTarget]:
        try:
            weight_file = self.run.restore(self.get_file_name(name))

            model.inner_module.load_state_dict(torch.load(weight_file))

            return model
        except:
            return None

    @abstractmethod
    async def save(self, model: PyTorchModel[TInput, TTarget], name: str):
        file_path: Path = self.get_file_path(name)

        torch.save(model.inner_module.state_dict(), str(file_path))

        self.run.save(str(file_path))