from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Generic
from machine_learning import TModel, TInput, TTarget
from machine_learning.training.plugins.repositories.model_repository import ModelRepository
from wandb.wandb_run import Run
from ....modeling.pytorch_model import PyTorchModel
import torch

__all__ = ['PyTorchModelWandBRepository']

class PyTorchModelWandBRepository(ModelRepository[PyTorchModel[TInput, TTarget]]):
    def __init__(self, run: Run, model_factory: Callable[[], PyTorchModel[TInput, TTarget]]):
        super().__init__()

        if run is None:
            raise TypeError('run')

        if model_factory is None:
            raise TypeError('model_factory')

        self.run: Run = run
        self.model_factory: Callable[[], PyTorchModel[TInput, TTarget]] = model_factory

    def get_file_name(self, name) -> str:
        return f'{name}.pth'

    def get_file_path(self, name) -> Path:
        return Path(self.run.dir) / self.get_file_name(name)

    async def get(self, name: str) -> PyTorchModel[TInput, TTarget]:
        try:
            weight_file = self.run.restore(self.get_file_name(name))

            model = self.model_factory()

            model.inner_module.load_state_dict(torch.load(weight_file))

            return model
        except:
            return None

    async def save(self, model: PyTorchModel[TInput, TTarget], name: str):
        file_path: Path = self.get_file_path(name)

        torch.save(model.inner_module.state_dict(), str(file_path))

        self.run.save(str(file_path))