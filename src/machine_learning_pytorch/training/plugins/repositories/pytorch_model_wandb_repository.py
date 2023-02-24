from pathlib import Path
from typing import Callable
from machine_learning import TInput, TTarget, TOutput
from machine_learning.training.plugins.repositories.model_repository import ModelRepository
from wandb.wandb_run import Run
from ....modeling.pytorch_model import PyTorchModel, TTrainStepOutput
import torch

__all__ = ['PyTorchModelWandBRepository']

class PyTorchModelWandBRepository(ModelRepository[PyTorchModel[TInput, TTarget, TOutput, TTrainStepOutput]]):
    def __init__(self, run: Run, model_factory: Callable[[], PyTorchModel[TInput, TTarget, TOutput, TTrainStepOutput]]):
        super().__init__()

        if run is None:
            raise TypeError('run')

        if model_factory is None:
            raise TypeError('model_factory')

        self.run: Run = run
        self.model_factory: Callable[[], PyTorchModel[TInput, TTarget, TOutput, TTrainStepOutput]] = model_factory
        self.files_dir: Path = Path(self.run.settings.files_dir)

    def get_file_name(self, name) -> str:
        return f'{name}.pth'

    def get_file_path(self, name) -> Path:
        return Path(self.run.dir) / self.get_file_name(name)

    async def get(self, name: str) -> PyTorchModel[TInput, TTarget, TOutput, TTrainStepOutput]:
        try:
            file_path: Path = self.files_dir / self.get_file_name(name)

            model = self.model_factory()

            model.inner_module.load_state_dict(torch.load(str(file_path)))

            return model
        except:
            return None

    async def save(self, model: PyTorchModel[TInput, TTarget, TOutput, TTrainStepOutput], name: str):
        file_path: Path = self.get_file_path(name)

        torch.save(model.inner_module.state_dict(), str(file_path))

        self.run.save(str(file_path))