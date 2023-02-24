from pathlib import Path
from typing import Any, Callable, Dict
from wandb.wandb_run import Run
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
        self.files_dir: Path = Path(self.run.settings.files_dir)

    def get_file_name(self, name) -> str:
        return f'{name}.pth'

    def get_file_path(self, name) -> Path:
        return Path(self.run.dir) / self.get_file_name(name)

    async def get(self, name: str) -> _LRScheduler:
        try:
            file_path: Path = self.files_dir / self.get_file_name(name)

            state_dict: Dict[str, Any] = torch.load(str(file_path))

            scheduler = self.scheduler_factory()

            scheduler.load_state_dict(state_dict)

            return scheduler
        except:
            return None

    async def save(self, scheduler: _LRScheduler, name: str):
        file_path: Path = self.get_file_path(name)

        torch.save(scheduler.state_dict(), str(file_path))

        self.run.save(str(file_path))