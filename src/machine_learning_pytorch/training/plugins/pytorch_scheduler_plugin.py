import asyncio
from logging import Logger
from machine_learning.training import PreLoop, PostEpoch, TrainingContext, TTrainer
from machine_learning import TInput, TTarget, TModel
from torch.optim.lr_scheduler import _LRScheduler
from wandb.wandb_run import Run
from .repositories.pytorch_scheduler_repository import PyTorchSchedulerRepository

__all__ = ['PyTorchSchedulerPlugin']

LATEST_SCHEDULER_NAME: str = "latest-scheduler"

class PyTorchSchedulerPlugin(PreLoop[TInput, TTarget, TModel, TTrainer], PostEpoch[TInput, TTarget, TModel, TTrainer]):
    def __init__(self, scheduler: _LRScheduler, repository: PyTorchSchedulerRepository, event_loop: asyncio.AbstractEventLoop = None):
        super().__init__()

        if scheduler is None:
            raise TypeError('scheduler')

        if repository is None:
            raise TypeError('repository')

        self.scheduler: _LRScheduler = scheduler
        self.repository: PyTorchSchedulerRepository = repository
        self.event_loop: asyncio.AbstractEventLoop = event_loop if event_loop != None else asyncio.get_event_loop()

    def pre_loop(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TModel, TTrainer]):
        self.scheduler = self.event_loop.run_until_complete(self.repository.get(LATEST_SCHEDULER_NAME))

    def post_epoch(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TModel, TTrainer]):
        self.scheduler.step()