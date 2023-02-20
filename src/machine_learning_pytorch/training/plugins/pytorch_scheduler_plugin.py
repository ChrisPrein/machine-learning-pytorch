import asyncio
from logging import Logger
from machine_learning.training import PreLoop, PostEpoch, TrainingContext, TTrainer
from machine_learning import TInput, TTarget, TModel, TOutput
from torch.optim.lr_scheduler import _LRScheduler
from .repositories.pytorch_scheduler_repository import PyTorchSchedulerRepository

__all__ = ['PyTorchSchedulerPlugin']

LATEST_SCHEDULER_NAME: str = "latest-scheduler"

class PyTorchSchedulerPlugin(PreLoop[TInput, TTarget, TOutput, TModel, TTrainer], PostEpoch[TInput, TTarget, TOutput, TModel, TTrainer]):
    def __init__(self, scheduler: _LRScheduler, repository: PyTorchSchedulerRepository, event_loop: asyncio.AbstractEventLoop = None):
        super().__init__()

        if scheduler is None:
            raise TypeError('scheduler')

        if repository is None:
            raise TypeError('repository')

        self.scheduler: _LRScheduler = scheduler
        self.repository: PyTorchSchedulerRepository = repository
        self.event_loop: asyncio.AbstractEventLoop = event_loop if event_loop != None else asyncio.get_event_loop()

    def pre_loop(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        logger.info('Loading scheduler checkpoint...')
        loaded_scheduler = self.event_loop.run_until_complete(self.repository.get(LATEST_SCHEDULER_NAME))

        if loaded_scheduler != None:
            self.scheduler = loaded_scheduler
            logger.info('Scheduler checkpoint loaded!')
        else:
            logger.info('No scheduler checkpoint found!')

    def post_epoch(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        logger.info('Performing scheduler step...')
        self.scheduler.step()
        logger.info('Scheduler step performed!')

        logger.info('Creating scheduler checkpoint...')
        self.event_loop.create_task(self.repository.save(self.scheduler, LATEST_SCHEDULER_NAME))
        logger.info('Scheduler checkpoint created!')