import asyncio
from logging import Logger
from typing import Any, Dict, Optional, TypeGuard
from machine_learning.training import PreLoop, TrainingContext, TTrainer, PostValidationPlugin
from machine_learning.evaluation import EvaluationResult
from machine_learning import TInput, TTarget, TModel, TOutput
from torch.optim.lr_scheduler import _LRScheduler
from .repositories.pytorch_scheduler_repository import PyTorchSchedulerRepository

__all__ = ['PyTorchPlateauSchedulerPlugin']

LATEST_SCHEDULER_NAME: str = "latest-scheduler"

def is_nested_dict(val: Dict[str, Any]) -> TypeGuard[Dict[str, Dict[str, float]]]:
    return all(isinstance(value, dict) for value in val.values())

class PyTorchPlateauSchedulerPlugin(PreLoop[TInput, TTarget, TOutput, TModel, TTrainer], PostValidationPlugin[TInput, TTarget, TOutput, TModel, TTrainer]):
    def __init__(self, scheduler: _LRScheduler, repository: PyTorchSchedulerRepository, metric_key: str, dataset_name: Optional[str] = None, event_loop: asyncio.AbstractEventLoop = None):
        super().__init__()

        if scheduler is None:
            raise TypeError('scheduler')

        if repository is None:
            raise TypeError('repository')

        self.scheduler: _LRScheduler = scheduler
        self.repository: PyTorchSchedulerRepository = repository
        self.metric_key: str = metric_key
        self.dataset_name: Optional[str] = dataset_name
        self.event_loop: asyncio.AbstractEventLoop = event_loop if event_loop != None else asyncio.get_event_loop()

    def pre_loop(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer]):
        logger.info('Loading scheduler checkpoint...')
        loaded_scheduler = self.event_loop.run_until_complete(self.repository.get(LATEST_SCHEDULER_NAME))

        if loaded_scheduler != None:
            self.scheduler = loaded_scheduler
            logger.info('Scheduler checkpoint loaded!')
        else:
            logger.info('No scheduler checkpoint found!')

    def post_validation(self, logger: Logger, training_context: TrainingContext[TInput, TTarget, TOutput, TModel, TTrainer], validation_result: EvaluationResult):
        is_nested_result: bool = is_nested_dict(validation_result)

        if is_nested_result and self.dataset_name is None:
            raise TypeError('Datasetname has to be set for nested validation results.')

        current_performance: float = validation_result[self.metric_key] if not is_nested_result else validation_result[self.dataset_name][self.metric_key]
        
        logger.info('Performing scheduler step...')
        self.scheduler.step(current_performance)
        logger.info('Scheduler step performed!')

        logger.info('Creating scheduler checkpoint...')
        self.event_loop.create_task(self.repository.save(self.scheduler, LATEST_SCHEDULER_NAME))
        logger.info('Scheduler checkpoint created!')