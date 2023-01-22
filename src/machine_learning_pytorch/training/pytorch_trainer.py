from logging import Logger
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from machine_learning.training.trainer import Trainer, Input, Target, TrainerResult, TInput, TTarget, TModel
import torch
from ..modeling.pytorch_model import PyTorchModel

__all__ = ['PyTorchTrainer', 'TPyTorchModel']

TPyTorchModel = TypeVar('TPyTorchModel', bound=PyTorchModel)

class PyTorchTrainer(Trainer[TInput, TTarget, TPyTorchModel]):
    def __init__(self, loss: torch.nn.Module, optimizer: torch.optim.Optimizer, clip_max_norm: float = 0.):
        super().__init__()

        if loss is None:
            raise TypeError("loss")

        if optimizer is None:
            raise TypeError("optimizer")

        self.loss: torch.nn.Module = loss
        self.optimizer: torch.optim.Optimizer = optimizer
        self.clip_max_norm: float = clip_max_norm

    def train_step(self, model: TPyTorchModel, input: Input[TInput], target: Target[TTarget], logger: Optional[Logger] = None) -> TrainerResult[TTarget]:
        model.inner_module.train()
        self.loss.train()

        predictions, raw_predictions, raw_targets = model.training_step(input, target)

        loss_result: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = self.loss(raw_predictions, raw_targets)

        if not isinstance(loss_result, tuple):
            loss = loss_result
            sub_losses = {'loss': loss}
        else:
            loss, sub_losses = loss_result

        self.optimizer.zero_grad()

        loss.backward()

        if self.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.inner_module.parameters(), self.clip_max_norm)

        self.optimizer.step()

        return predictions, dict(sub_losses)

    __call__ : Callable[..., Any] = train_step
