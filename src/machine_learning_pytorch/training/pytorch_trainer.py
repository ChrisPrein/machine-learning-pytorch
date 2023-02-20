from logging import Logger
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
from machine_learning.training.trainer import Trainer, Input, Target, TrainerResult, TInput, TTarget
from machine_learning import TOutput
import torch
from torch import device
from ..modeling.pytorch_model import PyTorchModel, TTrainStepOutput

__all__ = ['PyTorchTrainer', 'TPyTorchModel']

TPyTorchModel = TypeVar('TPyTorchModel', bound=PyTorchModel)

class PyTorchTrainer(Generic[TInput, TTarget, TOutput, TTrainStepOutput, TPyTorchModel], Trainer[TInput, TTarget, TPyTorchModel]):
    def __init__(self, loss: torch.nn.Module, optimizer: torch.optim.Optimizer, clip_max_norm: float = 0., device: device = device('cpu')):
        super().__init__()

        if loss is None:
            raise TypeError("loss")

        if optimizer is None:
            raise TypeError("optimizer")

        self.loss: torch.nn.Module = loss
        self.optimizer: torch.optim.Optimizer = optimizer
        self.clip_max_norm: float = clip_max_norm
        self.device = device

        self.loss.to(self.device)

    def train_step(self, model: TPyTorchModel, input: Input[TInput], target: Target[TTarget], logger: Optional[Logger] = None) -> TrainerResult[TOutput]:
        if model.device != self.device:
            raise ValueError("Model device has to match trainer device!")
        
        model.inner_module.train()
        self.loss.train()

        train_out: TTrainStepOutput = model.training_step(input, target)

        loss_result: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = self.loss(train_out.module_output, train_out.converted_target)

        has_sublosses = isinstance(loss_result, tuple)

        if not has_sublosses:
            loss = loss_result
            sub_losses = None
        else:
            loss, sub_losses = loss_result

        self.optimizer.zero_grad()

        loss.backward()

        if self.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.inner_module.parameters(), self.clip_max_norm)

        self.optimizer.step()

        if has_sublosses:
            return train_out.model_output, {loss_name: loss_value.item() for loss_name, loss_value in sub_losses.items()}
        else:
            return train_out.model_output, loss.item()

    __call__ : Callable[..., Any] = train_step
