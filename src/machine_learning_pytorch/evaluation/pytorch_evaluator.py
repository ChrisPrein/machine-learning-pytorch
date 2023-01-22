from logging import Logger
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
import torch
from ..modeling.pytorch_model import PyTorchModel
from machine_learning.evaluation.evaluator import Evaluator, Input, Target, EvaluatorResult
from machine_learning import Evaluator, TInput, TTarget

__all__ = ['PyTorchEvaluator', 'TPyTorchModel']

TPyTorchModel = TypeVar('TPyTorchModel', bound=PyTorchModel)

class PyTorchEvaluator(Evaluator[TInput, TTarget, TPyTorchModel]):
    def __init__(self, loss: torch.nn.Module):
        super().__init__()

        if loss is None:
            raise TypeError("loss")

        self.loss: torch.nn.Module = loss

    def evaluation_step(self, model: TPyTorchModel, input: Input[TInput], target: Target[TTarget], logger: Optional[Logger] = None) -> EvaluatorResult[TTarget]:
        model.inner_module.eval()
        self.loss.eval()

        predictions, raw_predictions, raw_targets = model.training_step(input, target)

        loss_result: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = self.loss(raw_predictions, raw_targets)

        has_sublosses = isinstance(loss_result, tuple)

        if not has_sublosses:
            loss = loss_result
            return predictions, loss.item()
        else:
            loss, sub_losses = loss_result
            return predictions, {loss_name: loss_value.item() for loss_name, loss_value in sub_losses.items()}

    __call__ : Callable[..., Any] = evaluation_step