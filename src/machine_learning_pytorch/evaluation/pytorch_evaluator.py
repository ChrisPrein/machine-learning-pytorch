from logging import Logger
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
import torch
from ..modeling.pytorch_model import PyTorchModel, TTrainStepOutput
from machine_learning.modeling import Output
from machine_learning.evaluation.evaluator import Evaluator, Input, Target, EvaluatorResult
from machine_learning import Evaluator, TInput, TTarget, TOutput

__all__ = ['PyTorchEvaluator', 'TPyTorchModel']

TPyTorchModel = TypeVar('TPyTorchModel', bound=PyTorchModel)

class PyTorchEvaluator(Generic[TInput, TTarget, TOutput, TTrainStepOutput, TPyTorchModel], Evaluator[TInput, TTarget, TOutput, TPyTorchModel]):
    def __init__(self, loss: torch.nn.Module):
        super().__init__()

        if loss is None:
            raise TypeError("loss")

        self.loss: torch.nn.Module = loss

    def evaluation_step(self, model: TPyTorchModel, input: Input[TInput], target: Target[TTarget], logger: Optional[Logger] = None) -> EvaluatorResult[TOutput]:
        model.inner_module.eval()
        self.loss.eval()

        train_out: TTrainStepOutput = model.training_step(input, target)

        loss_result: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = self.loss(train_out.module_output, train_out.converted_target)

        has_sublosses = isinstance(loss_result, tuple)

        if not has_sublosses:
            loss = loss_result
            return train_out.model_output, loss.item()
        else:
            loss, sub_losses = loss_result
            return train_out.model_output, {loss_name: loss_value.item() for loss_name, loss_value in sub_losses.items()}

    __call__ : Callable[..., Any] = evaluation_step