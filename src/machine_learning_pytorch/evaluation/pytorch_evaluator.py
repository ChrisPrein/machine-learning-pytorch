from logging import Logger
from typing import Optional, TypeVar
import torch
from ..modeling.pytorch_model import PyTorchModel
from machine_learning.evaluation.evaluator import Evaluator, Input, Target, EvaluatorResult
from machine_learning import Evaluator, TInput, TTarget


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

        loss, sub_losses = self.loss(raw_predictions, raw_targets)

        return predictions, dict(sub_losses)