from logging import Logger
from typing import Optional, TypeVar
from machine_learning.evaluation.evaluator import Evaluator, INPUT, TARGET, EVALUATOR_RESULT, TInput, TTarget, TModel
import torch
from ..modeling.pytorch_model import PyTorchModel

TPyTorchModel = TypeVar('TPyTorchModel', bound=PyTorchModel)

class PyTorchTrainer(Evaluator[TInput, TTarget, TPyTorchModel]):
    def __init__(self, loss: torch.nn.Module, optimizer: torch.optim.Optimizer, clip_max_norm: float = 0.):
        super().__init__()

        if loss is None:
            raise TypeError("loss")

        if optimizer is None:
            raise TypeError("optimizer")

        self.loss: torch.nn.Module = loss
        self.optimizer: torch.optim.Optimizer = optimizer
        self.clip_max_norm: float = clip_max_norm

    def evaluation_step(self, model: TPyTorchModel, input: INPUT, target: TARGET, logger: Optional[Logger] = None) -> EVALUATOR_RESULT:
        model.inner_module.train()
        self.loss.train()

        predictions, raw_predictions, raw_targets = model.predict_step_pytorch(input)

        loss, sub_losses = self.loss(raw_predictions, raw_targets)

        return predictions, dict(sub_losses)