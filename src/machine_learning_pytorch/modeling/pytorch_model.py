from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple, Union, overload
from machine_learning.modeling.model import Model, TInput, TTarget, InputBatch, Input, Target, TargetBatch
from torch.nn import Module

__all__ = ['PyTorchModel', 'PytorchTargetBatch', 'PytorchTarget']

PytorchTargetBatch = List[Tuple[TTarget, Any, Any]]
PytorchTarget = Union[Tuple[TTarget, Any, Any], PytorchTargetBatch[TTarget]]

class PyTorchModel(Model[TInput, TTarget], ABC):
    def __init__(self, inner_module: Module):
        super().__init__()

        if inner_module is None:
            raise TypeError("inner_module")

        self.inner_module: Module = inner_module

    @overload
    def training_step(self, input: TInput, target: TTarget) -> Tuple[TTarget, Any, Any]: ...
    @overload
    def training_step(self, input: InputBatch[TInput], target: TargetBatch[TTarget]) -> PytorchTargetBatch[TTarget]: ...
    @abstractmethod
    def training_step(self, input: Input[TInput], target: Target[TTarget]) -> PytorchTarget[TTarget]: ...

    def predict_step(self, input: Input[TInput]) -> Target[TTarget]:
        self.inner_module.eval()
        return self.training_step(input)[0]

    __call__ : Callable[..., Any] = predict_step
