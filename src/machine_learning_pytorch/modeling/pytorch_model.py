from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple, Union, overload
from machine_learning.modeling.model import Model, TInput, TTarget, InputBatch, Input, Target
from torch.nn import Module

PytorchTargetBatch = List[Tuple[TTarget, Any, Any]]
PytorchTarget = Union[Tuple[TTarget, Any, Any], PytorchTargetBatch[TTarget]]

class PyTorchModel(Model[TInput, TTarget], ABC):
    def __init__(self, inner_module: Module):
        super().__init__()

        if inner_module is None:
            raise TypeError("inner_module")

        self.inner_module: Module = inner_module

    @overload
    def predict_step_pytorch(self, input: TInput) -> Tuple[TTarget, Any, Any]: ...
    @overload
    def predict_step_pytorch(self, input: InputBatch[TInput]) -> PytorchTargetBatch[TTarget]: ...
    @abstractmethod
    def predict_step_pytorch(self, input: Input[TInput]) -> PytorchTarget[TTarget]: ...

    def predict_step(self, input: Input[TInput]) -> Target[TTarget]:
        return self.predict_step_pytorch(input)[0]

    __call__ : Callable[..., Any] = predict_step
