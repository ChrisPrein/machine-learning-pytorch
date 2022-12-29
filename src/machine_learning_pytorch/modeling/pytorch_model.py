from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, overload
from machine_learning.modeling.model import Model, TInput, TTarget, InputBatch, Input, Target
from torch.nn import Module

PYTORCH_TARGET_BATCH = List[Tuple[TTarget, Any, Any]]
PYTORCH_TARGET = Union[Tuple[TTarget, Any, Any], PYTORCH_TARGET_BATCH]

class PyTorchModel(Model, ABC):
    def __init__(self, inner_module: Module):
        super().__init__()

        if inner_module is None:
            raise TypeError("inner_module")

        self.inner_module: Module = inner_module

    @overload
    def predict_step_pytorch(self, input: TInput) -> Tuple[TTarget, Any, Any]: ...
    @overload
    def predict_step_pytorch(self, input: InputBatch) -> PYTORCH_TARGET_BATCH: ...
    @abstractmethod
    def predict_step_pytorch(self, input: Input) -> PYTORCH_TARGET: ...

    def predict_step(self, input: Input) -> Target:
        return self.predict_step_pytorch(input)[0]
