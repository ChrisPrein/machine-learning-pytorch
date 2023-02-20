from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generic, List, TypeVar, Union, overload
from machine_learning.modeling.model import Model, TInput, TTarget, TOutput, InputBatch, Input, Output
from torch.nn import Module
from torch import Tensor, device

__all__ = ['PyTorchModel', 'TargetBatch', 'Target', 'TPytorchOutput']

TargetBatch = List[TTarget]
Target = Union[TTarget, TargetBatch[TTarget]]

@dataclass
class PytorchTrainStepOutput(Generic[TOutput]):
    model_output: Output[TOutput]
    module_output: Tensor
    converted_target: Tensor

TTrainStepOutput = TypeVar('TTrainStepOutput', bound=PytorchTrainStepOutput)

class PyTorchModel(Generic[TInput, TTarget, TOutput, TTrainStepOutput], Model[TInput, TOutput], ABC):
    def __init__(self, inner_module: Module, device: device = device('cpu')):
        super().__init__()

        if inner_module is None:
            raise TypeError("inner_module")

        self.inner_module: Module = inner_module
        self.device: device = device

        self.inner_module.to(self.device)

    @overload
    def training_step(self, input: TInput, target: TTarget) -> TTrainStepOutput: ...
    @overload
    def training_step(self, input: InputBatch[TInput], target: TargetBatch[TTarget]) -> TTrainStepOutput: ...
    @abstractmethod
    def training_step(self, input: Input[TInput], target: Target[TTarget]) -> TTrainStepOutput: ...

    def predict_step(self, input: Input[TInput]) -> Output[TOutput]:
        self.inner_module.eval()

        train_out: TTrainStepOutput = self.training_step(input)

        return train_out.model_output

    __call__ : Callable[..., Any] = predict_step
