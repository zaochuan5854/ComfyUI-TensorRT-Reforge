from onnx.reference.ops.aionnx_preview_training.op_adagrad import Adagrad as Adagrad
from onnx.reference.ops.aionnx_preview_training.op_adam import Adam as Adam
from onnx.reference.ops.aionnx_preview_training.op_momentum import Momentum as Momentum
from typing import Any

__all__ = ['load_op', 'Adagrad', 'Adam', 'Momentum']

def load_op(domain: str, op_type: str, version: None | int, custom: Any = None) -> Any: ...
