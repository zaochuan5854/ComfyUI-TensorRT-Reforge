import numpy as np
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from enum import IntEnum
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl as OpRunAiOnnxMl

class AggregationFunction(IntEnum):
    AVERAGE = 0
    SUM = 1
    MIN = 2
    MAX = 3

class PostTransform(IntEnum):
    NONE = 0
    SOFTMAX = 1
    LOGISTIC = 2
    SOFTMAX_ZERO = 3
    PROBIT = 4

class Mode(IntEnum):
    LEQ = 0
    LT = 1
    GTE = 2
    GT = 3
    EQ = 4
    NEQ = 5
    MEMBER = 6

class Leaf:
    weight: Incomplete
    target_id: Incomplete
    def __init__(self, weight: float, target_id: int) -> None: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...

class Node:
    compare: Callable[[float, float | set[float]], bool]
    true_branch: Node | Leaf
    false_branch: Node | Leaf
    feature: int
    mode: Incomplete
    value: Incomplete
    def __init__(self, mode: Mode, value: float | set[float], feature: int, missing_tracks_true: bool) -> None: ...
    def predict(self, x: np.ndarray) -> float: ...

class TreeEnsemble(OpRunAiOnnxMl): ...
