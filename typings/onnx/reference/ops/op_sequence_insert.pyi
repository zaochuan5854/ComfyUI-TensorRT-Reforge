import numpy as np
from onnx.reference.op_run import OpRun as OpRun
from typing import Any

def sequence_insert_reference_implementation(sequence: list[Any] | np.ndarray, tensor: np.ndarray, position: np.ndarray | None = None) -> list[Any]: ...

class SequenceInsert(OpRun): ...
