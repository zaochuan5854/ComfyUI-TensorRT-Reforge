import numpy as np
from onnx.reference.op_run import OpRun as OpRun

class SplitToSequence(OpRun):
    def common_run(self, mat: np.ndarray, split: np.ndarray | None, axis: int) -> list[np.ndarray]: ...
