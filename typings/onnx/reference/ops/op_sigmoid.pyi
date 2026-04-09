import numpy as np
from onnx.reference.ops._op import OpRunUnaryNum as OpRunUnaryNum

def sigmoid(x: np.ndarray) -> np.ndarray: ...

class Sigmoid(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params) -> None: ...
