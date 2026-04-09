from _typeshed import Incomplete
from onnx.reference.ops._op import OpRunBinaryNumpy as OpRunBinaryNumpy

class BitShift(OpRunBinaryNumpy):
    numpy_fct: Incomplete
    def __init__(self, onnx_node, run_params) -> None: ...
