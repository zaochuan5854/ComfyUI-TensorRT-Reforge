from onnx.reference.ops._op import OpRunBinaryNumpy as OpRunBinaryNumpy

class Sub(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params) -> None: ...
