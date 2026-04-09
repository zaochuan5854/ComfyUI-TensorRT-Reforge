from onnx.reference.ops._op import OpRunBinaryNumpy as OpRunBinaryNumpy

class Div(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params) -> None: ...
