from onnx.reference.ops._op import OpRunUnaryNum as OpRunUnaryNum

class Erf(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params) -> None: ...
