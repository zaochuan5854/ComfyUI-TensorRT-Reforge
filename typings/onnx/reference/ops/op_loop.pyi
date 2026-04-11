from _typeshed import Incomplete
from onnx.reference.op_run import OpRun as OpRun

class Loop(OpRun):
    output_index: Incomplete
    N: Incomplete
    K: Incomplete
    def __init__(self, onnx_node, run_params) -> None: ...
    def need_context(self) -> bool: ...
