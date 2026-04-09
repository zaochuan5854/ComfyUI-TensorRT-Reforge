from _typeshed import Incomplete
from onnx.reference.op_run import OpRun as OpRun

class Scan(OpRun):
    input_directions_: Incomplete
    input_axes_: Incomplete
    input_names: Incomplete
    output_names: Incomplete
    def __init__(self, onnx_node, run_params) -> None: ...
