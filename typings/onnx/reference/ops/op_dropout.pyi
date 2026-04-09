import abc
from _typeshed import Incomplete
from onnx.reference.op_run import OpRun as OpRun

class DropoutBase(OpRun, metaclass=abc.ABCMeta):
    n_outputs: Incomplete
    def __init__(self, onnx_node, run_params) -> None: ...

class Dropout_7(DropoutBase): ...
class Dropout_12(DropoutBase): ...
