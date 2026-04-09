import abc
from _typeshed import Incomplete
from onnx.reference.op_run import OpRun as OpRun, RefAttrName as RefAttrName

class ConstantCommon(OpRun, metaclass=abc.ABCMeta): ...

class Constant_1(ConstantCommon):
    cst: Incomplete
    def __init__(self, onnx_node, run_params) -> None: ...

class Constant_9(Constant_1):
    def __init__(self, onnx_node, run_params) -> None: ...

class Constant_11(ConstantCommon):
    cst: Incomplete
    def __init__(self, onnx_node, run_params) -> None: ...

class Constant_12(ConstantCommon):
    cst_name: str
    cst: Incomplete
    cst_convert: Incomplete
    def __init__(self, onnx_node, run_params) -> None: ...
