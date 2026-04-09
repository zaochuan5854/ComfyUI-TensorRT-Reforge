import abc
from onnx.helper import tensor_dtype_to_np_dtype as tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun as OpRun

class _CommonRandom(OpRun, metaclass=abc.ABCMeta):
    def __init__(self, onnx_node, run_params) -> None: ...
    @staticmethod
    def numpy_type(dtype): ...
