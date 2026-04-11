import abc
from onnx.helper import tensor_dtype_to_np_dtype as tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun as OpRun

class _CommonWindow(OpRun, metaclass=abc.ABCMeta): ...
