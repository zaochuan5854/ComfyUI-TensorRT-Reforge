import abc
from onnx.reference.op_run import OpRun as OpRun

class OpRunTraining(OpRun, metaclass=abc.ABCMeta):
    op_domain: str
