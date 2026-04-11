import numpy as np
import numpy.typing as npt
from .onnx_model import ONNXModel as ONNXModel
from .quant_utils import attribute_to_kwarg as attribute_to_kwarg
from _typeshed import Incomplete
from onnx.onnx_pb import GraphProto as GraphProto, ModelProto as ModelProto, NodeProto as NodeProto, TensorProto as TensorProto
from onnxruntime.capi._pybind_state import quantize_matmul_bnb4 as quantize_matmul_bnb4

logger: Incomplete

class MatMulBnb4Quantizer:
    FP4: int
    NF4: int
    model: Incomplete
    quant_type: Incomplete
    block_size: Incomplete
    nodes_to_exclude: Incomplete
    def __init__(self, model: ModelProto, quant_type: int, block_size: int, nodes_to_exclude=None) -> None: ...
    def bnb4_block_quant(self, fpweight: npt.ArrayLike) -> np.ndarray: ...
    def process(self) -> None: ...

def parse_args(): ...
