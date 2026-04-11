import onnx
from ..onnx_model import ONNXModel as ONNXModel
from .fusion import Fusion as Fusion
from _typeshed import Incomplete

class ReplaceUpsampleWithResize(Fusion):
    opset: Incomplete
    def __init__(self, model: ONNXModel, opset) -> None: ...
    def fuse(self, node: onnx.NodeProto, input_name_to_nodes: dict[str, list[onnx.NodeProto]], output_name_to_node: dict[str, onnx.NodeProto]): ...
    def apply(self) -> bool: ...
