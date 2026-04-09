import onnx
from ..onnx_model import ONNXModel as ONNXModel
from .fusion import Fusion as Fusion

class FusionLayerNormalization(Fusion):
    def __init__(self, model: ONNXModel) -> None: ...
    def fuse(self, reduce_mean_node: onnx.NodeProto, input_name_to_nodes: dict[str, list[onnx.NodeProto]], output_name_to_node: dict[str, onnx.NodeProto]): ...
