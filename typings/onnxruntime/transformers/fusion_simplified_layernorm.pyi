from _typeshed import Incomplete
from fusion_base import Fusion
from fusion_skiplayernorm import FusionSkipLayerNormalization
from onnx_model import OnnxModel as OnnxModel

logger: Incomplete

class FusionSimplifiedLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel) -> None: ...
    def fuse(self, node, input_name_to_nodes: dict, output_name_to_node: dict): ...

class FusionSkipSimplifiedLayerNormalization(FusionSkipLayerNormalization):
    def __init__(self, model: OnnxModel) -> None: ...
    def fuse(self, node, input_name_to_nodes, output_name_to_node) -> None: ...
