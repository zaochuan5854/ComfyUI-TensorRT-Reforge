from _typeshed import Incomplete
from fusion_base import Fusion
from onnx_model import OnnxModel as OnnxModel

logger: Incomplete

class FusionQuickGelu(Fusion):
    def __init__(self, model: OnnxModel) -> None: ...
    def fuse(self, node, input_name_to_nodes, output_name_to_node) -> None: ...
