from _typeshed import Incomplete
from fusion_base import Fusion
from onnx_model import OnnxModel as OnnxModel

logger: Incomplete

class FusionReshape(Fusion):
    prune_graph: bool
    def __init__(self, model: OnnxModel) -> None: ...
    def replace_reshape_node(self, shape, reshape_node, concat_node) -> None: ...
    def fuse(self, reshape_node, input_name_to_nodes, output_name_to_node) -> None: ...
