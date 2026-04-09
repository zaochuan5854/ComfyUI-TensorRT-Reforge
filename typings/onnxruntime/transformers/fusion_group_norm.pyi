from _typeshed import Incomplete
from fusion_base import Fusion
from onnx_model import OnnxModel as OnnxModel

logger: Incomplete

class FusionGroupNorm(Fusion):
    channels_last: Incomplete
    def __init__(self, model: OnnxModel, channels_last: bool = True) -> None: ...
    prune_graph: bool
    def fuse(self, add_node, input_name_to_nodes: dict, output_name_to_node: dict): ...
