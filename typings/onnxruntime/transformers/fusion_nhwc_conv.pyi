from _typeshed import Incomplete
from fusion_base import Fusion
from onnx_model import OnnxModel as OnnxModel

logger: Incomplete

class FusionNhwcConv(Fusion):
    update_weight: Incomplete
    fusion_utils: Incomplete
    def __init__(self, model: OnnxModel, update_weight: bool = False) -> None: ...
    def create_transpose_node(self, input_name: str, perm: list[int], output_name=None): ...
    def fuse(self, conv, input_name_to_nodes, output_name_to_node) -> None: ...
