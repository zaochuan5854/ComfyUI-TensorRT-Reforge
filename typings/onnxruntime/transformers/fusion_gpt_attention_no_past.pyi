from _typeshed import Incomplete
from fusion_base import Fusion
from onnx_model import OnnxModel as OnnxModel

logger: Incomplete

class FusionGptAttentionNoPast(Fusion):
    num_heads: Incomplete
    mask_filter_value: Incomplete
    def __init__(self, model: OnnxModel, num_heads: int) -> None: ...
    def create_attention_node(self, gemm, gemm_qkv, input, output) -> None: ...
    prune_graph: bool
    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node) -> None: ...
