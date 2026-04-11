from _typeshed import Incomplete
from fusion_attention import AttentionMask as AttentionMask, FusionAttention
from onnx_model import OnnxModel as OnnxModel

logger: Incomplete

class FusionBartAttention(FusionAttention):
    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int, attention_mask: AttentionMask) -> None: ...
    use_multi_head_attention: bool
    prune_graph: bool
    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node) -> None: ...
