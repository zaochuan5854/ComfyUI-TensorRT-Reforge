from _typeshed import Incomplete
from fusion_attention import FusionAttention
from onnx import NodeProto as NodeProto
from onnx_model import OnnxModel as OnnxModel

logger: Incomplete

class FusionAttentionClip(FusionAttention):
    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int) -> None: ...
    num_heads_warning: bool
    hidden_size_warning: bool
    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto) -> tuple[int, int]: ...
    prune_graph: bool
    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node) -> None: ...
