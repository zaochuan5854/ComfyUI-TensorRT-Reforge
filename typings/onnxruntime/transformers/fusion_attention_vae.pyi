from _typeshed import Incomplete
from fusion_base import Fusion
from onnx import NodeProto as NodeProto
from onnx_model import OnnxModel as OnnxModel

logger: Incomplete

class FusionAttentionVae(Fusion):
    hidden_size: Incomplete
    num_heads: Incomplete
    num_heads_warning: bool
    hidden_size_warning: bool
    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int) -> None: ...
    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto, add_q: NodeProto) -> tuple[int, int]: ...
    def create_attention_node(self, q_matmul: NodeProto, q_add: NodeProto, k_matmul: NodeProto, k_add: NodeProto, v_matmul: NodeProto, v_add: NodeProto, num_heads: int, hidden_size: int, input_name: str, output_name: str) -> NodeProto | None: ...
    prune_graph: bool
    def fuse(self, softmax_node, input_name_to_nodes, output_name_to_node) -> None: ...
