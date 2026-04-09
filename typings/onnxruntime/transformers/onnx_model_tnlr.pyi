from _typeshed import Incomplete
from fusion_attention import AttentionMask, FusionAttention
from onnx import NodeProto as NodeProto
from onnx_model import OnnxModel as OnnxModel
from onnx_model_bert import BertOnnxModel

logger: Incomplete

class FusionTnlrAttention(FusionAttention):
    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int, attention_mask: AttentionMask) -> None: ...
    def create_attention_node(self, mask_index: str, matmul: NodeProto, add: NodeProto, num_heads: int, hidden_size: int, input: str, output: str, add_qk_str: str) -> NodeProto | None: ...
    prune_graph: bool
    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node) -> None: ...

class TnlrOnnxModel(BertOnnxModel):
    attention_mask: Incomplete
    attention_fusion: Incomplete
    def __init__(self, model, num_heads, hidden_size) -> None: ...
    def fuse_attention(self) -> None: ...
