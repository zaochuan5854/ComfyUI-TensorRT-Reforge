from _typeshed import Incomplete
from onnx import ModelProto as ModelProto
from onnx_model_bert import BertOnnxModel

logger: Incomplete

class ClipOnnxModel(BertOnnxModel):
    clip_attention_fusion: Incomplete
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0) -> None: ...
    def get_fused_operator_statistics(self): ...
    def fuse_attention(self) -> None: ...
