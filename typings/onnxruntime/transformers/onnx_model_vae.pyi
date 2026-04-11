from _typeshed import Incomplete
from fusion_options import FusionOptions as FusionOptions
from onnx import ModelProto as ModelProto
from onnx_model_unet import UnetOnnxModel

logger: Incomplete

class VaeOnnxModel(UnetOnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0) -> None: ...
    def fuse_multi_head_attention(self, options: FusionOptions | None = None): ...
    def get_fused_operator_statistics(self): ...
