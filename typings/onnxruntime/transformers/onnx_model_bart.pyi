from _typeshed import Incomplete
from fusion_options import FusionOptions as FusionOptions
from fusion_reshape import FusionReshape
from onnx_model import OnnxModel as OnnxModel
from onnx_model_bert import BertOnnxModel

logger: Incomplete

class FusionBartReshape(FusionReshape):
    def __init__(self, model: OnnxModel) -> None: ...
    def fuse(self, reshape_node, input_name_to_nodes, output_name_to_node) -> None: ...

class BartOnnxModel(BertOnnxModel):
    attention_mask: Incomplete
    attention_fusion: Incomplete
    bart_reshape_fusion_preprocess: Incomplete
    def __init__(self, model, num_heads, hidden_size, model_impl: str = 'hf') -> None: ...
    def optimize(self, options: FusionOptions | None = None, add_dynamic_axes: bool = False): ...
    def fuse_attention(self) -> None: ...
    def preprocess(self) -> None: ...
