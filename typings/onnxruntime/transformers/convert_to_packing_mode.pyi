from _typeshed import Incomplete
from onnx_model import NodeProto as NodeProto, OnnxModel

logger: Incomplete

class PackingAttentionBase:
    model: OnnxModel
    nodes_to_remove: list
    nodes_to_add: list
    prune_graph: bool
    node_name_to_graph_name: dict
    this_graph_name: str
    attention_op_type: Incomplete
    attention_nodes: Incomplete
    def __init__(self, model: OnnxModel, attention_op_type: str) -> None: ...
    def convert(self, use_symbolic_shape_infer: bool = True) -> None: ...

class PackingAttention(PackingAttentionBase):
    def __init__(self, model: OnnxModel) -> None: ...

class PackingMultiHeadAttention(PackingAttentionBase):
    def __init__(self, model: OnnxModel) -> None: ...

class PackingMode:
    model: Incomplete
    def __init__(self, model: OnnxModel) -> None: ...
    def convert(self, use_symbolic_shape_infer: bool = True) -> None: ...

def main() -> None: ...
