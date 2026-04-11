from _typeshed import Incomplete
from collections import defaultdict
from collections.abc import Sequence
from onnx import NodeProto as NodeProto, TensorProto as TensorProto
from onnx_model import OnnxModel as OnnxModel
from typing import Any

logger: Incomplete

class Fusion:
    search_op_types: list[str]
    fused_op_type: str
    description: str
    model: OnnxModel
    nodes_to_remove: list
    nodes_to_add: list
    prune_graph: bool
    node_name_to_graph_name: dict
    this_graph_name: str | None
    fused_count: defaultdict
    def __init__(self, model: OnnxModel, fused_op_type: str, search_op_types: str | list[str], description: str = '') -> None: ...
    def increase_counter(self, fused_op_name: str): ...
    def fuse(self, node: NodeProto, input_name_to_nodes: dict[str, list[NodeProto]], output_name_to_node: dict[str, NodeProto]): ...
    def apply(self) -> None: ...
    def add_initializer(self, name: str, data_type: int, dims: Sequence[int], vals: Any, raw: bool = True): ...
    def remove_initializer(self, tensor: TensorProto): ...
    def add_nodes_to_remove(self, nodes: list[NodeProto]): ...
    def add_nodes_to_remove_with_nodes_to_keep(self, nodes: list[NodeProto], nodes_to_keep: list[NodeProto]): ...
