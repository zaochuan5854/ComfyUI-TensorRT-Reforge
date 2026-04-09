import numpy as np
import onnx.model_container
from _typeshed import Incomplete
from onnx.onnx_pb import FunctionProto as FunctionProto, GraphProto as GraphProto, ModelProto as ModelProto, NodeProto as NodeProto, TensorProto as TensorProto, TypeProto as TypeProto
from onnx.reference import op_run as op_run
from onnx.reference.ops_optimized import optimized_operators as optimized_operators
from typing import Any

class ReferenceEvaluator:
    output_types_: Incomplete
    input_types_: Incomplete
    container_: onnx.model_container.ModelContainer | None
    proto_: Incomplete
    functions_: dict[tuple[str, str], ReferenceEvaluator]
    attributes_: list[str]
    onnx_graph_: Incomplete
    opsets_: Incomplete
    input_names_: Incomplete
    output_names_: Incomplete
    inits_: Incomplete
    nodes_: Incomplete
    all_types_: Incomplete
    verbose: Incomplete
    new_ops_: dict[tuple[str, str], type[op_run.OpRun]]
    def __init__(self, proto: Any, opsets: dict[str, int] | None = None, functions: list[ReferenceEvaluator | FunctionProto] | None = None, verbose: int = 0, new_ops: list[type[op_run.OpRun]] | None = None, optimized: bool = True) -> None: ...
    def retrieve_external_data(self, initializer: TensorProto) -> np.ndarray: ...
    @property
    def input_names(self): ...
    @property
    def input_types(self): ...
    @property
    def output_names(self): ...
    @property
    def output_types(self): ...
    @property
    def opsets(self): ...
    @property
    def has_linked_attribute(self): ...
    def get_result_types(self, name: str, exc: bool = True) -> Any: ...
    def run(self, output_names, feed_inputs: dict[str, Any], attributes: dict[str, Any] | None = None, intermediate: bool = False) -> dict[str, Any] | list[Any]: ...
