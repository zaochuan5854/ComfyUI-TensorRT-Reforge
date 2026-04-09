import onnx
import onnx.onnx_cpp2py_export.shape_inference as C
import os
from collections.abc import Sequence
from onnx.onnx_pb import AttributeProto as AttributeProto, FunctionProto as FunctionProto, IR_VERSION as IR_VERSION, ModelProto as ModelProto, TypeProto as TypeProto

GraphInferencer = C.GraphInferencer
InferenceContext = C.InferenceContext

def infer_shapes(model: ModelProto | bytes, check_type: bool = False, strict_mode: bool = False, data_prop: bool = False) -> ModelProto: ...
def infer_shapes_path(model_path: str | os.PathLike, output_path: str | os.PathLike = '', check_type: bool = False, strict_mode: bool = False, data_prop: bool = False) -> None: ...
def infer_node_outputs(schema: onnx.defs.OpSchema, node: onnx.NodeProto, input_types: dict[str, onnx.TypeProto], input_data: dict[str, onnx.TensorProto] | None = None, input_sparse_data: dict[str, onnx.SparseTensorProto] | None = None, opset_imports: list[onnx.OperatorSetIdProto] | None = None, ir_version: int = ...) -> dict[str, onnx.TypeProto]: ...
def infer_function_output_types(function: FunctionProto, input_types: Sequence[TypeProto], attributes: Sequence[AttributeProto]) -> list[TypeProto]: ...
InferenceError = C.InferenceError
