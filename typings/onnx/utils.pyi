import os
from _typeshed import Incomplete
from onnx.onnx_pb import FunctionProto as FunctionProto, ModelProto as ModelProto, NodeProto as NodeProto, TensorProto as TensorProto, ValueInfoProto as ValueInfoProto

class Extractor:
    model: Incomplete
    graph: Incomplete
    initializers: dict[str, TensorProto]
    value_infos: dict[str, ValueInfoProto]
    outmap: dict[str, int]
    def __init__(self, model: ModelProto) -> None: ...
    def extract_model(self, input_names: list[str], output_names: list[str]) -> ModelProto: ...

def extract_model(input_path: str | os.PathLike, output_path: str | os.PathLike, input_names: list[str], output_names: list[str], check_model: bool = True, infer_shapes: bool = True) -> None: ...
