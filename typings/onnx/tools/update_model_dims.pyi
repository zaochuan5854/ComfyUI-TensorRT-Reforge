from onnx import ModelProto as ModelProto, ValueInfoProto as ValueInfoProto
from typing import Any

def update_inputs_outputs_dims(model: ModelProto, input_dims: dict[str, list[Any]], output_dims: dict[str, list[Any]]) -> ModelProto: ...
