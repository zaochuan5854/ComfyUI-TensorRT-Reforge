import onnx.onnx_cpp2py_export.version_converter as C
from onnx import ModelProto as ModelProto

def convert_version(model: ModelProto, target_version: int) -> ModelProto: ...
ConvertError = C.ConvertError
