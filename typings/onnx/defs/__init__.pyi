import onnx.onnx_cpp2py_export.defs as C
from _typeshed import Incomplete

__all__ = ['C', 'ONNX_DOMAIN', 'ONNX_ML_DOMAIN', 'AI_ONNX_PREVIEW_TRAINING_DOMAIN', 'has', 'register_schema', 'deregister_schema', 'get_schema', 'get_all_schemas', 'get_all_schemas_with_history', 'onnx_opset_version', 'get_function_ops', 'OpSchema', 'SchemaError']

ONNX_DOMAIN: str
ONNX_ML_DOMAIN: str
AI_ONNX_PREVIEW_TRAINING_DOMAIN: str
has: Incomplete
get_schema: Incomplete
get_all_schemas = C.get_all_schemas
get_all_schemas_with_history = C.get_all_schemas_with_history
deregister_schema = C.deregister_schema

def onnx_opset_version() -> int: ...
OpSchema = C.OpSchema

def get_function_ops() -> list[OpSchema]: ...
SchemaError = C.SchemaError

def register_schema(schema: OpSchema) -> None: ...
