from onnx.reference.op_run import OpFunction as OpFunction, OpRun as OpRun
from onnx.reference.ops._helpers import build_registered_operators_any_domain as build_registered_operators_any_domain
from onnx.reference.ops.experimental.op_im2col import Im2Col as Im2Col
from typing import Any

def load_op(domain: str, op_type: str, version: None | int, custom: Any = None) -> Any: ...
