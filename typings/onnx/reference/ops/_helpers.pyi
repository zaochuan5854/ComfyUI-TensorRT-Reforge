from onnx.reference.op_run import OpRun as OpRun
from typing import Any

def build_registered_operators_any_domain(module_context: dict[str, Any]) -> dict[str, dict[int | None, type[OpRun]]]: ...
