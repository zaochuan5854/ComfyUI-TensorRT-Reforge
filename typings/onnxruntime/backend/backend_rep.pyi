from onnx.backend.base import BackendRep
from onnxruntime import RunOptions as RunOptions
from typing import Any

class OnnxRuntimeBackendRep(BackendRep):
    def __init__(self, session) -> None: ...
    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]: ...
