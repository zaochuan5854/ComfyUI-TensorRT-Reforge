from _typeshed import Incomplete
from onnx import ModelProto as ModelProto
from onnx.backend.base import Backend
from onnxruntime import InferenceSession as InferenceSession, SessionOptions as SessionOptions, get_available_providers as get_available_providers, get_device as get_device
from onnxruntime.backend.backend_rep import OnnxRuntimeBackendRep as OnnxRuntimeBackendRep

class OnnxRuntimeBackend(Backend):
    allowReleasedOpsetsOnly: Incomplete
    @classmethod
    def is_compatible(cls, model, device=None, **kwargs): ...
    @classmethod
    def is_opset_supported(cls, model): ...
    @classmethod
    def supports_device(cls, device): ...
    @classmethod
    def prepare(cls, model, device=None, **kwargs): ...
    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs): ...
    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs) -> None: ...

is_compatible: Incomplete
prepare: Incomplete
run: Incomplete
supports_device: Incomplete
