import numpy
from _typeshed import Incomplete
from collections.abc import Sequence
from onnx import IR_VERSION as IR_VERSION, ModelProto as ModelProto, NodeProto as NodeProto
from typing import Any

class DeviceType:
    CPU: _Type
    CUDA: _Type

class Device:
    type: Incomplete
    device_id: int
    def __init__(self, device: str) -> None: ...

def namedtupledict(typename: str, field_names: Sequence[str], *args: Any, **kwargs: Any) -> type[tuple[Any, ...]]: ...

class BackendRep:
    def run(self, inputs: Any, **kwargs: Any) -> tuple[Any, ...]: ...

class Backend:
    @classmethod
    def is_compatible(cls, model: ModelProto, device: str = 'CPU', **kwargs: Any) -> bool: ...
    @classmethod
    def prepare(cls, model: ModelProto, device: str = 'CPU', **kwargs: Any) -> BackendRep | None: ...
    @classmethod
    def run_model(cls, model: ModelProto, inputs: Any, device: str = 'CPU', **kwargs: Any) -> tuple[Any, ...]: ...
    @classmethod
    def run_node(cls, node: NodeProto, inputs: Any, device: str = 'CPU', outputs_info: Sequence[tuple[numpy.dtype, tuple[int, ...]]] | None = None, **kwargs: dict[str, Any]) -> tuple[Any, ...] | None: ...
    @classmethod
    def supports_device(cls, device: str) -> bool: ...
