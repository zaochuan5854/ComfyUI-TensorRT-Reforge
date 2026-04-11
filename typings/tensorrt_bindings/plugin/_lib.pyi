import tensorrt as trt
import types
from ._export import IS_AOT_ENABLED as IS_AOT_ENABLED, public_api as public_api
from _typeshed import Incomplete
from typing import Callable

class _PluginNamespace(types.ModuleType):
    def __init__(self, namespace) -> None: ...
    def define(self, name, plugin_def) -> None: ...
    def __getattr__(self, name) -> None: ...

class _Op(types.ModuleType):
    def __init__(self) -> None: ...
    def define_or_get(self, namespace): ...
    def __getattr__(self, name) -> None: ...

op: Incomplete
QDP_CREATORS: Incomplete
QDP_REGISTRY: Incomplete

class PluginDef:
    plugin_id: Incomplete
    register_func: Incomplete
    impl_func: Incomplete
    aot_impl_func: Incomplete
    autotune_func: Incomplete
    autotune_attr_names: Incomplete
    input_tensor_names: Incomplete
    input_attrs: Incomplete
    impl_attr_names: Incomplete
    aot_impl_attr_names: Incomplete
    num_outputs: Incomplete
    input_arg_schema: Incomplete
    expects_tactic: Incomplete
    def __init__(self) -> None: ...
    def __call__(self, *args, **kwargs) -> tuple[list[trt.ITensor], list[trt.ITensor], trt.IPluginV3]: ...

class _TemplatePluginCreator(trt.IPluginCreatorV3Quick):
    name: Incomplete
    plugin_namespace: Incomplete
    plugin_version: str
    field_names: Incomplete
    def __init__(self, name, namespace, attrs) -> None: ...
    def create_plugin(self, name, namespace, fc, phase, qpcr: trt.QuickPluginCreationRequest = None): ...

def register(plugin_id: str, lazy_register: bool = False) -> Callable: ...
def impl(plugin_id: str) -> Callable: ...
def aot_impl(plugin_id: str) -> Callable: ...
def autotune(plugin_id: str) -> Callable: ...
