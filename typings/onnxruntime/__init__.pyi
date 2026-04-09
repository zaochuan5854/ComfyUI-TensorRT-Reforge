from . import experimental as experimental
from _typeshed import Incomplete
from onnxruntime.capi import onnxruntime_validation as onnxruntime_validation
from onnxruntime.capi._pybind_state import ExecutionMode as ExecutionMode, ExecutionOrder as ExecutionOrder, GraphOptimizationLevel as GraphOptimizationLevel, LoraAdapter as LoraAdapter, ModelMetadata as ModelMetadata, NodeArg as NodeArg, OrtAllocatorType as OrtAllocatorType, OrtArenaCfg as OrtArenaCfg, OrtCompileApiFlags as OrtCompileApiFlags, OrtDeviceMemoryType as OrtDeviceMemoryType, OrtEpAssignedNode as OrtEpAssignedNode, OrtEpAssignedSubgraph as OrtEpAssignedSubgraph, OrtEpDevice as OrtEpDevice, OrtExecutionProviderDevicePolicy as OrtExecutionProviderDevicePolicy, OrtExternalInitializerInfo as OrtExternalInitializerInfo, OrtHardwareDevice as OrtHardwareDevice, OrtHardwareDeviceType as OrtHardwareDeviceType, OrtMemType as OrtMemType, OrtMemoryInfo as OrtMemoryInfo, OrtMemoryInfoDeviceType as OrtMemoryInfoDeviceType, OrtSparseFormat as OrtSparseFormat, OrtSyncStream as OrtSyncStream, RunOptions as RunOptions, SessionIOBinding as SessionIOBinding, SessionOptions as SessionOptions, create_and_register_allocator as create_and_register_allocator, create_and_register_allocator_v2 as create_and_register_allocator_v2, disable_telemetry_events as disable_telemetry_events, enable_telemetry_events as enable_telemetry_events, get_all_providers as get_all_providers, get_available_providers as get_available_providers, get_build_info as get_build_info, get_device as get_device, get_ep_devices as get_ep_devices, get_version_string as get_version_string, has_collective_ops as has_collective_ops, register_execution_provider_library as register_execution_provider_library, set_default_logger_severity as set_default_logger_severity, set_default_logger_verbosity as set_default_logger_verbosity, set_global_thread_pool_sizes as set_global_thread_pool_sizes, set_seed as set_seed, unregister_execution_provider_library as unregister_execution_provider_library
from onnxruntime.capi.onnxruntime_inference_collection import AdapterFormat as AdapterFormat, IOBinding as IOBinding, InferenceSession as InferenceSession, ModelCompiler as ModelCompiler, OrtDevice as OrtDevice, OrtValue as OrtValue, SparseTensor as SparseTensor, copy_tensors as copy_tensors

__version__: str
import_capi_exception: Incomplete
import_capi_exception = e
package_name: Incomplete
version: Incomplete
cuda_version: Incomplete
__version__ = version

def print_debug_info(): ...
def preload_dlls(cuda: bool = True, cudnn: bool = True, msvc: bool = True, directory=None): ...
