import onnx
from .calibrate import CalibrationDataReader as CalibrationDataReader, CalibrationMethod as CalibrationMethod, TensorsData as TensorsData, create_calibrator as create_calibrator
from .onnx_quantizer import ONNXQuantizer as ONNXQuantizer
from .qdq_quantizer import QDQQuantizer as QDQQuantizer
from .quant_utils import MODEL_SIZE_THRESHOLD as MODEL_SIZE_THRESHOLD, QuantFormat as QuantFormat, QuantType as QuantType, QuantizationMode as QuantizationMode, load_model_with_shape_infer as load_model_with_shape_infer, model_has_pre_process_metadata as model_has_pre_process_metadata, save_and_reload_model_with_shape_infer as save_and_reload_model_with_shape_infer, update_opset_version as update_opset_version
from .registry import IntegerOpsRegistry as IntegerOpsRegistry, QDQRegistry as QDQRegistry, QLinearOpsRegistry as QLinearOpsRegistry
from .tensor_quant_overrides import TensorQuantOverridesHelper as TensorQuantOverridesHelper
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from pathlib import Path
from typing import Any

class QuantConfig:
    op_types_to_quantize: Incomplete
    per_channel: Incomplete
    reduce_range: Incomplete
    weight_type: Incomplete
    activation_type: Incomplete
    nodes_to_quantize: Incomplete
    nodes_to_exclude: Incomplete
    use_external_data_format: Incomplete
    def __init__(self, activation_type=..., weight_type=..., op_types_to_quantize=None, nodes_to_quantize=None, nodes_to_exclude=None, per_channel: bool = False, reduce_range: bool = False, use_external_data_format: bool = False) -> None: ...

class StaticQuantConfig(QuantConfig):
    calibration_data_reader: Incomplete
    calibrate_method: Incomplete
    quant_format: Incomplete
    calibration_providers: Incomplete
    extra_options: Incomplete
    def __init__(self, calibration_data_reader: CalibrationDataReader, calibrate_method=..., quant_format=..., activation_type=..., weight_type=..., op_types_to_quantize=None, nodes_to_quantize=None, nodes_to_exclude=None, per_channel: bool = False, reduce_range: bool = False, use_external_data_format: bool = False, calibration_providers=None, extra_options=None) -> None: ...

def get_qdq_config(model_input: str | Path | onnx.ModelProto, calibration_data_reader: CalibrationDataReader, calibrate_method=..., calibrate_args: dict[str, Any] | None = None, activation_type=..., weight_type=..., activation_symmetric: bool = False, weight_symmetric: bool | None = None, per_channel: bool = False, reduce_range: bool = False, keep_removable_activations: bool = False, min_real_range: float | None = None, tensor_quant_overrides: dict[str, list[dict[str, Any]]] | None = None, calibration_providers: list[str] | None = None, op_types_to_quantize: list[str] | None = None, nodes_to_exclude: list[str] | Callable[[onnx.ModelProto, onnx.NodeProto], bool] | None = None, extra_options: dict | None = None) -> StaticQuantConfig: ...

class DynamicQuantConfig(QuantConfig):
    extra_options: Incomplete
    def __init__(self, weight_type=..., op_types_to_quantize=None, nodes_to_quantize=None, nodes_to_exclude=None, per_channel: bool = False, reduce_range: bool = False, use_external_data_format: bool = False, extra_options=None) -> None: ...

def check_static_quant_arguments(quant_format: QuantFormat, activation_type: QuantType, weight_type: QuantType): ...
def quantize_static(model_input: str | Path | onnx.ModelProto, model_output: str | Path, calibration_data_reader: CalibrationDataReader, quant_format=..., op_types_to_quantize=None, per_channel: bool = False, reduce_range: bool = False, activation_type=..., weight_type=..., nodes_to_quantize=None, nodes_to_exclude=None, use_external_data_format: bool = False, calibrate_method=..., calibration_providers=None, extra_options=None): ...
def quantize_dynamic(model_input: str | Path | onnx.ModelProto, model_output: str | Path, op_types_to_quantize=None, per_channel: bool = False, reduce_range: bool = False, weight_type=..., nodes_to_quantize=None, nodes_to_exclude=None, use_external_data_format: bool = False, extra_options=None): ...
def quantize(model_input: str | Path | onnx.ModelProto, model_output: str | Path, quant_config: QuantConfig): ...
