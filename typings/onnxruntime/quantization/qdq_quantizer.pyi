import onnx
from .base_quantizer import BaseQuantizer as BaseQuantizer, QuantizationParams as QuantizationParams
from .calibrate import TensorData as TensorData
from .quant_utils import DEQUANT_OP_NAME as DEQUANT_OP_NAME, ONNX_TYPE_TO_NP_TYPE as ONNX_TYPE_TO_NP_TYPE, QUANT_OP_NAME as QUANT_OP_NAME, QuantizedValue as QuantizedValue, QuantizedValueType as QuantizedValueType, __producer__ as __producer__, __version__ as __version__, add_dequant_output_suffix as add_dequant_output_suffix, add_dequant_suffix as add_dequant_suffix, add_quant_input_suffix as add_quant_input_suffix, add_quant_output_suffix as add_quant_output_suffix, add_quant_suffix as add_quant_suffix, compute_data_quant_params as compute_data_quant_params, compute_scale_zp as compute_scale_zp, compute_scale_zp_float8 as compute_scale_zp_float8, find_by_name as find_by_name, get_qmin_qmax_for_qType as get_qmin_qmax_for_qType, ms_domain as ms_domain, normalize_axis as normalize_axis, quantize_onnx_initializer as quantize_onnx_initializer, tensor_proto_to_array as tensor_proto_to_array
from .registry import CreateQDQQuantizer as CreateQDQQuantizer
from _typeshed import Incomplete
from dataclasses import dataclass
from enum import Enum
from onnx import TensorProto
from typing import Any

class QDQQuantTensorType(Enum):
    ACTIVATION = 0
    WEIGHT = 1
    BIAS = 2

@dataclass
class QDQQuantParamProvider:
    input_name: str
    node_name: str

class QDQTensorQuantInfo:
    tensor_type: Incomplete
    quant_para_provider: Incomplete
    axis: Incomplete
    is_shared: Incomplete
    data_type: Incomplete
    def __init__(self, tensor_type=..., quant_para_provider=None, axis=None, data_type=None) -> None: ...

@dataclass
class QDQBiasQuantInfo:
    node_name: str
    input_name: str
    weight_name: str
    beta: float

@dataclass
class QDQTensorQuantParams:
    original: QuantizationParams
    converted: QuantizationParams | None
    converted_recv_nodes: set[str] | None
    def get_for_consumer(self, consumer_node_name) -> QuantizationParams: ...

@dataclass
class QDQScaleZpInitializers:
    scale: TensorProto
    zero_point: TensorProto

@dataclass
class QDQTensorScaleZpInitializers:
    original: QDQScaleZpInitializers
    converted: QDQScaleZpInitializers | None
    converted_recv_nodes: set[str] | None

@dataclass
class QDQTensorQuantizedValue:
    original: QuantizedValue
    converted: QuantizedValue | None
    converted_recv_nodes: set[str] | None
    def get_for_consumer(self, consumer_node_name) -> QuantizedValue: ...

class QDQQuantizer(BaseQuantizer):
    tensors_to_quantize: dict[str, QDQTensorQuantInfo]
    bias_to_quantize: dict[str, QDQBiasQuantInfo]
    nodes_to_remove: Incomplete
    op_types_to_exclude_output_quantization: Incomplete
    add_qdq_pair_to_weight: Incomplete
    quantize_bias: Incomplete
    dedicated_qdq_pair: Incomplete
    tensor_to_its_receiving_nodes: dict[str, list[onnx.NodeProto]]
    tensor_to_producing_dq: dict[str, onnx.NodeProto]
    qdq_op_type_per_channel_support_to_axis: Incomplete
    qdq_op_domain: Incomplete
    qdq_keep_removable_activations: Incomplete
    qdq_disable_weight_adjust_for_int32_bias: Incomplete
    quantization_params: Incomplete
    initializer_quant_params: dict[str, QuantizationParams]
    quantized_value_map: Incomplete
    def __init__(self, model, per_channel, reduce_range, weight_qType, activation_qType, tensors_range, nodes_to_quantize, nodes_to_exclude, op_types_to_quantize, extra_options=None) -> None: ...
    def quantize_activation_tensor(self, tensor_name: str): ...
    def quantize_output_same_as_input(self, output_name: str, input_name: str, node_name: str): ...
    def quantize_weight_tensor(self, tensor_name: str): ...
    def quantize_weight_tensor_per_channel(self, tensor_name, axis) -> None: ...
    def quantize_bias_tensor(self, node_name, bias_name, input_name, weight_name, beta: float = 1.0) -> None: ...
    def remove_node(self, node) -> None: ...
    def remove_nodes(self) -> None: ...
    def quantize_model(self): ...
    def try_replacing_upstream_output(self, upstream_output_name, output_name): ...
    def is_tensor_quantized(self, tensor_name: str): ...
    def is_tensor_per_channel(self, tensor_name: str, default_axis: int, op_type: str | None = None) -> tuple[bool, int | None]: ...
    def quantize_bias_static(self, bias_name: str, bias_info: QDQBiasQuantInfo) -> str: ...
    def calc_quant_params(self, tensor_data: TensorData, quant_overrides: dict[str, Any]) -> QuantizationParams: ...
    def calc_graph_quant_params(self) -> dict[str, QDQTensorQuantParams]: ...
