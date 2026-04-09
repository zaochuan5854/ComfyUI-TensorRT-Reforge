import numpy as np
import numpy.typing as npt
from .calibrate import CalibrationDataReader as CalibrationDataReader
from .neural_compressor import gptq_quantize as gptq_quantize, rtn_quantize as rtn_quantize
from .onnx_model import ONNXModel as ONNXModel
from .quant_utils import QuantFormat as QuantFormat, attribute_to_kwarg as attribute_to_kwarg
from _typeshed import Incomplete
from onnx.onnx_pb import GraphProto as GraphProto, ModelProto as ModelProto, NodeProto as NodeProto, TensorProto
from onnxruntime.capi._pybind_state import quantize_matmul_2bits as quantize_matmul_2bits, quantize_matmul_4bits as quantize_matmul_4bits, quantize_matmul_8bits as quantize_matmul_8bits, quantize_qdq_matmul_4bits as quantize_qdq_matmul_4bits

logger: Incomplete

class WeightOnlyQuantConfig:
    algorithm: Incomplete
    quant_format: Incomplete
    op_types_to_quantize: Incomplete
    quant_axes: Incomplete
    customized_weight_config: Incomplete
    def __init__(self, algorithm: str, quant_format: QuantFormat, op_types_to_quantize: tuple[str, ...] | None = None, quant_axes: tuple[tuple[str, int], ...] | None = None, customized_weight_config: dict | None = None) -> None: ...

class RTNWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    ratios: Incomplete
    def __init__(self, ratios=None, quant_format=..., op_types_to_quantize: tuple[str, ...] | None = None, customized_weight_config: dict | None = None) -> None: ...

class KQuantWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    ratios: Incomplete
    def __init__(self, ratios=None, quant_format=..., op_types_to_quantize: tuple[str, ...] | None = None, customized_weight_config: dict | None = None) -> None: ...

class GPTQWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    calibration_data_reader: Incomplete
    percdamp: Incomplete
    block_size: Incomplete
    actorder: Incomplete
    mse: Incomplete
    perchannel: Incomplete
    def __init__(self, calibration_data_reader: CalibrationDataReader | None = None, percdamp: float = 0.01, block_size: int = 128, actorder: bool = False, mse: bool = False, perchannel: bool = True, quant_format=..., op_types_to_quantize: tuple[str, ...] | None = None) -> None: ...

class HQQWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    block_size: Incomplete
    bits: Incomplete
    axis: Incomplete
    def __init__(self, block_size: int = 128, bits: int = 4, axis: int = 1, quant_format=..., op_types_to_quantize: tuple[str, ...] | None = None, quant_axes: tuple[tuple[str, int], ...] | None = None) -> None: ...

class DefaultWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    block_size: Incomplete
    is_symmetric: Incomplete
    bits: Incomplete
    accuracy_level: Incomplete
    channel_wised_quantize: Incomplete
    def __init__(self, block_size: int = 128, is_symmetric: bool = False, accuracy_level: int | None = None, quant_format=..., op_types_to_quantize: tuple[str, ...] | None = None, quant_axes: tuple[tuple[str, int], ...] | None = None, bits: int = 4, channel_wised_quantize: bool = False) -> None: ...

class NVAWQWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    torch: Incomplete
    DataLoader: Incomplete
    load_dataset: Incomplete
    AutoConfig: Incomplete
    AutoTokenizer: Incomplete
    calibration_data_reader: Incomplete
    calibration_method: Incomplete
    def __init__(self, tokenizer_dir, dataset_name: str = 'cnn', cache_dir: str = './cache', calibration_method: str = 'awq_lite') -> None: ...
    def make_model_input(self, config, input_ids_arg, attention_mask_arg, add_past_kv_inputs, device, use_fp16, use_buffer_share, add_position_ids): ...
    def get_calib_inputs(self, dataset_name, model_name, cache_dir, calib_size, batch_size, block_size, device, use_fp16, use_buffer_share, add_past_kv_inputs, max_calib_rows_to_load, add_position_ids): ...

def is_divisible(val1, val2): ...

class HQQWeightOnlyQuantizer:
    config: Incomplete
    def __init__(self, config: HQQWeightOnlyQuantConfig) -> None: ...
    @staticmethod
    def optimize_weights(tensor, scale, zero, min_max: list[int], axis: int = 0, opt_params: dict | None = None, verbose: bool = False): ...
    @staticmethod
    def pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits) -> None: ...
    def quantize_internal(self, tensor, bits: int = 4, channel_wise: bool = True, group_size: int = 64, optimize: bool = True, round_zero: bool = True, axis: int = 1): ...
    def quantize(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]: ...

def get_initializer(name, graph_path: list[GraphProto]) -> tuple[TensorProto, GraphProto]: ...
def transpose_packed_int4_matrix(packed, rows, cols): ...

class DefaultWeightOnlyQuantizer:
    config: Incomplete
    def __init__(self, config: DefaultWeightOnlyQuantConfig) -> None: ...
    def qbits_block_quant(self, fp32weight: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def quantize_matmul(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]: ...
    @staticmethod
    def quant_slice_symmetric(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    @staticmethod
    def quant_slice_asymmetric(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    @staticmethod
    def pack_int8_to_int4(data: np.ndarray) -> np.ndarray: ...
    @staticmethod
    def quantize_ndarray(data: np.ndarray, quantize_axis: int, block_size: int, is_symmetric: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]: ...
    def quantize_gather(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]: ...
    def quantize(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]: ...

class NVAWQWeightOnlyQuantizer:
    config: Incomplete
    def __init__(self, config: NVAWQWeightOnlyQuantConfig) -> None: ...
    def quantize_awq(self, model: ModelProto | str) -> ModelProto: ...

class MatMulNBitsQuantizer:
    model: Incomplete
    model_path: Incomplete
    bits: Incomplete
    block_size: Incomplete
    is_symmetric: Incomplete
    accuracy_level: Incomplete
    nodes_to_exclude: Incomplete
    nodes_to_include: Incomplete
    node_quantizer: Incomplete
    algo_config: Incomplete
    def __init__(self, model: ModelProto | str, bits: int = 4, block_size: int = 128, is_symmetric: bool = False, accuracy_level: int | None = None, nodes_to_exclude=None, nodes_to_include: list[str] | None = None, quant_format=..., op_types_to_quantize: tuple[str, ...] | None = None, quant_axes: tuple[tuple[str, int], ...] | None = None, channel_wised_quantize: bool = False, algo_config: WeightOnlyQuantConfig | None = None) -> None: ...
    def int4_quant_algo(self) -> None: ...
    def process(self) -> None: ...

def ort_convert_str_to_bool(value): ...
def parse_key_value_pair(s): ...
def parse_args(): ...
