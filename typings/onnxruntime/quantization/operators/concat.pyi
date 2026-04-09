from ..quant_utils import QuantizedValue as QuantizedValue, QuantizedValueType as QuantizedValueType, TENSOR_NAME_QUANT_SUFFIX as TENSOR_NAME_QUANT_SUFFIX, attribute_to_kwarg as attribute_to_kwarg, ms_domain as ms_domain
from .base_operator import QuantOperatorBase as QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase as QDQOperatorBase

class QLinearConcat(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node) -> None: ...
    def quantize(self): ...
