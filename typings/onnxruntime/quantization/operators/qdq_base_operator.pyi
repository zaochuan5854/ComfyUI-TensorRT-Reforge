from ..quant_utils import QuantizedValue as QuantizedValue, QuantizedValueType as QuantizedValueType, attribute_to_kwarg as attribute_to_kwarg, quantize_nparray as quantize_nparray
from .base_operator import QuantOperatorBase as QuantOperatorBase
from _typeshed import Incomplete

class QDQOperatorBase:
    quantizer: Incomplete
    node: Incomplete
    disable_qdq_for_node_output: Incomplete
    def __init__(self, onnx_quantizer, onnx_node) -> None: ...
    def quantize(self) -> None: ...
