from ..quant_utils import QuantizedValue as QuantizedValue, QuantizedValueType as QuantizedValueType, attribute_to_kwarg as attribute_to_kwarg
from .base_operator import QuantOperatorBase as QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase as QDQOperatorBase

class QSplit(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node) -> None: ...
    def quantize(self): ...

class QDQSplit(QDQOperatorBase):
    def quantize(self) -> None: ...
