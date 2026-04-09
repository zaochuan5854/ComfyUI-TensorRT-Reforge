from .operators.activation import QDQRemovableActivation as QDQRemovableActivation, QLinearActivation as QLinearActivation
from .operators.argmax import QArgMax as QArgMax
from .operators.attention import AttentionQuant as AttentionQuant
from .operators.base_operator import QuantOperatorBase as QuantOperatorBase
from .operators.binary_op import QLinearBinaryOp as QLinearBinaryOp
from .operators.concat import QLinearConcat as QLinearConcat
from .operators.conv import ConvInteger as ConvInteger, QDQConv as QDQConv, QLinearConv as QLinearConv
from .operators.direct_q8 import Direct8BitOp as Direct8BitOp, QDQDirect8BitOp as QDQDirect8BitOp
from .operators.embed_layernorm import EmbedLayerNormalizationQuant as EmbedLayerNormalizationQuant
from .operators.gather import GatherQuant as GatherQuant, QDQGather as QDQGather
from .operators.gavgpool import QGlobalAveragePool as QGlobalAveragePool
from .operators.gemm import QDQGemm as QDQGemm, QLinearGemm as QLinearGemm
from .operators.lstm import LSTMQuant as LSTMQuant
from .operators.matmul import MatMulInteger as MatMulInteger, QDQMatMul as QDQMatMul, QLinearMatMul as QLinearMatMul
from .operators.maxpool import QDQMaxPool as QDQMaxPool, QMaxPool as QMaxPool
from .operators.norm import QDQNormalization as QDQNormalization
from .operators.pad import QDQPad as QDQPad, QPad as QPad
from .operators.pooling import QLinearPool as QLinearPool
from .operators.qdq_base_operator import QDQOperatorBase as QDQOperatorBase
from .operators.resize import QDQResize as QDQResize, QResize as QResize
from .operators.softmax import QLinearSoftmax as QLinearSoftmax
from .operators.split import QDQSplit as QDQSplit, QSplit as QSplit
from .operators.where import QDQWhere as QDQWhere, QLinearWhere as QLinearWhere
from .quant_utils import QuantizationMode as QuantizationMode
from _typeshed import Incomplete

CommonOpsRegistry: Incomplete
IntegerOpsRegistry: Incomplete
QLinearOpsRegistry: Incomplete
QDQRegistry: Incomplete

def CreateDefaultOpQuantizer(onnx_quantizer, node): ...
def CreateOpQuantizer(onnx_quantizer, node): ...
def CreateQDQQuantizer(onnx_quantizer, node): ...
