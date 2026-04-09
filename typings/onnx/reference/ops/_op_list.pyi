from onnx import NodeProto, TypeProto
from onnx.reference.ops.op_abs import Abs as Abs
from onnx.reference.ops.op_acos import Acos as Acos
from onnx.reference.ops.op_acosh import Acosh as Acosh
from onnx.reference.ops.op_add import Add as Add
from onnx.reference.ops.op_affine_grid import AffineGrid as AffineGrid
from onnx.reference.ops.op_and import And as And
from onnx.reference.ops.op_argmax import ArgMax_1 as ArgMax_1, ArgMax_12 as ArgMax_12
from onnx.reference.ops.op_argmin import ArgMin_1 as ArgMin_1, ArgMin_12 as ArgMin_12
from onnx.reference.ops.op_asin import Asin as Asin
from onnx.reference.ops.op_asinh import Asinh as Asinh
from onnx.reference.ops.op_atan import Atan as Atan
from onnx.reference.ops.op_atanh import Atanh as Atanh
from onnx.reference.ops.op_attention import Attention as Attention
from onnx.reference.ops.op_attribute_has_value import AttributeHasValue as AttributeHasValue
from onnx.reference.ops.op_average_pool import AveragePool_1 as AveragePool_1, AveragePool_11 as AveragePool_11, AveragePool_19 as AveragePool_19, AveragePool_7 as AveragePool_7
from onnx.reference.ops.op_batch_normalization import BatchNormalization_14 as BatchNormalization_14, BatchNormalization_6 as BatchNormalization_6, BatchNormalization_9 as BatchNormalization_9
from onnx.reference.ops.op_bernoulli import Bernoulli as Bernoulli
from onnx.reference.ops.op_bitcast import BitCast as BitCast
from onnx.reference.ops.op_bitshift import BitShift as BitShift
from onnx.reference.ops.op_bitwise_and import BitwiseAnd as BitwiseAnd
from onnx.reference.ops.op_bitwise_not import BitwiseNot as BitwiseNot
from onnx.reference.ops.op_bitwise_or import BitwiseOr as BitwiseOr
from onnx.reference.ops.op_bitwise_xor import BitwiseXor as BitwiseXor
from onnx.reference.ops.op_blackman_window import BlackmanWindow as BlackmanWindow
from onnx.reference.ops.op_cast import Cast_1 as Cast_1, Cast_19 as Cast_19, Cast_24 as Cast_24
from onnx.reference.ops.op_cast_like import CastLike_15 as CastLike_15, CastLike_19 as CastLike_19
from onnx.reference.ops.op_ceil import Ceil as Ceil
from onnx.reference.ops.op_celu import Celu as Celu
from onnx.reference.ops.op_center_crop_pad import CenterCropPad as CenterCropPad
from onnx.reference.ops.op_clip import Clip_11 as Clip_11, Clip_6 as Clip_6
from onnx.reference.ops.op_col2im import Col2Im as Col2Im
from onnx.reference.ops.op_compress import Compress as Compress
from onnx.reference.ops.op_concat import Concat as Concat
from onnx.reference.ops.op_concat_from_sequence import ConcatFromSequence as ConcatFromSequence
from onnx.reference.ops.op_constant import Constant_1 as Constant_1, Constant_11 as Constant_11, Constant_12 as Constant_12, Constant_9 as Constant_9
from onnx.reference.ops.op_constant_of_shape import ConstantOfShape as ConstantOfShape
from onnx.reference.ops.op_conv import Conv as Conv
from onnx.reference.ops.op_conv_integer import ConvInteger as ConvInteger
from onnx.reference.ops.op_conv_transpose import ConvTranspose as ConvTranspose
from onnx.reference.ops.op_cos import Cos as Cos
from onnx.reference.ops.op_cosh import Cosh as Cosh
from onnx.reference.ops.op_cum_prod import CumProd as CumProd
from onnx.reference.ops.op_cum_sum import CumSum as CumSum
from onnx.reference.ops.op_deform_conv import DeformConv as DeformConv
from onnx.reference.ops.op_depth_to_space import DepthToSpace as DepthToSpace
from onnx.reference.ops.op_dequantize_linear import DequantizeLinear_19 as DequantizeLinear_19, DequantizeLinear_21 as DequantizeLinear_21
from onnx.reference.ops.op_det import Det as Det
from onnx.reference.ops.op_dft import DFT_17 as DFT_17, DFT_20 as DFT_20
from onnx.reference.ops.op_div import Div as Div
from onnx.reference.ops.op_dropout import Dropout_12 as Dropout_12, Dropout_7 as Dropout_7
from onnx.reference.ops.op_dynamic_quantize_linear import DynamicQuantizeLinear as DynamicQuantizeLinear
from onnx.reference.ops.op_einsum import Einsum as Einsum
from onnx.reference.ops.op_elu import Elu as Elu
from onnx.reference.ops.op_equal import Equal as Equal
from onnx.reference.ops.op_erf import Erf as Erf
from onnx.reference.ops.op_exp import Exp as Exp
from onnx.reference.ops.op_expand import Expand as Expand
from onnx.reference.ops.op_eyelike import EyeLike as EyeLike
from onnx.reference.ops.op_flatten import Flatten as Flatten
from onnx.reference.ops.op_floor import Floor as Floor
from onnx.reference.ops.op_gather import Gather as Gather
from onnx.reference.ops.op_gather_elements import GatherElements as GatherElements
from onnx.reference.ops.op_gathernd import GatherND as GatherND
from onnx.reference.ops.op_gemm import Gemm_6 as Gemm_6, Gemm_7 as Gemm_7
from onnx.reference.ops.op_global_average_pool import GlobalAveragePool as GlobalAveragePool
from onnx.reference.ops.op_global_max_pool import GlobalMaxPool as GlobalMaxPool
from onnx.reference.ops.op_greater import Greater as Greater
from onnx.reference.ops.op_greater_or_equal import GreaterOrEqual as GreaterOrEqual
from onnx.reference.ops.op_grid_sample import GridSample as GridSample
from onnx.reference.ops.op_gru import GRU as GRU
from onnx.reference.ops.op_hamming_window import HammingWindow as HammingWindow
from onnx.reference.ops.op_hann_window import HannWindow as HannWindow
from onnx.reference.ops.op_hard_sigmoid import HardSigmoid as HardSigmoid
from onnx.reference.ops.op_hardmax import Hardmax as Hardmax
from onnx.reference.ops.op_identity import Identity as Identity
from onnx.reference.ops.op_if import If as If
from onnx.reference.ops.op_image_decoder import ImageDecoder as ImageDecoder
from onnx.reference.ops.op_instance_normalization import InstanceNormalization as InstanceNormalization
from onnx.reference.ops.op_isinf import IsInf as IsInf
from onnx.reference.ops.op_isnan import IsNaN as IsNaN
from onnx.reference.ops.op_layer_normalization import LayerNormalization as LayerNormalization
from onnx.reference.ops.op_leaky_relu import LeakyRelu as LeakyRelu
from onnx.reference.ops.op_less import Less as Less
from onnx.reference.ops.op_less_or_equal import LessOrEqual as LessOrEqual
from onnx.reference.ops.op_log import Log as Log
from onnx.reference.ops.op_log_softmax import LogSoftmax as LogSoftmax
from onnx.reference.ops.op_loop import Loop as Loop
from onnx.reference.ops.op_lp_normalization import LpNormalization as LpNormalization
from onnx.reference.ops.op_lp_pool import LpPool as LpPool
from onnx.reference.ops.op_lrn import LRN as LRN
from onnx.reference.ops.op_lstm import LSTM as LSTM
from onnx.reference.ops.op_matmul import MatMul as MatMul
from onnx.reference.ops.op_matmul_integer import MatMulInteger as MatMulInteger
from onnx.reference.ops.op_max import Max as Max
from onnx.reference.ops.op_max_pool import MaxPool as MaxPool
from onnx.reference.ops.op_max_unpool import MaxUnpool as MaxUnpool
from onnx.reference.ops.op_mean import Mean as Mean
from onnx.reference.ops.op_mel_weight_matrix import MelWeightMatrix as MelWeightMatrix
from onnx.reference.ops.op_min import Min as Min
from onnx.reference.ops.op_mod import Mod as Mod
from onnx.reference.ops.op_mul import Mul as Mul
from onnx.reference.ops.op_neg import Neg as Neg
from onnx.reference.ops.op_negative_log_likelihood_loss import NegativeLogLikelihoodLoss as NegativeLogLikelihoodLoss
from onnx.reference.ops.op_non_max_suppression import NonMaxSuppression as NonMaxSuppression
from onnx.reference.ops.op_non_zero import NonZero as NonZero
from onnx.reference.ops.op_not import Not as Not
from onnx.reference.ops.op_one_hot import OneHot as OneHot
from onnx.reference.ops.op_optional import Optional as Optional
from onnx.reference.ops.op_optional_get_element import OptionalGetElement as OptionalGetElement
from onnx.reference.ops.op_optional_has_element import OptionalHasElement as OptionalHasElement
from onnx.reference.ops.op_or import Or as Or
from onnx.reference.ops.op_pad import Pad_1 as Pad_1, Pad_11 as Pad_11, Pad_18 as Pad_18, Pad_2 as Pad_2
from onnx.reference.ops.op_pow import Pow as Pow
from onnx.reference.ops.op_prelu import PRelu as PRelu
from onnx.reference.ops.op_qlinear_conv import QLinearConv as QLinearConv
from onnx.reference.ops.op_qlinear_matmul import QLinearMatMul as QLinearMatMul
from onnx.reference.ops.op_quantize_linear import QuantizeLinear_10 as QuantizeLinear_10, QuantizeLinear_19 as QuantizeLinear_19, QuantizeLinear_21 as QuantizeLinear_21
from onnx.reference.ops.op_random_normal import RandomNormal as RandomNormal
from onnx.reference.ops.op_random_normal_like import RandomNormalLike as RandomNormalLike
from onnx.reference.ops.op_random_uniform import RandomUniform as RandomUniform
from onnx.reference.ops.op_random_uniform_like import RandomUniformLike as RandomUniformLike
from onnx.reference.ops.op_range import Range as Range
from onnx.reference.ops.op_reciprocal import Reciprocal as Reciprocal
from onnx.reference.ops.op_reduce_l1 import ReduceL1_1 as ReduceL1_1, ReduceL1_18 as ReduceL1_18
from onnx.reference.ops.op_reduce_l2 import ReduceL2_1 as ReduceL2_1, ReduceL2_18 as ReduceL2_18
from onnx.reference.ops.op_reduce_log_sum import ReduceLogSum_1 as ReduceLogSum_1, ReduceLogSum_18 as ReduceLogSum_18
from onnx.reference.ops.op_reduce_log_sum_exp import ReduceLogSumExp_1 as ReduceLogSumExp_1, ReduceLogSumExp_18 as ReduceLogSumExp_18
from onnx.reference.ops.op_reduce_max import ReduceMax_1 as ReduceMax_1, ReduceMax_18 as ReduceMax_18
from onnx.reference.ops.op_reduce_mean import ReduceMean_1 as ReduceMean_1, ReduceMean_18 as ReduceMean_18
from onnx.reference.ops.op_reduce_min import ReduceMin_1 as ReduceMin_1, ReduceMin_18 as ReduceMin_18
from onnx.reference.ops.op_reduce_prod import ReduceProd_1 as ReduceProd_1, ReduceProd_18 as ReduceProd_18
from onnx.reference.ops.op_reduce_sum import ReduceSum_1 as ReduceSum_1, ReduceSum_13 as ReduceSum_13
from onnx.reference.ops.op_reduce_sum_square import ReduceSumSquare_1 as ReduceSumSquare_1, ReduceSumSquare_18 as ReduceSumSquare_18
from onnx.reference.ops.op_regex_full_match import RegexFullMatch as RegexFullMatch
from onnx.reference.ops.op_relu import Relu as Relu
from onnx.reference.ops.op_reshape import Reshape_14 as Reshape_14, Reshape_5 as Reshape_5
from onnx.reference.ops.op_resize import Resize as Resize
from onnx.reference.ops.op_reverse_sequence import ReverseSequence as ReverseSequence
from onnx.reference.ops.op_rms_normalization import RMSNormalization as RMSNormalization
from onnx.reference.ops.op_rnn import RNN_14 as RNN_14, RNN_7 as RNN_7
from onnx.reference.ops.op_roi_align import RoiAlign as RoiAlign
from onnx.reference.ops.op_rotary_embedding import RotaryEmbedding as RotaryEmbedding
from onnx.reference.ops.op_round import Round as Round
from onnx.reference.ops.op_scan import Scan as Scan
from onnx.reference.ops.op_scatter_elements import ScatterElements as ScatterElements
from onnx.reference.ops.op_scatternd import ScatterND as ScatterND
from onnx.reference.ops.op_selu import Selu as Selu
from onnx.reference.ops.op_sequence_at import SequenceAt as SequenceAt
from onnx.reference.ops.op_sequence_construct import SequenceConstruct as SequenceConstruct
from onnx.reference.ops.op_sequence_empty import SequenceEmpty as SequenceEmpty
from onnx.reference.ops.op_sequence_erase import SequenceErase as SequenceErase
from onnx.reference.ops.op_sequence_insert import SequenceInsert as SequenceInsert
from onnx.reference.ops.op_sequence_length import SequenceLength as SequenceLength
from onnx.reference.ops.op_sequence_map import SequenceMap as SequenceMap
from onnx.reference.ops.op_shape import Shape_1 as Shape_1, Shape_15 as Shape_15
from onnx.reference.ops.op_shrink import Shrink as Shrink
from onnx.reference.ops.op_sigmoid import Sigmoid as Sigmoid
from onnx.reference.ops.op_sign import Sign as Sign
from onnx.reference.ops.op_sin import Sin as Sin
from onnx.reference.ops.op_sinh import Sinh as Sinh
from onnx.reference.ops.op_size import Size as Size
from onnx.reference.ops.op_slice import Slice_1 as Slice_1, Slice_10 as Slice_10
from onnx.reference.ops.op_softmax import Softmax as Softmax
from onnx.reference.ops.op_softmax_cross_entropy_loss import SoftmaxCrossEntropyLoss as SoftmaxCrossEntropyLoss
from onnx.reference.ops.op_softplus import Softplus as Softplus
from onnx.reference.ops.op_softsign import Softsign as Softsign
from onnx.reference.ops.op_space_to_depth import SpaceToDepth as SpaceToDepth
from onnx.reference.ops.op_split import Split_11 as Split_11, Split_13 as Split_13, Split_18 as Split_18, Split_2 as Split_2
from onnx.reference.ops.op_split_to_sequence import SplitToSequence as SplitToSequence
from onnx.reference.ops.op_sqrt import Sqrt as Sqrt
from onnx.reference.ops.op_squeeze import Squeeze_1 as Squeeze_1, Squeeze_11 as Squeeze_11, Squeeze_13 as Squeeze_13
from onnx.reference.ops.op_stft import STFT as STFT
from onnx.reference.ops.op_string_concat import StringConcat as StringConcat
from onnx.reference.ops.op_string_normalizer import StringNormalizer as StringNormalizer
from onnx.reference.ops.op_string_split import StringSplit as StringSplit
from onnx.reference.ops.op_sub import Sub as Sub
from onnx.reference.ops.op_sum import Sum as Sum
from onnx.reference.ops.op_swish import Swish as Swish
from onnx.reference.ops.op_tan import Tan as Tan
from onnx.reference.ops.op_tanh import Tanh as Tanh
from onnx.reference.ops.op_tensor_scatter import TensorScatter as TensorScatter
from onnx.reference.ops.op_tfidf_vectorizer import TfIdfVectorizer as TfIdfVectorizer
from onnx.reference.ops.op_thresholded_relu import ThresholdedRelu as ThresholdedRelu
from onnx.reference.ops.op_tile import Tile as Tile
from onnx.reference.ops.op_topk import TopK_1 as TopK_1, TopK_10 as TopK_10, TopK_11 as TopK_11
from onnx.reference.ops.op_transpose import Transpose as Transpose
from onnx.reference.ops.op_trilu import Trilu as Trilu
from onnx.reference.ops.op_unique import Unique as Unique
from onnx.reference.ops.op_unsqueeze import Unsqueeze_1 as Unsqueeze_1, Unsqueeze_11 as Unsqueeze_11, Unsqueeze_13 as Unsqueeze_13
from onnx.reference.ops.op_upsample import Upsample as Upsample
from onnx.reference.ops.op_where import Where as Where
from onnx.reference.ops.op_xor import Xor as Xor
from typing import Any

__all__ = ['load_op', 'Abs', 'Acos', 'Acosh', 'Add', 'AffineGrid', 'And', 'ArgMax_1', 'ArgMax_12', 'ArgMin_1', 'ArgMin_12', 'Asin', 'Asinh', 'Atan', 'Atanh', 'Attention', 'AttributeHasValue', 'AveragePool_1', 'AveragePool_7', 'AveragePool_11', 'AveragePool_19', 'BatchNormalization_6', 'BatchNormalization_9', 'BatchNormalization_14', 'Bernoulli', 'BitCast', 'BitShift', 'BitwiseAnd', 'BitwiseNot', 'BitwiseOr', 'BitwiseXor', 'BlackmanWindow', 'Cast_1', 'Cast_19', 'Cast_24', 'CastLike_15', 'CastLike_19', 'Ceil', 'Celu', 'CenterCropPad', 'Clip_6', 'Clip_11', 'Col2Im', 'Compress', 'Concat', 'ConcatFromSequence', 'Constant_1', 'Constant_9', 'Constant_11', 'Constant_12', 'ConstantOfShape', 'Conv', 'ConvInteger', 'ConvTranspose', 'Cos', 'Cosh', 'CumProd', 'CumSum', 'DeformConv', 'DepthToSpace', 'DequantizeLinear_19', 'DequantizeLinear_21', 'Det', 'DFT_17', 'DFT_20', 'Div', 'Dropout_7', 'Dropout_12', 'DynamicQuantizeLinear', 'Einsum', 'Elu', 'Equal', 'Erf', 'Exp', 'Expand', 'EyeLike', 'Flatten', 'Floor', 'Gather', 'GatherElements', 'GatherND', 'Gemm_6', 'Gemm_7', 'GlobalAveragePool', 'GlobalMaxPool', 'Greater', 'GreaterOrEqual', 'GridSample', 'GRU', 'HammingWindow', 'HannWindow', 'HardSigmoid', 'Hardmax', 'Identity', 'If', 'ImageDecoder', 'InstanceNormalization', 'IsInf', 'IsNaN', 'LayerNormalization', 'LeakyRelu', 'Less', 'LessOrEqual', 'Log', 'LogSoftmax', 'Loop', 'LpNormalization', 'LpPool', 'LRN', 'LSTM', 'MatMul', 'MatMulInteger', 'Max', 'MaxPool', 'MaxUnpool', 'Mean', 'MelWeightMatrix', 'Min', 'Mod', 'Mul', 'Neg', 'NegativeLogLikelihoodLoss', 'NonMaxSuppression', 'NonZero', 'Not', 'OneHot', 'Optional', 'OptionalGetElement', 'OptionalHasElement', 'Or', 'Pad_1', 'Pad_2', 'Pad_11', 'Pad_18', 'Pow', 'PRelu', 'QLinearConv', 'QLinearMatMul', 'QuantizeLinear_10', 'QuantizeLinear_19', 'QuantizeLinear_21', 'RandomNormal', 'RandomNormalLike', 'RandomUniform', 'RandomUniformLike', 'Range', 'Reciprocal', 'ReduceL1_1', 'ReduceL1_18', 'ReduceL2_1', 'ReduceL2_18', 'ReduceLogSum_1', 'ReduceLogSum_18', 'ReduceLogSumExp_1', 'ReduceLogSumExp_18', 'ReduceMax_1', 'ReduceMax_18', 'ReduceMean_1', 'ReduceMean_18', 'ReduceMin_1', 'ReduceMin_18', 'ReduceProd_1', 'ReduceProd_18', 'ReduceSum_1', 'ReduceSum_13', 'ReduceSumSquare_1', 'ReduceSumSquare_18', 'RegexFullMatch', 'Relu', 'Reshape_5', 'Reshape_14', 'Resize', 'ReverseSequence', 'RMSNormalization', 'RNN_7', 'RNN_14', 'RoiAlign', 'RotaryEmbedding', 'Round', 'Scan', 'ScatterElements', 'ScatterND', 'Selu', 'SequenceAt', 'SequenceConstruct', 'SequenceEmpty', 'SequenceErase', 'SequenceInsert', 'SequenceLength', 'SequenceMap', 'Shape_1', 'Shape_15', 'Shrink', 'Sigmoid', 'Sign', 'Sin', 'Sinh', 'Size', 'Slice_1', 'Slice_10', 'Softmax', 'SoftmaxCrossEntropyLoss', 'Softplus', 'Softsign', 'Swish', 'SpaceToDepth', 'Split_2', 'Split_11', 'Split_13', 'Split_18', 'SplitToSequence', 'Sqrt', 'Squeeze_1', 'Squeeze_11', 'Squeeze_13', 'STFT', 'StringConcat', 'StringNormalizer', 'StringSplit', 'Sub', 'Sum', 'Tan', 'Tanh', 'TensorScatter', 'TfIdfVectorizer', 'ThresholdedRelu', 'Tile', 'TopK_1', 'TopK_10', 'TopK_11', 'Transpose', 'Trilu', 'Unique', 'Unsqueeze_1', 'Unsqueeze_11', 'Unsqueeze_13', 'Upsample', 'Where', 'Xor']

def load_op(domain: str, op_type: str, version: None | int = None, custom: Any = None, node: None | NodeProto = None, input_types: None | list[TypeProto] = None, expand: bool = False, evaluator_cls: type | None = None) -> Any: ...
