import numpy as np
from onnx.onnx_pb import TensorProto as TensorProto
from typing import NamedTuple

class TensorDtypeMap(NamedTuple):
    np_dtype: np.dtype
    storage_dtype: int
    name: str

TENSOR_TYPE_MAP: dict[int, TensorDtypeMap]
