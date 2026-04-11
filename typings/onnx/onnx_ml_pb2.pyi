# ---------------------------------------------------------------------
# ONNX Main ML Protocol Buffer Type Stub (Full Specification)
# Generated for Pylance / Pyright
# ---------------------------------------------------------------------

from typing import (
    List as _List,
    Optional as _Optional,
    Sequence as _Sequence,
    Iterable as _Iterable,
    Any as _Any,
    Union as _Union,
    Dict as _Dict,
)

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

# --- Enums ---

class Version(int):
    _START_VERSION: int = 0
    IR_VERSION_2017_10_10: int = 1
    IR_VERSION_2017_10_30: int = 2
    IR_VERSION_2017_11_3: int = 3
    IR_VERSION_2019_1_22: int = 4
    IR_VERSION_2019_3_18: int = 5
    IR_VERSION_2019_9_19: int = 6
    IR_VERSION_2020_5_8: int = 7
    IR_VERSION_2021_7_30: int = 8
    IR_VERSION_2023_5_5: int = 9
    IR_VERSION_2024_3_25: int = 10
    IR_VERSION_2025_05_12: int = 11
    IR_VERSION_2025_08_26: int = 12
    IR_VERSION: int = 13

class OperatorStatus(int):
    EXPERIMENTAL: int = 0
    STABLE: int = 1

# --- Messages ---

class AttributeProto(_message.Message):
    class AttributeType(int):
        UNDEFINED: int = 0
        FLOAT: int = 1
        INT: int = 2
        STRING: int = 3
        TENSOR: int = 4
        GRAPH: int = 5
        SPARSE_TENSOR: int = 11
        TYPE_PROTO: int = 13
        FLOATS: int = 6
        INTS: int = 7
        STRINGS: int = 8
        TENSORS: int = 9
        GRAPHS: int = 10
        SPARSE_TENSORS: int = 12
        TYPE_PROTOS: int = 14

    name: str
    ref_attr_name: str
    doc_string: str
    type: AttributeType
    f: float
    i: int
    s: bytes
    t: TensorProto
    g: GraphProto
    sparse_tensor: SparseTensorProto
    tp: TypeProto
    floats: _containers.RepeatedScalarFieldContainer[float]
    ints: _containers.RepeatedScalarFieldContainer[int]
    strings: _containers.RepeatedScalarFieldContainer[bytes]
    tensors: _containers.RepeatedCompositeFieldContainer[TensorProto]
    graphs: _containers.RepeatedCompositeFieldContainer[GraphProto]
    sparse_tensors: _containers.RepeatedCompositeFieldContainer[SparseTensorProto]
    type_protos: _containers.RepeatedCompositeFieldContainer[TypeProto]

    def __init__(self, name: _Optional[str] = ..., type: _Optional[AttributeType] = ..., f: _Optional[float] = ..., i: _Optional[int] = ..., s: _Optional[bytes] = ..., t: _Optional[TensorProto] = ..., g: _Optional[GraphProto] = ..., sparse_tensor: _Optional[SparseTensorProto] = ..., tp: _Optional[TypeProto] = ..., floats: _Optional[_Iterable[float]] = ..., ints: _Optional[_Iterable[int]] = ..., strings: _Optional[_Iterable[bytes]] = ..., tensors: _Optional[_Iterable[TensorProto]] = ..., graphs: _Optional[_Iterable[GraphProto]] = ..., sparse_tensors: _Optional[_Iterable[SparseTensorProto]] = ..., type_protos: _Optional[_Iterable[TypeProto]] = ...) -> None: ...

class ValueInfoProto(_message.Message):
    name: str
    type: TypeProto
    doc_string: str
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    def __init__(self, name: _Optional[str] = ..., type: _Optional[TypeProto] = ..., doc_string: _Optional[str] = ..., metadata_props: _Optional[_Iterable[StringStringEntryProto]] = ...) -> None: ...

class NodeProto(_message.Message):
    input: _containers.RepeatedScalarFieldContainer[str]
    output: _containers.RepeatedScalarFieldContainer[str]
    name: str
    op_type: str
    domain: str
    overload: str
    attribute: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    doc_string: str
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    device_configurations: _containers.RepeatedCompositeFieldContainer[NodeDeviceConfigurationProto]
    def __init__(self, input: _Optional[_Iterable[str]] = ..., output: _Optional[_Iterable[str]] = ..., name: _Optional[str] = ..., op_type: _Optional[str] = ..., domain: _Optional[str] = ..., attribute: _Optional[_Iterable[AttributeProto]] = ..., **kwargs) -> None: ...

class ModelProto(_message.Message):
    ir_version: int
    opset_import: _containers.RepeatedCompositeFieldContainer[OperatorSetIdProto]
    producer_name: str
    producer_version: str
    domain: str
    model_version: int
    doc_string: str
    graph: GraphProto
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    training_info: _containers.RepeatedCompositeFieldContainer[TrainingInfoProto]
    functions: _containers.RepeatedCompositeFieldContainer[FunctionProto]
    configuration: _containers.RepeatedCompositeFieldContainer[DeviceConfigurationProto]
    def __init__(self, ir_version: _Optional[int] = ..., graph: _Optional[GraphProto] = ..., opset_import: _Optional[_Iterable[OperatorSetIdProto]] = ..., producer_name: _Optional[str] = ..., producer_version: _Optional[str] = ..., domain: _Optional[str] = ..., model_version: _Optional[int] = ..., doc_string: _Optional[str] = ..., metadata_props: _Optional[_Iterable[StringStringEntryProto]] = ...) -> None: ...

class GraphProto(_message.Message):
    node: _containers.RepeatedCompositeFieldContainer[NodeProto]
    name: str
    initializer: _containers.RepeatedCompositeFieldContainer[TensorProto]
    sparse_initializer: _containers.RepeatedCompositeFieldContainer[SparseTensorProto]
    doc_string: str
    input: _containers.RepeatedCompositeFieldContainer[ValueInfoProto]
    output: _containers.RepeatedCompositeFieldContainer[ValueInfoProto]
    value_info: _containers.RepeatedCompositeFieldContainer[ValueInfoProto]
    quantization_annotation: _containers.RepeatedCompositeFieldContainer[TensorAnnotation]
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    def __init__(self, node: _Optional[_Iterable[NodeProto]] = ..., name: _Optional[str] = ..., initializer: _Optional[_Iterable[TensorProto]] = ..., input: _Optional[_Iterable[ValueInfoProto]] = ..., output: _Optional[_Iterable[ValueInfoProto]] = ...) -> None: ...

class TensorProto(_message.Message):
    class DataType(int):
        UNDEFINED: int = 0
        FLOAT: int = 1
        UINT8: int = 2
        INT8: int = 3
        UINT16: int = 4
        INT16: int = 5
        INT32: int = 6
        INT64: int = 7
        STRING: int = 8
        BOOL: int = 9
        FLOAT16: int = 10
        DOUBLE: int = 11
        UINT32: int = 12
        UINT64: int = 13
        COMPLEX64: int = 14
        COMPLEX128: int = 15
        BFLOAT16: int = 16
        FLOAT8E4M3FN: int = 17
        FLOAT8E4M3FNUZ: int = 18
        FLOAT8E5M2: int = 19
        FLOAT8E5M2FNUZ: int = 20
        UINT4: int = 21
        INT4: int = 22
        FLOAT4E2M1: int = 23
        FLOAT8E8M0: int = 24
        UINT2: int = 25
        INT2: int = 26

    class DataLocation(int):
        DEFAULT: int = 0
        EXTERNAL: int = 1

    class Segment(_message.Message):
        begin: int
        end: int

    dims: _containers.RepeatedScalarFieldContainer[int]
    data_type: int
    segment: Segment
    float_data: _containers.RepeatedScalarFieldContainer[float]
    int32_data: _containers.RepeatedScalarFieldContainer[int]
    string_data: _containers.RepeatedScalarFieldContainer[bytes]
    int64_data: _containers.RepeatedScalarFieldContainer[int]
    name: str
    doc_string: str
    raw_data: bytes
    external_data: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    data_location: DataLocation
    double_data: _containers.RepeatedScalarFieldContainer[float]
    uint64_data: _containers.RepeatedScalarFieldContainer[int]
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]

    def __init__(self, dims: _Optional[_Iterable[int]] = ..., data_type: _Optional[int] = ..., name: _Optional[str] = ..., raw_data: _Optional[bytes] = ..., **kwargs) -> None: ...

class TypeProto(_message.Message):
    class Tensor(_message.Message):
        elem_type: int
        shape: TensorShapeProto
    class Sequence(_message.Message):
        elem_type: TypeProto
    class Map(_message.Message):
        key_type: int
        value_type: TypeProto
    class Optional(_message.Message):
        elem_type: TypeProto
    class SparseTensor(_message.Message):
        elem_type: int
        shape: TensorShapeProto
    class Opaque(_message.Message):
        domain: str
        name: str

    tensor_type: Tensor
    sequence_type: Sequence
    map_type: Map
    optional_type: Optional
    sparse_tensor_type: SparseTensor
    opaque_type: Opaque
    denotation: str

class TensorShapeProto(_message.Message):
    class Dimension(_message.Message):
        dim_value: int
        dim_param: str
        denotation: str
    dim: _containers.RepeatedCompositeFieldContainer[Dimension]

class FunctionProto(_message.Message):
    name: str
    input: _containers.RepeatedScalarFieldContainer[str]
    output: _containers.RepeatedScalarFieldContainer[str]
    attribute: _containers.RepeatedScalarFieldContainer[str]
    attribute_proto: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    node: _containers.RepeatedCompositeFieldContainer[NodeProto]
    doc_string: str
    opset_import: _containers.RepeatedCompositeFieldContainer[OperatorSetIdProto]
    domain: str
    overload: str

class SparseTensorProto(_message.Message):
    values: TensorProto
    indices: TensorProto
    dims: _containers.RepeatedScalarFieldContainer[int]

class OperatorSetIdProto(_message.Message):
    domain: str
    version: int

class StringStringEntryProto(_message.Message):
    key: str
    value: str

class TensorAnnotation(_message.Message):
    tensor_name: str
    quant_parameter_tensor_names: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]

# その他、バイナリに含まれる内部補助メッセージ群
class NodeDeviceConfigurationProto(_message.Message): ...
class IntIntListEntryProto(_message.Message): ...
class TrainingInfoProto(_message.Message): ...
class DeviceConfigurationProto(_message.Message): ...
class ShardingSpecProto(_message.Message): ...
class ShardedDimProto(_message.Message): ...
class SimpleShardedDimProto(_message.Message): ...