# ---------------------------------------------------------------------
# ONNX Data Protocol Buffer Type Stub
# Generated manually from binary descriptor for Pylance/Pyright
# ---------------------------------------------------------------------

from typing import (
    List as _List,
    Optional as _Optional,
    Sequence as _Sequence,
    Iterable as _Iterable,
    Union as _Union,
    overload as _overload,
)

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

# 依存する ONNX 基本プロトタイプをインポート
from .onnx_ml_pb2 import TensorProto, SparseTensorProto

# モジュールレベルの記述子
DESCRIPTOR: _descriptor.FileDescriptor

class SequenceProto(_message.Message):
    """ONNX Sequence Data Type"""
    
    # 内部 Enum 定義
    class DataType(int):
        @classmethod
        def Name(cls, number: int) -> str: ...
        @classmethod
        def Value(cls, name: str) -> int: ...
        @classmethod
        def keys(cls) -> _List[str]: ...
        @classmethod
        def values(cls) -> _List[int]: ...
        @classmethod
        def items(cls) -> _List[tuple[str, int]]: ...

    UNDEFINED: DataType
    TENSOR: DataType
    SPARSE_TENSOR: DataType
    SEQUENCE: DataType
    MAP: DataType
    OPTIONAL: DataType

    NAME_FIELD_NUMBER: int
    ELEM_TYPE_FIELD_NUMBER: int
    TENSOR_VALUES_FIELD_NUMBER: int
    SPARSE_TENSOR_VALUES_FIELD_NUMBER: int
    SEQUENCE_VALUES_FIELD_NUMBER: int
    MAP_VALUES_FIELD_NUMBER: int
    OPTIONAL_VALUES_FIELD_NUMBER: int

    name: str
    elem_type: int
    
    # 繰り返しフィールド（リスト構造）
    tensor_values: _containers.RepeatedCompositeFieldContainer[TensorProto]
    sparse_tensor_values: _containers.RepeatedCompositeFieldContainer[SparseTensorProto]
    sequence_values: _containers.RepeatedCompositeFieldContainer[SequenceProto]
    map_values: _containers.RepeatedCompositeFieldContainer[MapProto]
    optional_values: _containers.RepeatedCompositeFieldContainer[OptionalProto]

    def __init__(
        self,
        name: _Optional[str] = ...,
        elem_type: _Optional[int] = ...,
        tensor_values: _Optional[_Iterable[TensorProto]] = ...,
        sparse_tensor_values: _Optional[_Iterable[SparseTensorProto]] = ...,
        sequence_values: _Optional[_Iterable[SequenceProto]] = ...,
        map_values: _Optional[_Iterable[MapProto]] = ...,
        optional_values: _Optional[_Iterable[OptionalProto]] = ...,
    ) -> None: ...

class MapProto(_message.Message):
    """ONNX Map Data Type"""
    
    NAME_FIELD_NUMBER: int
    KEY_TYPE_FIELD_NUMBER: int
    KEYS_FIELD_NUMBER: int
    STRING_KEYS_FIELD_NUMBER: int
    VALUES_FIELD_NUMBER: int

    name: str
    key_type: int
    keys: _containers.RepeatedScalarFieldContainer[int]
    string_keys: _containers.RepeatedScalarFieldContainer[bytes]
    values: SequenceProto

    def __init__(
        self,
        name: _Optional[str] = ...,
        key_type: _Optional[int] = ...,
        keys: _Optional[_Iterable[int]] = ...,
        string_keys: _Optional[_Iterable[bytes]] = ...,
        values: _Optional[SequenceProto] = ...,
    ) -> None: ...

class OptionalProto(_message.Message):
    """ONNX Optional Data Type"""

    class DataType(int):
        UNDEFINED: int = 0
        TENSOR: int = 1
        SPARSE_TENSOR: int = 2
        SEQUENCE: int = 3
        MAP: int = 4
        OPTIONAL: int = 5

    NAME_FIELD_NUMBER: int
    ELEM_TYPE_FIELD_NUMBER: int
    TENSOR_VALUE_FIELD_NUMBER: int
    SPARSE_TENSOR_VALUE_FIELD_NUMBER: int
    SEQUENCE_VALUE_FIELD_NUMBER: int
    MAP_VALUE_FIELD_NUMBER: int
    OPTIONAL_VALUE_FIELD_NUMBER: int

    name: str
    elem_type: int
    tensor_value: TensorProto
    sparse_tensor_value: SparseTensorProto
    sequence_value: SequenceProto
    map_value: MapProto
    optional_value: OptionalProto

    def __init__(
        self,
        name: _Optional[str] = ...,
        elem_type: _Optional[int] = ...,
        tensor_value: _Optional[TensorProto] = ...,
        sparse_tensor_value: _Optional[SparseTensorProto] = ...,
        sequence_value: _Optional[SequenceProto] = ...,
        map_value: _Optional[MapProto] = ...,
        optional_value: _Optional[OptionalProto] = ...,
    ) -> None: ...