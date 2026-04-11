# ---------------------------------------------------------------------
# ONNX Operators ML Protocol Buffer Type Stub
# Generated from binary descriptor for Pylance/Pyright
# ---------------------------------------------------------------------

from typing import (
    List as _List,
    Optional as _Optional,
    Iterable as _Iterable,
    Union as _Union,
)

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

# 依存関係にある onnx_ml_pb2 から型をインポート
from .onnx_ml_pb2 import OperatorStatus, FunctionProto

DESCRIPTOR: _descriptor.FileDescriptor

class OperatorProto(_message.Message):
    """ONNX Operator Definition"""
    OP_TYPE_FIELD_NUMBER: int
    SINCE_VERSION_FIELD_NUMBER: int
    STATUS_FIELD_NUMBER: int
    DOC_STRING_FIELD_NUMBER: int

    op_type: str
    since_version: int
    status: OperatorStatus
    doc_string: str

    def __init__(
        self,
        op_type: _Optional[str] = ...,
        since_version: _Optional[int] = ...,
        status: _Optional[OperatorStatus] = ...,
        doc_string: _Optional[str] = ...,
    ) -> None: ...

class OperatorSetProto(_message.Message):
    """ONNX Operator Set Definition"""
    MAGIC_FIELD_NUMBER: int
    IR_VERSION_FIELD_NUMBER: int
    IR_VERSION_PRERELEASE_FIELD_NUMBER: int
    IR_BUILD_METADATA_FIELD_NUMBER: int
    DOMAIN_FIELD_NUMBER: int
    OPSET_VERSION_FIELD_NUMBER: int
    DOC_STRING_FIELD_NUMBER: int
    OPERATOR_FIELD_NUMBER: int
    FUNCTIONS_FIELD_NUMBER: int

    magic: str
    ir_version: int
    ir_version_prerelease: str
    ir_build_metadata: str
    domain: str
    opset_version: int
    doc_string: str
    
    # 繰り返しフィールド（リスト構造）
    operator: _containers.RepeatedCompositeFieldContainer[OperatorProto]
    functions: _containers.RepeatedCompositeFieldContainer[FunctionProto]

    def __init__(
        self,
        magic: _Optional[str] = ...,
        ir_version: _Optional[int] = ...,
        ir_version_prerelease: _Optional[str] = ...,
        ir_build_metadata: _Optional[str] = ...,
        domain: _Optional[str] = ...,
        opset_version: _Optional[int] = ...,
        doc_string: _Optional[str] = ...,
        operator: _Optional[_Iterable[OperatorProto]] = ...,
        functions: _Optional[_Iterable[FunctionProto]] = ...,
    ) -> None: ...