from _typeshed import Incomplete
from collections.abc import Iterable

autogen_header: str
LITE_OPTION: str
DEFAULT_PACKAGE_NAME: str
IF_ONNX_ML_REGEX: Incomplete
ENDIF_ONNX_ML_REGEX: Incomplete
ELSE_ONNX_ML_REGEX: Incomplete

def process_ifs(lines: Iterable[str], onnx_ml: bool) -> Iterable[str]: ...

IMPORT_REGEX: Incomplete
PACKAGE_NAME_REGEX: Incomplete
ML_REGEX: Incomplete

def process_package_name(lines: Iterable[str], package_name: str) -> Iterable[str]: ...

PROTO_SYNTAX_REGEX: Incomplete
OPTIONAL_REGEX: Incomplete

def convert_to_proto3(lines: Iterable[str]) -> Iterable[str]: ...
def gen_proto3_code(protoc_path: str, proto3_path: str, include_path: str, cpp_out: str, python_out: str) -> None: ...
def translate(source: str, proto: int, onnx_ml: bool, package_name: str) -> str: ...
def qualify(f: str, pardir: str | None = None) -> str: ...
def convert(stem: str, package_name: str, output: str, do_onnx_ml: bool = False, lite: bool = False, protoc_path: str = '') -> None: ...
def main() -> None: ...
