import pathlib
import typing
from ..logger import get_logger as get_logger
from .operator_type_usage_processors import OperatorTypeUsageManager as OperatorTypeUsageManager
from .ort_model_processor import OrtFormatModelProcessor as OrtFormatModelProcessor
from _typeshed import Incomplete

log: Incomplete

def create_config_from_models(model_files: typing.Iterable[pathlib.Path], output_file: pathlib.Path, enable_type_reduction: bool): ...
