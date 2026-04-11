import enum
import pathlib
from .file_utils import files_from_file_or_dir as files_from_file_or_dir, path_match_suffix_ignore_case as path_match_suffix_ignore_case
from .onnx_model_utils import get_optimization_level as get_optimization_level
from .ort_format_model import create_config_from_models as create_config_from_models

class OptimizationStyle(enum.Enum):
    Fixed = 0
    Runtime = 1

def parse_args(): ...
def convert_onnx_models_to_ort(model_path_or_dir: pathlib.Path, output_dir: pathlib.Path | None = None, optimization_styles: list[OptimizationStyle] | None = None, custom_op_library_path: pathlib.Path | None = None, target_platform: str | None = None, save_optimized_onnx_model: bool = False, allow_conversion_failures: bool = False, enable_type_reduction: bool = False): ...
