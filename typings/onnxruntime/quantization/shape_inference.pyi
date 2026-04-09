import onnx
from .fusions import ReplaceUpsampleWithResize as ReplaceUpsampleWithResize
from .onnx_model import ONNXModel as ONNXModel
from .quant_utils import add_pre_process_metadata as add_pre_process_metadata, save_and_reload_model_with_shape_infer as save_and_reload_model_with_shape_infer
from _typeshed import Incomplete
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference as SymbolicShapeInference
from onnxruntime.transformers.onnx_utils import extract_raw_data_from_model as extract_raw_data_from_model, has_external_data as has_external_data
from pathlib import Path

logger: Incomplete

def quant_pre_process(input_model: str | Path | onnx.ModelProto | None = None, output_model_path: str | Path | None = None, skip_optimization: bool = False, skip_onnx_shape: bool = False, skip_symbolic_shape: bool = False, auto_merge: bool = False, int_max: int = ..., guess_output_rank: bool = False, verbose: int = 0, save_as_external_data: bool = False, all_tensors_to_one_file: bool = False, external_data_location: str | None = None, external_data_size_threshold: int = 1024, **deprecated_kwargs) -> None: ...
