from _typeshed import Incomplete
from symbolic_shape_infer import SymbolicShapeInference

file_path: Incomplete
logger: Incomplete

class SymbolicShapeInferenceHelper(SymbolicShapeInference):
    model_: Incomplete
    all_shapes_inferred_: bool
    is_inferred_: bool
    dynamic_axis_mapping_: dict[str, int]
    def __init__(self, model, verbose: int = 0, int_max=..., auto_merge: bool = True, guess_output_rank: bool = False) -> None: ...
    def infer(self, dynamic_axis_mapping: dict[str, int], max_runs: int = 200): ...
    def get_edge_shape(self, edge): ...
    def compare_shape(self, edge, edge_other): ...
