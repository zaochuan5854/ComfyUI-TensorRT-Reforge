from _typeshed import Incomplete
from onnx.reference.op_run import OpRun as OpRun

class PreCalc:
    pos1: Incomplete
    pos2: Incomplete
    pos3: Incomplete
    pos4: Incomplete
    w1: Incomplete
    w2: Incomplete
    w3: Incomplete
    w4: Incomplete
    def __init__(self, pos1: int = 0, pos2: int = 0, pos3: int = 0, pos4: int = 0, w1: int = 0, w2: int = 0, w3: int = 0, w4: int = 0) -> None: ...

class RoiAlign(OpRun):
    @staticmethod
    def pre_calc_for_bilinear_interpolate(height: int, width: int, pooled_height: int, pooled_width: int, iy_upper: int, ix_upper: int, roi_start_h, roi_start_w, bin_size_h, bin_size_w, roi_bin_grid_h: int, roi_bin_grid_w: int, pre_calc): ...
    @staticmethod
    def roi_align_forward(output_shape: tuple[int, int, int, int], bottom_data, spatial_scale, height: int, width: int, sampling_ratio, bottom_rois, num_roi_cols: int, top_data, mode, half_pixel: bool, batch_indices_ptr): ...
