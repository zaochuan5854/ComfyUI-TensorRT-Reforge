import os
import onnx
import enum
import json
import shutil
import mmap
import gc
import torch
import tensorrt as trt # pyright: ignore[reportMissingTypeStubs]
import onnxruntime as ort # pyright: ignore[reportMissingTypeStubs]
from onnx import helper, TensorProto
from onnx.onnx_pb import ModelProto
from typing import Literal, Any, overload, Optional, TypeVar

T = TypeVar("T")

class BundleEntryType(enum.Enum):
    TRT_ENGINE = 1
    ONNX_MODEL = 2
    WEIGHTS_MAP = 3

WeightsNameMap = dict[str, str]
WeightsShapeMap = dict[str, tuple[tuple[int, ...], bool]]

class ModelBundle:
    def __init__(self, merged_path: str):
        self.merged_path = merged_path
        self._file = open(merged_path, "r+b")
        self._mm: Optional[mmap.mmap] = None
        self._entry_views: dict[BundleEntryType, memoryview] = {}
        self._reload_views()

    def _open_views(self):
        if self._mm is None:
            raise RuntimeError("mmap is not initialized")

        self._entry_views.clear()
        self._mv = memoryview(self._mm)
        file_size = len(self._mv)
        if file_size < 8:
            raise ValueError(f"Invalid bundle file (too small): {self.merged_path}")

        cursor = file_size - 8
        meta_size = int.from_bytes(self._mv[cursor: cursor + 8], "little")
        cursor -= meta_size
        if cursor < 0:
            raise ValueError(f"Invalid bundle metadata size in: {self.merged_path}")

        self._meta_offset = cursor
        self._meta_view = self._mv[self._meta_offset: self._meta_offset + meta_size]

        # Parse chunks backward from metadata start.
        # Chunk layout is: [Data][Size:8B][Type:1B]
        while cursor > 0:
            if cursor < 9:
                raise ValueError(f"Corrupted bundle chunk footer in: {self.merged_path}")

            cursor -= 1
            entry_type = BundleEntryType(int.from_bytes(self._mv[cursor:cursor + 1], "little"))

            cursor -= 8
            entry_size = int.from_bytes(self._mv[cursor:cursor + 8], "little")

            cursor -= entry_size
            if cursor < 0:
                raise ValueError(f"Corrupted bundle chunk size in: {self.merged_path}")

            if entry_type not in self._entry_views:
                self._entry_views[entry_type] = self._mv[cursor: cursor + entry_size]

    def _reload_views(self):
        self.close_views()
        self._close_mmap()

        if os.path.getsize(self.merged_path) == 0:
            raise ValueError(f"Invalid bundle file (empty): {self.merged_path}")

        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._open_views()

    @overload
    def __getitem__(self, entry: Literal[BundleEntryType.TRT_ENGINE]) -> trt.ICudaEngine:
        ...
    @overload
    def __getitem__(self, entry: Literal[BundleEntryType.ONNX_MODEL]) -> ModelProto:
        ...
    @overload
    def __getitem__(self, entry: Literal[BundleEntryType.WEIGHTS_MAP]) -> tuple[WeightsNameMap, WeightsShapeMap]:
        ...

    def __getitem__(self, entry: BundleEntryType) -> Any:
        if entry not in self._entry_views:
            raise KeyError(f"Entry {entry} not found in bundle")
        raw_view = self._entry_views[entry]
        match entry:
            case BundleEntryType.TRT_ENGINE:
                logger = trt.Logger(trt.Logger.INFO) # type: ignore
                runtime = trt.Runtime(logger) # type: ignore
                return runtime.deserialize_cuda_engine(raw_view) # type: ignore
            case BundleEntryType.ONNX_MODEL:
                # Need to copy to bytes to avoid memoryview being released (which would cause the engine/model to become invalid)
                return onnx.load_model_from_string(raw_view.tobytes()) # pyright: ignore[reportUnknownMemberType]
            case BundleEntryType.WEIGHTS_MAP:
                payload = json.loads(raw_view.tobytes().decode("utf-8"))
                weights_name = payload.get("weights_name", {})
                raw_shape = payload.get("weights_shape", {})
                weights_shape: WeightsShapeMap = {
                    k: (tuple(v[0]), bool(v[1])) for k, v in raw_shape.items()
                }
                return weights_name, weights_shape
            case _:
                raise KeyError(f"Unknown entry type: {entry}")
    
    @overload
    def get(self, entry: Literal[BundleEntryType.TRT_ENGINE], default: T = None) -> trt.ICudaEngine | T:
        ...
    @overload
    def get(self, entry: Literal[BundleEntryType.ONNX_MODEL], default: T = None) -> ModelProto | T:
        ...
    @overload
    def get(self, entry: Literal[BundleEntryType.WEIGHTS_MAP], default: T = None) -> tuple[WeightsNameMap, WeightsShapeMap] | T:
        ...

    def get(self, entry: BundleEntryType, default: Any = None) -> Any:
        if entry in self._entry_views:
            return self[entry]
        return default

    def save_weights_mapping(self, weights_name: WeightsNameMap, weights_shape: WeightsShapeMap):
        payload: dict[str, Any] = {
            "weights_name": weights_name,
            "weights_shape": {
                k: [list(shape), is_transpose]
                for k, (shape, is_transpose) in weights_shape.items()
            },
        }
        self.append_entry(BundleEntryType.WEIGHTS_MAP, json.dumps(payload).encode("utf-8"))

    def load_weights_mapping(self) -> tuple[WeightsNameMap, WeightsShapeMap]:
        if BundleEntryType.WEIGHTS_MAP not in self._entry_views:
            raise KeyError("weights mapping is not stored in bundle")
        return self[BundleEntryType.WEIGHTS_MAP]
            
    def append_entry(self, entry_type: BundleEntryType, data: bytes):
        metadata_bytes = self._meta_view.tobytes()
        self.close_views()
        self._close_mmap()

        self._file.seek(self._meta_offset)
        self._file.write(data)
        self._file.write(len(data).to_bytes(8, "little"))
        self._file.write(entry_type.value.to_bytes(1, "little"))

        self._file.write(metadata_bytes)
        self._file.write(len(metadata_bytes).to_bytes(8, "little"))
        self._file.truncate()
        self._file.flush()
        self._reload_views()
    
    def __contains__(self, key: BundleEntryType) -> bool:
        return key in self._entry_views
    
    def __keys__(self):
        return self._entry_views.keys()

    @property
    def metadata(self) -> dict[Any, Any]:
        if not hasattr(self, "_cached_metadata"):
            # Parse metadata JSON if not cached. If meta_view is empty, treat as empty dict.
            if len(self._meta_view) == 0:
                self._cached_metadata = {}
            else:
                self._cached_metadata = json.loads(self._meta_view.tobytes().decode("utf-8"))
        return self._cached_metadata

    @metadata.setter
    def metadata(self, value: dict[Any, Any]):
        raw = json.dumps(value).encode("utf-8")
        new_metadata_size = len(raw)
        new_size = self._meta_offset + new_metadata_size + 8

        self.close_views()
        self._close_mmap()

        self._file.truncate(new_size)
        self._file.seek(self._meta_offset)
        self._file.write(raw)
        self._file.write(new_metadata_size.to_bytes(8, "little"))
        self._file.flush()

        self._reload_views()
        self._cached_metadata = value

    @staticmethod
    def _serialize_onnx_model(onnx_path: str):
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        onnx_data_file = os.path.join(os.path.dirname(onnx_path), os.path.splitext(os.path.basename(onnx_path))[0] + ".onnx.data")
        if os.path.exists(onnx_data_file):
            if os.path.getsize(onnx_path) + os.path.getsize(onnx_data_file) < 1.5 * 1024 * 1024 * 1024:
                # To sirialize ONNX model with external data, we need to load and re-serialize it, which can be memory intensive.
                # If the model data file is larger than 1.5GB, we will raise an error for now to avoid potential OOM. Future optimization can be done by directly copying the external data file and adjusting the ONNX model's external data reference without fully loading it into memory.
                model = onnx.load(onnx_path) # pyright: ignore[reportUnknownMemberType]
                onnx_bytes = model.SerializeToString()
                with open(onnx_path, "wb") as f_out:
                    f_out.write(onnx_bytes)
                    f_out.flush()
                return onnx_path
            else:
                raise NotImplementedError(f"ONNX model with large external data file (>1.5GB) is not supported yet: {onnx_data_file}")

    @classmethod
    def from_onnx(cls, onnx_path: str, output_path: str, replace_source: bool = True):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cls._serialize_onnx_model(onnx_path)
        if replace_source:
            os.rename(onnx_path, output_path)
            byte_length = os.path.getsize(output_path).to_bytes(8, "little")
            with open(output_path, "ab") as f_out:
                f_out.write(byte_length)
                f_out.write(BundleEntryType.ONNX_MODEL.value.to_bytes(1, "little"))
                f_out.write((0).to_bytes(8, "little")) # meta size 0
                f_out.flush()
        else:
            byte_length = os.path.getsize(onnx_path).to_bytes(8, "little")
            with open(onnx_path, "rb") as f_source, open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_source, f_out)
                f_out.write(byte_length)
                f_out.write(BundleEntryType.ONNX_MODEL.value.to_bytes(1, "little"))
                f_out.write((0).to_bytes(8, "little")) # meta size 0
                f_out.flush()

        return cls(output_path)
    
    @classmethod
    def from_trt_engine(cls, trt_engine_path: str, output_path: str, replace_source: bool = True):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        trt_byte_length = os.path.getsize(trt_engine_path).to_bytes(8, "little")
        if replace_source:
            with open(trt_engine_path, "ab") as f:
                f.write(trt_byte_length)
                f.write(BundleEntryType.TRT_ENGINE.value.to_bytes(1, "little"))
                f.write((0).to_bytes(8, "little")) # meta size 0
                f.flush()
            os.rename(trt_engine_path, output_path)

        else:
            with open(trt_engine_path, "rb") as f_source, open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_source, f_out)
                f_out.write(trt_byte_length)
                f_out.write(BundleEntryType.TRT_ENGINE.value.to_bytes(1, "little"))
                f_out.write((0).to_bytes(8, "little")) # meta size 0
                f_out.flush()

        return cls(output_path)

    @classmethod
    def from_onnx_and_trt_engine(cls, onnx_path: str, trt_engine_path: str, output_path: str, replace_source: bool = True) -> "ModelBundle":
        cls._serialize_onnx_model(onnx_path)
        len_onnx_bytes = os.path.getsize(onnx_path)
        len_trt_bytes = os.path.getsize(trt_engine_path)
        
        if len_onnx_bytes >= len_trt_bytes:
            base_path, source_path = onnx_path, trt_engine_path
            base_type, source_type = BundleEntryType.ONNX_MODEL, BundleEntryType.TRT_ENGINE
            base_size, source_size = len_onnx_bytes, len_trt_bytes
        else:
            base_path, source_path = trt_engine_path, onnx_path
            base_type, source_type = BundleEntryType.TRT_ENGINE, BundleEntryType.ONNX_MODEL
            base_size, source_size = len_trt_bytes, len_onnx_bytes
        os.rename(base_path, output_path)
        with open(output_path, "ab") as f_base, open(source_path, "rb") as f_source:
            f_base.write(base_size.to_bytes(8, "little"))
            f_base.write(base_type.value.to_bytes(1, "little"))
            
            shutil.copyfileobj(f_source, f_base)
            f_base.write(source_size.to_bytes(8, "little"))
            f_base.write(source_type.value.to_bytes(1, "little"))
            # Writing empty metadata
            f_base.write((0).to_bytes(8, "little")) # meta size 0
            f_base.flush()

        return cls(output_path)

    def close_views(self):
        for view in self._entry_views.values():
            try:
                view.release()
            except ValueError:
                pass
        self._entry_views.clear()

        if hasattr(self, "_meta_view"):
            try:
                self._meta_view.release()
            except ValueError:
                pass
            del self._meta_view

        if hasattr(self, "_mv"):
            try:
                self._mv.release()
            except ValueError:
                pass
            del self._mv

    def _close_mmap(self):
        if self._mm is None:
            return
        try:
            self._mm.close()
        except BufferError:
            # Be defensive for lingering exported pointers.
            gc.collect()
            self._mm.close()
        finally:
            self._mm = None

    def close(self):
        self.close_views()
        self._close_mmap()

        if hasattr(self, "_file"):
            self._file.close()
            del self._file

    def __del__(self):
        self.close()

def trt_datatype_to_torch(datatype: trt.DataType) -> torch.dtype:
    type_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    
    if hasattr(trt.DataType, "BF16"):
        type_map[trt.DataType.BF16] = torch.bfloat16
    
    if datatype in type_map:
        return type_map[datatype]
        
    raise ValueError(f"Unsupported TensorRT datatype: {datatype}")

def torch_dtype_to_trt(datatype: torch.dtype) -> trt.DataType:
    type_map = {
        torch.float32: trt.DataType.FLOAT,
        torch.float16: trt.DataType.HALF,
        torch.int32: trt.DataType.INT32,
        torch.bool: trt.DataType.BOOL,
    }
    
    if hasattr(torch, "bfloat16"):
        type_map[torch.bfloat16] = trt.DataType.BF16
    
    if datatype in type_map:
        return type_map[datatype]
        
    raise ValueError(f"Unsupported torch dtype for TensorRT conversion: {datatype}")

def check_cuda_compatibility() -> bool:
    """
    Check if CUDA is available and compatible with ONNX Runtime."""
    node = helper.make_node("Add", ["A", "B"], ["C"])
    graph = helper.make_graph([node], "test", [
        helper.make_tensor_value_info("A", TensorProto.DataType.FLOAT, [1]),
        helper.make_tensor_value_info("B", TensorProto.DataType.FLOAT, [1])
    ], [helper.make_tensor_value_info("C", TensorProto.DataType.FLOAT, [1])])
    model = helper.make_model(graph)
    
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"]) # type: ignore
        return True
    except Exception as e:
        print(f"[Warning] CUDA found but incompatible: {e}")
        return False

# /**
#  * OMNI Hybrid Sequential Format Spec (2026)
#  * -----------------------------------------------------------------------------
#  * [File Layout]
#  * +-----------------------------------+ <--- Offset 0
#  * | [ID:1][Size:8B][TRT Data...]       | Data Chunk 1
#  * +-----------------------------------+
#  * | [ID:2][Size:8B][ONNX Data...]      | Data Chunk 2
#  * +-----------------------------------+
#  * | [ID:3][Size:8B][WeightsMap JSON...]| Data Chunk 3 (Optional)
#  * +-----------------------------------+ <--- End of Data Chunks
#  * |                                   |
#  * |      Metadata Section (JSON)      | Variable Length (No ID)
#  * |                                   |
#  * +-----------------------------------+ <--- Metadata End (EOF - 8 bytes)
#  * |   Metadata Size (8 bytes)         | uint64, Little Endian
#  * +-----------------------------------+ <--- EOF (End of File)
#  * -----------------------------------------------------------------------------
#  * [Parsing Logic]
#  * 1. Read last 8 bytes to get `meta_size`.
#  * 2. Metadata starts at: (EOF - 8 - meta_size).
#  * 3. Chunks start at 0 and repeat until (EOF - 8 - meta_size).
#  *
#  * [Advantages]
#  * - Metadata update: Truncate at `End of Data Chunks`, then write new JSON + size.
#  * - No shifting: Huge model data (GBs) never moves.
#  */
    