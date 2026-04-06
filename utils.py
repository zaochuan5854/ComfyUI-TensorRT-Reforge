import os
import onnx
import mmap
import torch
import tensorrt as trt # pyright: ignore[reportMissingTypeStubs]
import onnxruntime as ort # pyright: ignore[reportMissingTypeStubs]
from onnx import helper, TensorProto
from onnx.onnx_pb import ModelProto
from typing import Optional, no_type_check, cast


class UnifiedModel:
    def __init__(self, merged_path: str):
        self.merged_path = merged_path
        self._file = open(merged_path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._mv = memoryview(self._mm)
        
        self.onnx_size = int.from_bytes(self._mv[0:8], "little")
        self.trt_size = int.from_bytes(self._mv[8:16], "little")
        
        offset = 16
        self.onnx_view = self._mv[offset : offset + self.onnx_size]
        offset += self.onnx_size
        self.engine_view = self._mv[offset : offset + self.trt_size]

        # インスタンス保持用
        self.onnx_model: Optional[ModelProto] = None
        self.trt_engine: Optional[object] = None # type: ignore

    @staticmethod
    def unify_onnx_and_trt_engine(onnx_path: str, trt_engine_path: str, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx_data = onnx_path + ".data"
        onnx_data_exists = os.path.exists(onnx_data)
        model = onnx.load(onnx_path) # pyright: ignore[reportUnknownMemberType]
        mono_onnx = os.path.join(os.path.dirname(onnx_path), "mono_" + os.path.basename(onnx_path))
        onnx.save_model(model, mono_onnx, save_as_external_data=False, all_tensors_to_one_file=onnx_data_exists) # pyright: ignore[reportUnknownMemberType]
        onnx_data_size = os.path.getsize(mono_onnx).to_bytes(8, "little")
        trt_engine_size = os.path.getsize(trt_engine_path).to_bytes(8, "little")
        header = onnx_data_size + trt_engine_size
        with open(output_path, "wb") as f_out:
            f_out.write(header)
            with open(mono_onnx, "rb") as f_onnx:
                f_out.write(f_onnx.read())
            with open(trt_engine_path, "rb") as f_trt:
                f_out.write(f_trt.read())

        print(f"Unified model saved to: {output_path}")


    def load_instances(self) -> tuple[ModelProto, object]:
        """インスタンスを生成する"""
        logger = trt.Logger(trt.Logger.INFO) # type: ignore
        runtime = trt.Runtime(logger) # type: ignore
        self.trt_engine = cast(object, runtime.deserialize_cuda_engine(self.engine_view)) # type: ignore

        self.onnx_model = onnx.load_model_from_string(self.onnx_view.tobytes())
        self.onnx_view.release()
        if self.trt_engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine from memory")
        return self.onnx_model, self.trt_engine

    def close(self):
        """
        すべての memoryview を明示的に解放してから mmap を閉じる
        """
        if hasattr(self, 'onnx_view'):
            self.onnx_view.release()
            del self.onnx_view
            
        if hasattr(self, 'engine_view'):
            self.engine_view.release()
            del self.engine_view

        if hasattr(self, '_mv'):
            self._mv.release()
            del self._mv

        if hasattr(self, '_mm'):
            self._mm.close()
            del self._mm

        if hasattr(self, '_file'):
            self._file.close()
            del self._file

    def __del__(self):
        self.close()


@no_type_check
def trt_datatype_to_torch(datatype: object) -> torch.dtype:
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

@no_type_check
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
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [1])
    ], [helper.make_tensor_value_info("C", TensorProto.FLOAT, [1])])
    model = helper.make_model(graph)
    
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"]) # type: ignore
        return True
    except Exception as e:
        print(f"[Warning] CUDA found but incompatible: {e}")
        return False
