from typing import Any, Optional, cast

import torch
import tensorrt as trt
import onnxruntime as ort
from onnx.onnx_pb import ModelProto, TensorProto
from onnxruntime.capi.onnxruntime_inference_collection import Session

from .base_diffuser import TRTDiffuser
from ..trt_exporter import WeightsNameMap, WeightsShapeMap

class TRTAnimaDiffuser(TRTDiffuser):
    def __init__(self, onnx_model: ModelProto, device: torch.device, engine_path: Optional[str] = None, engine: Optional[trt.ICudaEngine] = None, weight_map: Optional[WeightsNameMap] = None, shape_map: Optional[WeightsShapeMap] = None) -> None:
        super().__init__(engine_path=engine_path, engine=engine, weight_map=weight_map, shape_map=shape_map)
        self.llm_adapter_onnx = AnimaONNXWrapper(onnx_model, device=device, dtype=self.dtype)

    def preprocess_text_embeds(self, text_embeds: torch.Tensor, text_ids: Optional[torch.Tensor] = None, t5xxl_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if text_ids is not None:
            origin_device, origin_dtype = text_embeds.device, text_embeds.dtype
            out = self.llm_adapter_onnx(text_embeds, text_ids)
            if out.dtype != origin_dtype or out.device != origin_device:
                out = out.to(origin_dtype).to(origin_device)
            if t5xxl_weights is not None:
                out = out * t5xxl_weights

            if out.shape[1] < 512:
                out = torch.nn.functional.pad(out, (0, 0, 0, 512 - out.shape[1]))
            return out
        else:
            return text_embeds

class AnimaONNXWrapper(torch.nn.Module):
    def __init__(self, onnx_model: ModelProto, device: torch.device, dtype: torch.dtype, device_id: int = 0) -> None:
        super().__init__()
        # if device.type == "cuda" and not check_cuda_compatibility():
        #     raise RuntimeError("CUDA is not compatible with ONNX Runtime.")
        self.device = device
        self.device_type = "cuda" if device.type == "cuda" else "cpu"
        self.device_id = device_id
        
        model_data = onnx_model.SerializeToString()
        
        providers: list[str | tuple[str, dict[str, Any]]] = [
            ("CUDAExecutionProvider", {"device_id": device_id}),
            "CPUExecutionProvider"
        ] if self.device_type == "cuda" else ["CPUExecutionProvider"]

        self.session = cast(Session, ort.InferenceSession(model_data, providers=providers)) 
        
        self.target_dim = 1024
        self.dtype = dtype if self.device_type == "cuda" else torch.float32
        
        match self.dtype:
            case torch.float16:
                self.tp_dtype = TensorProto.DataType.FLOAT16
            case torch.bfloat16:
                self.tp_dtype = TensorProto.DataType.BFLOAT16
            case torch.float32:
                self.tp_dtype = TensorProto.DataType.FLOAT
            case _:
                raise ValueError(f"Unsupported dtype {dtype}")

    def forward(self, 
                source_hidden_states: torch.Tensor, 
                target_input_ids: torch.Tensor, 
                target_attention_mask: Optional[torch.Tensor] = None, 
                source_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        B, source_seq = source_hidden_states.shape[0], source_hidden_states.shape[1]
        target_seq = target_input_ids.shape[1]

        if source_hidden_states.dtype != self.dtype or source_hidden_states.device.type != self.device_type:
            source_hidden_states = source_hidden_states.to(self.dtype).to(self.device)
        if target_input_ids.dtype != torch.int64 or target_input_ids.device.type != self.device_type:
            target_input_ids = target_input_ids.to(torch.int64).to(self.device)
        if target_attention_mask is not None and (target_attention_mask.dtype != self.dtype or target_attention_mask.device.type != self.device_type):
            target_attention_mask = target_attention_mask.to(self.dtype).to(self.device)
        if source_attention_mask is not None and (source_attention_mask.dtype != self.dtype or source_attention_mask.device.type != self.device_type):
            source_attention_mask = source_attention_mask.to(self.dtype).to(self.device)

        if target_attention_mask is None:
            target_attention_mask = torch.ones((B, target_seq), dtype=self.dtype, device=self.device)
        if source_attention_mask is None:
            source_attention_mask = torch.ones((B, source_seq), dtype=self.dtype, device=self.device)

        io_binding = self.session.io_binding()

        io_binding.bind_input("source_hidden_states", self.device_type, self.device_id, self.tp_dtype, # pyright: ignore[reportUnknownMemberType]
                              source_hidden_states.shape, source_hidden_states.data_ptr())
        
        io_binding.bind_input("target_input_ids", self.device_type, self.device_id, TensorProto.DataType.INT64, # pyright: ignore[reportUnknownMemberType]
                              target_input_ids.shape, target_input_ids.data_ptr())
        
        io_binding.bind_input("target_attention_mask", self.device_type, self.device_id, self.tp_dtype, # pyright: ignore[reportUnknownMemberType]
                              target_attention_mask.shape, target_attention_mask.data_ptr())
        io_binding.bind_input("source_attention_mask", self.device_type, self.device_id, self.tp_dtype, # pyright: ignore[reportUnknownMemberType]
                              source_attention_mask.shape, source_attention_mask.data_ptr())

        out_shape = (B, target_seq, self.target_dim)
        out_buffer = torch.empty(out_shape, dtype=self.dtype, device=self.device)
        io_binding.bind_output("output", self.device_type, self.device_id, self.tp_dtype, # pyright: ignore[reportUnknownMemberType]
                               out_shape, out_buffer.data_ptr())

        self.session.run_with_iobinding(io_binding) # pyright: ignore[reportUnknownMemberType]
        
        return out_buffer
