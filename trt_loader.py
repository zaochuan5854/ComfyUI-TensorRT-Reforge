from typing_extensions import TYPE_CHECKING
if TYPE_CHECKING or __name__ == "__main__":
    import sys
    from pathlib import Path
    comfy_path = Path(__file__).parent.parent.parent
    sys.path.append(str(comfy_path))

import os
import torch
import tensorrt as trt # pyright: ignore[reportMissingTypeStubs]
import onnxruntime as ort # pyright: ignore[reportMissingTypeStubs]
from onnx import TensorProto
from onnx.onnx_pb import ModelProto
from onnxruntime.capi.onnxruntime_inference_collection import Session # pyright: ignore[reportMissingTypeStubs]

import folder_paths
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
from comfy_api.latest import io
from typing_extensions import override, no_type_check, TypedDict
from typing import Any, Dict, Optional, cast, List

try:
    from .trt_exporter import SupportedModelType
    from .utils import UnifiedModel, trt_datatype_to_torch # pyright: ignore[reportUnknownVariableType]
except ImportError:
    from trt_exporter import SupportedModelType
    from utils import UnifiedModel, trt_datatype_to_torch # type: ignore

SupportedModelName = [e.name for e in SupportedModelType]

if "tensorrt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tensorrt"][0].append(
        os.path.join(folder_paths.models_dir, "tensorrt"))
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".engine")
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".onnx_and_engine.zip")
    folder_paths.folder_names_and_paths["tensorrt"][1].add(".onnx_and_engine")
else:
    folder_paths.folder_names_and_paths["tensorrt"] = (
        [os.path.join(folder_paths.models_dir, "tensorrt")], {".engine", ".onnx_and_engine.zip", ".onnx_and_engine"})


trt.init_libnvinfer_plugins(None, "") # type: ignore
logger = trt.Logger(trt.Logger.INFO) # type: ignore
trt.init_libnvinfer_plugins(logger, "") # type: ignore
runtime = trt.Runtime(logger) # type: ignore

class ModelInputNames(TypedDict):
    latent: str
    timestep: str
    context: str | None
    vector_cond: str | None

class ModelInputMapping(TypedDict):
    latent: list[str]
    timestep: list[str]
    context: list[str]
    vector_cond: list[str]

class TrTDiffuser:
    @no_type_check
    def __init__(self, engine_path: Optional[str] = None, engine: Optional[object] = None) -> None:
        if engine is not None:
            self.engine = engine
        elif engine_path is not None:
            self.engine_path = engine_path
            with open(engine_path, "rb") as f:
                deserialized_engine = runtime.deserialize_cuda_engine(f.read())
                if deserialized_engine is None:
                    raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}")
                self.engine = deserialized_engine
        else:
            raise ValueError("Either engine_path or engine must be provided.")
        
        trt_dtype = self.engine.get_tensor_dtype(self.engine.get_tensor_name(0))

        self.dtype = cast(torch.dtype, trt_datatype_to_torch(trt_dtype))
        context = self.engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")
        self.context = context

        self.model_input_names: ModelInputNames = {
            "latent": "",
            "timestep": "",
            "context": None,
            "vector_cond": None
        }
        self.input_aliases_map: ModelInputMapping = {
            "latent": ["latent", "x", "cx"],
            "timestep": ["timestep", "timesteps", "t"],
            "context": ["context", "ctx"],
            "vector_cond": ["vector_cond", "y"]
        }
        self.model_output_names: list[str] = []

        self.rename_inputs()

    @no_type_check
    def rename_inputs(self):
        engine_model_input_names: List[str] = []
        engine_model_output_names: List[str] = []

        # tensorrt is 8.5+ by requirement
        num_io_tensors = self.engine.num_io_tensors
        for i in range(num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                engine_model_input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                engine_model_output_names.append(name)

        for key, aliases in self.input_aliases_map.items():
            for alias in cast(List[str], aliases):
                if alias in engine_model_input_names:
                    self.model_input_names[key] = alias
                    break

        self.model_output_names = engine_model_output_names

    @no_type_check
    def set_bindings_shape(self, inputs: Dict[str, torch.Tensor], split_batch: int) -> None:
        for k, tensor in inputs.items():
            shape = list(tensor.shape)
            shape[0] = shape[0] // split_batch
            self.context.set_input_shape(k, tuple(shape))

    @no_type_check
    def __call__(self, latent: torch.Tensor, timestep: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        model_inputs: Dict[str, torch.Tensor] = {self.model_input_names["latent"]: latent, self.model_input_names["timestep"]: timestep}

        for arg, value in kwargs.items():
            for key, aliases in self.input_aliases_map.items():
                if arg in cast(List[str], aliases) and self.model_input_names[key] is not None:
                    model_inputs[self.model_input_names[key]] = value
                    break
        
        batch_size = latent.shape[0]
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]

        curr_split_batch = 1
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)

        model_inputs_converted: Dict[str, torch.Tensor] = {}
        for arg, tensor in model_inputs.items():
            data_type = self.engine.get_tensor_dtype(arg)
            model_inputs_converted[arg] = tensor.to(dtype=trt_datatype_to_torch(data_type))

        assert len(self.model_output_names) == 1, "Only single output is supported"
        output_binding_name = self.model_output_names[0]
        out_shape_tuple = self.engine.get_tensor_shape(output_binding_name)
        out_shape: List[int] = list(out_shape_tuple)

        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                out_shape[idx] = latent.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(out_shape, 
                          device=latent.device, 
                          dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out

        stream = torch.cuda.default_stream(latent.device)
        for i in range(curr_split_batch):
            for arg, tensor in model_inputs_converted.items():
                chunk_size = tensor.shape[0] // curr_split_batch
                self.context.set_tensor_address(arg, tensor[chunk_size * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        return out

    def load_state_dict(self, sd: Dict[str, Any], strict: bool = False) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

class AnimaONNXWrapper(torch.nn.Module):
    def __init__(self, onnx_model: ModelProto, device: torch.device, dtype: torch.dtype, device_id: int = 0) -> None:
        super().__init__()
        # if device.type == "cuda" and not check_cuda_compatibility():
        #     raise RuntimeError("CUDA is not compatible with ONNX Runtime.")
        self.device = device
        self.device_type = "cuda" if device.type == "cuda" else "cpu"
        self.device_id = device_id
        
        model_data = onnx_model.SerializeToString()
        
        providers: Any = [
            ("CUDAExecutionProvider", {"device_id": device_id}),
            "CPUExecutionProvider"
        ] if self.device_type == "cuda" else ["CPUExecutionProvider"]

        self.session = cast(Session, ort.InferenceSession(model_data, providers=providers)) # type: ignore
        
        self.target_dim = 1024
        self.dtype = dtype if self.device_type == "cuda" else torch.float32
        
        match self.dtype:
            case torch.float16:
                self.tp_dtype = TensorProto.FLOAT16
            case torch.bfloat16:
                self.tp_dtype = TensorProto.BFLOAT16
            case torch.float32:
                self.tp_dtype = TensorProto.FLOAT
            case _:
                raise ValueError(f"Unsupported dtype {dtype}")

    def forward(self, 
                source_hidden_states: torch.Tensor, 
                target_input_ids: torch.Tensor, 
                target_attention_mask: torch.Tensor | None = None, 
                source_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        
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

        io_binding.bind_input("source_hidden_states", self.device_type, self.device_id, self.tp_dtype,  # pyright: ignore[reportUnknownMemberType]
                              source_hidden_states.shape, source_hidden_states.data_ptr())
        
        io_binding.bind_input("target_input_ids", self.device_type, self.device_id, TensorProto.INT64,  # pyright: ignore[reportUnknownMemberType]
                              target_input_ids.shape, target_input_ids.data_ptr())
        
        io_binding.bind_input("target_attention_mask", self.device_type, self.device_id, self.tp_dtype,  # pyright: ignore[reportUnknownMemberType]
                              target_attention_mask.shape, target_attention_mask.data_ptr())
        io_binding.bind_input("source_attention_mask", self.device_type, self.device_id, self.tp_dtype,  # pyright: ignore[reportUnknownMemberType]
                              source_attention_mask.shape, source_attention_mask.data_ptr())

        out_shape = (B, target_seq, self.target_dim)
        out_buffer = torch.empty(out_shape, dtype=self.dtype, device=self.device)
        io_binding.bind_output("output", self.device_type, self.device_id, self.tp_dtype,  # pyright: ignore[reportUnknownMemberType]
                               out_shape, out_buffer.data_ptr())

        self.session.run_with_iobinding(io_binding)  # pyright: ignore[reportUnknownMemberType]
        
        return out_buffer


class TRTAnimaDiffuser(TrTDiffuser):
    def __init__(self, onnx_model: ModelProto, device: torch.device, engine_path: str | None = None, engine: object | None = None) -> None:
        super().__init__(engine_path, engine) # pyright: ignore[reportUnknownMemberType]
        self.llm_adapter_onnx = AnimaONNXWrapper(onnx_model, device=device, dtype=self.dtype)

    def preprocess_text_embeds(self, text_embeds: torch.Tensor, text_ids: torch.Tensor | None=None, t5xxl_weights: torch.Tensor | None=None) -> torch.Tensor:
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

class TRTLoader(io.ComfyNode):
    """
    Loads a TensorRT engine file and its associated ONNX file (if available) from a zip archive.    
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TensorRTLoaderNode",
            display_name="TensorRT Loader Reforge",
            category="TensorRT",
            inputs=[
                io.Combo.Input(
                    id="model_path",
                    display_name="Model Path",
                    options=folder_paths.get_filename_list("tensorrt"),
                    default=folder_paths.folder_names_and_paths["tensorrt"][0][0] if folder_paths.get_filename_list("tensorrt") else ""
                ),
                io.Combo.Input(
                    id="model_type",
                    display_name="Model Type",
                    options=SupportedModelName,
                    default="sdxl_base"
                ),
            ],
            outputs=[
                io.Model.Output(id="engine", display_name="MODEL"),   
            ],
            is_output_node=True
        )

    @classmethod
    @override
    def execute(cls, **kwargs: dict[str, str]) -> io.NodeOutput:
        model_path = cast(str, kwargs["model_path"])

        try:
            model_type = SupportedModelType[cast(str, kwargs["model_type"])]
        except KeyError:
            raise ValueError(f"Unexpected model_type: {kwargs['model_type']}. Supported types are: {SupportedModelName}")

        full_path: Optional[str] = folder_paths.get_full_path("tensorrt", model_path)
        if full_path is None or not os.path.isfile(full_path):
            raise FileNotFoundError(f"File {model_path} does not exist in tensorrt path")

        diffuser: TrTDiffuser
        if full_path.endswith(".onnx_and_engine.zip") or full_path.endswith(".onnx_and_engine"):
            unified_model = UnifiedModel(full_path) # pyright: ignore[reportUnknownVariableType]
            onnx_model, trt_engine = unified_model.load_instances() # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if trt_engine is None:
                raise RuntimeError("Failed to extract TensorRT engine from UnifiedModel")
            diffuser = TRTAnimaDiffuser(onnx_model, device=cast(torch.device, comfy.model_management.get_torch_device()), engine=trt_engine) # pyright: ignore[reportUnknownArgumentType]
        else:
            diffuser = TrTDiffuser(engine_path=full_path)

        model: Any = None

        match model_type:
            case SupportedModelType.SD15:
                config = comfy.supported_models.SD15(model_type.value.config)
                config.unet_config["disable_unet_model_creation"] = True # pyright: ignore[reportUnknownMemberType]
                model = comfy.model_base.BaseModel(config)
            case SupportedModelType.SDXL:
                config = comfy.supported_models.SDXL(model_type.value.config)
                config.unet_config["disable_unet_model_creation"] = True # pyright: ignore[reportUnknownMemberType]
                model = comfy.model_base.SDXL(config)
            case SupportedModelType.AuraFlow:
                config = comfy.supported_models.AuraFlow(model_type.value.config)
                config.unet_config["disable_unet_model_creation"] = True # pyright: ignore[reportUnknownMemberType]
                model = config.get_model({}) # pyright: ignore[reportUnknownMemberType]
            case SupportedModelType.Flux:
                config = comfy.supported_models.Flux(model_type.value.config)
                config.unet_config["disable_unet_model_creation"] = True # pyright: ignore[reportUnknownMemberType]
                model = config.get_model({}) # pyright: ignore[reportUnknownMemberType]
            case SupportedModelType.SD3:
                config = comfy.supported_models.SD3(model_type.value.config)
                config.unet_config["disable_unet_model_creation"] = True # pyright: ignore[reportUnknownMemberType]
                model = config.get_model({}) # pyright: ignore[reportUnknownMemberType]
            case SupportedModelType.Anima:
                config = comfy.supported_models.Anima(model_type.value.config)
                config.unet_config["disable_unet_model_creation"] = True # pyright: ignore[reportArgumentType, reportUnknownMemberType]
                model = config.get_model({}) # pyright: ignore[reportUnknownMemberType]
            case SupportedModelType.SVD:
                config = comfy.supported_models.SVD_img2vid(model_type.value.config)
                config.unet_config["disable_unet_model_creation"] = True # pyright: ignore[reportUnknownMemberType]
                model = config.get_model({}) # pyright: ignore[reportUnknownMemberType]
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")

        model.diffusion_model = diffuser
        model.memory_required = lambda *args, **kwargs: 0 # type: ignore

        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=cast(torch.device, comfy.model_management.get_torch_device()),
            offload_device=cast(torch.device, comfy.model_management.unet_offload_device())
        )

        return (io.NodeOutput(patcher))
