from typing_extensions import override, TypedDict
from typing import Any, Optional, cast, Callable, Literal, no_type_check
from types import MethodType

import os
import time
import numpy as np
from tqdm import tqdm

import torch
import tensorrt as trt
import onnxruntime as ort
from onnx import TensorProto
from onnx.onnx_pb import ModelProto
from onnxruntime.capi.onnxruntime_inference_collection import Session

import comfy.utils
import comfy.lora
import comfy.model_base
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
from comfy_api.latest import io
import folder_paths

from .trt_exporter import SupportedModelType
from .trt_utils import ModelBundle, BundleEntryType, WeightsNameMap, trt_datatype_to_torch, torch_dtype_to_trt, resolve_safe_model_metadata_path

SupportedModelName = [e.name for e in SupportedModelType]


def _ensure_tensorrt_search_paths() -> None:
    models_trt_dir = os.path.join(folder_paths.models_dir, "tensorrt")
    output_trt_dir = os.path.join(folder_paths.get_output_directory(), "tensorrt")

    if "tensorrt" in folder_paths.folder_names_and_paths:
        search_paths, suffixes = folder_paths.folder_names_and_paths["tensorrt"]
        for candidate in (models_trt_dir, output_trt_dir):
            if candidate not in search_paths:
                search_paths.append(candidate)
        suffixes.add(".engine")
        suffixes.add(".bundle")
    else:
        folder_paths.folder_names_and_paths["tensorrt"] = (
            [models_trt_dir, output_trt_dir], {".engine", ".bundle"}
        )


_ensure_tensorrt_search_paths()

if "original_weight_name" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["original_weight_name"] = (
        [os.path.join(folder_paths.models_dir, "checkpoints"),
         os.path.join(folder_paths.models_dir, "diffusion_models")], 
        {".safetensors"}
    )
else:
    folder_paths.folder_names_and_paths["original_weight_name"][1].add(".safetensors")

WeightMapType = dict[str, str]
ShapeMapType = dict[str, tuple[tuple[int, ...], bool]]

# strength_patch, strength_model, (original_weight, lora_b, lora_a, alpha)
PatchType = tuple[float, Any, float, Any, Any]

logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, "") # pyright: ignore[reportArgumentType]
runtime = trt.Runtime(logger)

class ModelInputNames(TypedDict):
    latent: str
    timestep: str
    context: Optional[str]
    vector_cond: Optional[str]

class ModelInputMapping(TypedDict):
    latent: list[str]
    timestep: list[str]
    context: list[str]
    vector_cond: list[str]

class TRTDiffuser:
    def __init__(self, engine_path: Optional[str] = None, engine: Optional[trt.ICudaEngine] = None, weight_map: Optional[WeightsNameMap] = None, shape_map: Optional[ShapeMapType] = None) -> None:        
        if engine is not None:
            self.engine = engine
        elif engine_path is not None:
            self.engine_path = engine_path
            with open(engine_path, "rb") as f:
                deserialized_engine = runtime.deserialize_cuda_engine(f.read())
                self.engine = deserialized_engine
        else:
            raise ValueError("Either engine_path or engine must be provided.")
        
        trt_dtype = self.engine.get_tensor_dtype(self.engine.get_tensor_name(0))

        self.dtype = trt_datatype_to_torch(trt_dtype)
        self.context = self.engine.create_execution_context()

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
        self.weight_mapping: Optional[WeightMapType] = None
        self.shape_mapping: Optional[ShapeMapType] = None
        self.patches: dict[str, list[PatchType]] = {}
        self.source_state_dict: dict[str, torch.Tensor] = {}
        self._last_refit_layers: set[str] = set()
        self.refit_compute_device: torch.device = cast(torch.device, comfy.model_management.get_torch_device())

        self.rename_inputs()
        if weight_map is not None:
            self.weight_mapping = weight_map
        if shape_map is not None:
            self.shape_mapping = shape_map

    def rename_inputs(self):
        engine_model_input_names: list[str] = []
        engine_model_output_names: list[str] = []

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
            for alias in cast(list[str], aliases):
                if alias in engine_model_input_names:
                    self.model_input_names[key] = alias
                    break

        self.model_output_names = engine_model_output_names

    def set_bindings_shape(self, inputs: dict[str, torch.Tensor], split_batch: int) -> None:
        for k, tensor in inputs.items():
            shape = list(tensor.shape)
            shape[0] = shape[0] // split_batch
            self.context.set_input_shape(k, shape)

    def set_source_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.source_state_dict = state_dict

    def refit(self):
        self._validate_refit_support()
        refitter = trt.Refitter(self.engine, logger)
        keep_weights: dict[str, torch.Tensor] = {}

        target_layers = list(self._last_refit_layers)
        for layer in self.patches:
            if self._should_refit_layer(layer) and layer not in target_layers:
                target_layers.append(layer)

        pbar = tqdm(target_layers, desc="[TRT] Calculating and Refitting LoRA Layers", total=len(target_layers))
        weight_device_map = self._schedule_device(target_layers)
        cpu_buffers = self._create_cpu_buffer([layer for layer in target_layers if weight_device_map[layer] == "cpu"])
        event = torch.cuda.Event()
        event.record()
        try:
            for layer in pbar:
                pbar.set_postfix({"layer": layer.split('.')[-1]}) # pyright: ignore[reportUnknownMemberType]

                base_weight = self._build_base_weight(layer)
                patch_list = self.patches.get(layer, [])
                if patch_list:
                    base_weight = self._apply_patches(layer, base_weight, patch_list)
                final_weight = self._prepare_final_weight(layer, base_weight)
                if weight_device_map[layer] == "cpu":
                    cpu_buffers[layer].copy_(final_weight, non_blocking=True)
                    keep_weights[layer] = cpu_buffers[layer]
                else:
                    host_weight = self._set_refit_weights(refitter, layer, final_weight)
                    keep_weights[layer] = host_weight
        finally:
            pbar.close()
            event.record()
            event.synchronize()
            cpu_layers = [layer for layer in target_layers if weight_device_map[layer] == "cpu"]
            for cpu_layer in cpu_layers:
                host_weight = self._set_refit_weights(refitter, cpu_layer, cpu_buffers[cpu_layer])
        
        self._commit_refit(refitter, keep_weights)
        self._last_refit_layers = {layer for layer in target_layers if self._should_refit_layer(layer)}
        self._finalize_refit(keep_weights)

    def _validate_refit_support(self) -> None:
        if self.patches and not (self.weight_mapping and self.shape_mapping):
            raise RuntimeError("This model does not support LoRA.")
        if (self.patches or self._last_refit_layers) and not self.source_state_dict:
            raise RuntimeError("Source state_dict is required for TensorRT refit but not found.")

    def _should_refit_layer(self, layer: str) -> bool:
        return self.weight_mapping is not None and layer in self.weight_mapping

    def _schedule_device(self, layers: list[str]) -> dict[str, Literal["cpu", "cuda"]]:
        assert self.shape_mapping is not None, "Shape mapping is required for refitting but not found."
        free, _ = torch.cuda.mem_get_info("cuda")
        gpu_memory_margin = 1.8 # Alocate 1.8x the weight size on GPU to account for temporary tensors during refit
        weight_device_map: dict[str, Literal["cpu", "cuda"]] = {}
        # Process layers in reverse to prioritize weights scheduled for CPU by assigning them first.
        # This allows asynchronous transfers to start earlier, maximizing overlap with GPU tasks.
        for layer in reversed(layers):
            shapes = list(self.shape_mapping[layer][0])
            weight_size = self.dtype.itemsize * np.prod(shapes) * gpu_memory_margin
            free -= weight_size
            if free > 0:
                weight_device_map[layer] = "cuda"
            else:
                weight_device_map[layer] = "cpu"
        return weight_device_map
    
    def _create_cpu_buffer(self, layers: list[str]) -> dict[str, torch.Tensor]:
        assert self.shape_mapping is not None, "Shape mapping is required for refitting but not found."
        alignment = 256
        pointer = 0
        layer_range: dict[str, tuple[int, int]] = {}
        for layer in layers:
            shapes = list(self.shape_mapping[layer][0])
            weight_size = int(self.dtype.itemsize * np.prod(shapes))
            layer_range[layer] = (pointer, pointer + weight_size)
            pointer += (weight_size + alignment - 1) // alignment * alignment

        buffer_size = pointer
        storage = torch.UntypedStorage(buffer_size, device="cpu").pin_memory()
        cpu_buffers: dict[str, torch.Tensor] = {}
        for layer in layers:
            start, end = layer_range[layer]
            buffer = cast(torch.UntypedStorage, storage[start:end])
            cpu_buffers[layer] = torch.tensor(buffer, dtype=self.dtype, device="cpu").view(self.shape_mapping[layer][0])

        return cpu_buffers

    def _build_base_weight(self, layer: str) -> torch.Tensor:
        if layer not in self.source_state_dict:
            raise RuntimeError(f"Source weight for layer {layer} not found.")
        return self.source_state_dict[layer].detach().clone().to(device=self.refit_compute_device, dtype=torch.float32)

    def _apply_patches(self, layer: str, base_weight: torch.Tensor, patch_list: list[PatchType]) -> torch.Tensor:
        def _identity_convert(weight: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return weight

        original_weights = {
            layer: [(base_weight.detach().clone(), _identity_convert)],
        }
        out_weight: torch.Tensor = cast(torch.Tensor, comfy.lora.calculate_weight( # pyright: ignore[reportUnknownMemberType]
            patch_list,
            base_weight,
            layer,
            intermediate_dtype=torch.float32,
            original_weights=original_weights,
        ))
        return out_weight

    def _prepare_final_weight(self, layer: str, base_weight: torch.Tensor) -> torch.Tensor:
        if self.shape_mapping is None or layer not in self.shape_mapping:
            raise RuntimeError(f"Shape mapping is required for refitting layer {layer} but not found.")
        is_transpose = self.shape_mapping[layer][1]
        if is_transpose:
            final_weight = base_weight.T.contiguous()
        else:
            final_weight = base_weight.contiguous()

        # Match engine dtype (FP16/FP32/BF16 depending on engine build).
        return final_weight.to(self.dtype)

    def _set_refit_weights(self, refitter: trt.Refitter, layer: str, final_weight: torch.Tensor) -> torch.Tensor:
        assert self.weight_mapping is not None, "Weight mapping is required for refitting but not found."
        trt_weights = trt.Weights(
            torch_dtype_to_trt(final_weight.dtype),
            final_weight.data_ptr(),
            final_weight.numel(),
        )

        success = refitter.set_named_weights(self.weight_mapping[layer], trt_weights)
        if not success:
            raise RuntimeError(f"Failed to set weights for {layer}")
        return final_weight

    def _commit_refit(self, refitter: trt.Refitter, keep_weights: dict[str, torch.Tensor], num_tries: int = 5, current_try: int = 0) -> None:
        assert self.weight_mapping is not None, "Weight mapping is required for refitting but not found."
        print("[TRT] Committing refit to engine...")
        success = refitter.refit_cuda_engine()
        if success:
            print("[TRT] Engine refit successful.")
        else:
            time.sleep(1) # Wait a moment for refitter to release any GPU resources, improving chances of successful retrieval of missing weight info
            missing_onnx_weights = refitter.get_missing_weights()
            if current_try >= num_tries or not missing_onnx_weights:
                raise RuntimeError("Final engine refit failed without reporting missing weights.")

            missing_layers = [layer for layer, weight_name in self.weight_mapping.items() if weight_name in missing_onnx_weights]
            
            for layer in missing_layers:
                if layer not in self.source_state_dict:
                    raise RuntimeError(f"Final engine refit failed: Missing weight '{layer}' not found in original state dict.")
                
                print(f"[TRT] WARNING: Missing weight for '{layer}'. Assigning original weight from state dict.")
                base_weight = self._build_base_weight(layer)
                final_weight = self._prepare_final_weight(layer, base_weight)
                host_weight = self._set_refit_weights(refitter, layer, final_weight)
                keep_weights[layer] = host_weight
            
            self._commit_refit(refitter, keep_weights, num_tries, current_try + 1)

    def _finalize_refit(self, keep_weights: dict[str, torch.Tensor]) -> None:
        self.patches = {}
        del keep_weights

    def __call__(self, latent: torch.Tensor, timestep: torch.Tensor, **kwargs: Any) -> torch.Tensor:

        model_inputs: dict[str, torch.Tensor] = {self.model_input_names["latent"]: latent, self.model_input_names["timestep"]: timestep}

        for arg, value in kwargs.items():
            for key, aliases in self.input_aliases_map.items():
                if arg in cast(list[str], aliases) and self.model_input_names[key] is not None:
                    model_inputs[self.model_input_names[key]] = value
                    break
        
        batch_size = latent.shape[0]
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        # opt_batch = dims[1][0]
        max_batch = dims[2][0]

        curr_split_batch = 1
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)

        model_inputs_converted: dict[str, torch.Tensor] = {}
        for arg, tensor in model_inputs.items():
            data_type = self.engine.get_tensor_dtype(arg)
            model_inputs_converted[arg] = tensor.to(dtype=trt_datatype_to_torch(data_type))

        assert len(self.model_output_names) == 1, "Only single output is supported"
        output_binding_name = self.model_output_names[0]
        out_shape_tuple = self.engine.get_tensor_shape(output_binding_name)
        out_shape: list[int] = list(out_shape_tuple)

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

    def load_state_dict(self, sd: dict[str, Any], strict: bool = False) -> None:
        pass

    def state_dict(self) -> dict[str, torch.Tensor]:
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


class TRTAnimaDiffuser(TRTDiffuser):
    def __init__(self, onnx_model: ModelProto, device: torch.device, engine_path: str | None = None, engine: trt.ICudaEngine | None = None, weight_map: Optional[WeightsNameMap] = None, shape_map: Optional[ShapeMapType] = None) -> None:
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

class TRTLoader(io.ComfyNode):
    """
    Loads a TensorRT engine file and its associated ONNX file (if available) from a zip archive.    
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_options = folder_paths.get_filename_list("tensorrt")
        return io.Schema(
            node_id="TensorRTLoaderNode",
            display_name="TensorRT Loader Reforge",
            category="TensorRT",
            inputs=[
                io.Combo.Input(
                    id="model_path",
                    display_name="Model Path",
                    options=model_options,
                    default=model_options[0] if model_options else ""
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

        diffuser: TRTDiffuser
        weight_mapping: Optional[WeightsNameMap] = None
        shape_mapping: Optional[ShapeMapType] = None
        metadata: Optional[dict[str, Any]] = None

        if full_path.endswith(".bundle"):
            model_bundle = ModelBundle(full_path)
            weight_mapping, shape_mapping = model_bundle.get(BundleEntryType.WEIGHTS_MAP, (None, None))
            metadata = model_bundle.metadata
            if model_type == SupportedModelType.Anima:
                onnx_model, trt_engine = model_bundle[BundleEntryType.ONNX_MODEL], model_bundle[BundleEntryType.TRT_ENGINE]
                diffuser = TRTAnimaDiffuser(onnx_model, device=cast(torch.device, comfy.model_management.get_torch_device()), engine=trt_engine, weight_map=weight_mapping, shape_map=shape_mapping)
            else:
                trt_engine = model_bundle[BundleEntryType.TRT_ENGINE]
                diffuser = TRTDiffuser(engine=trt_engine, weight_map=weight_mapping, shape_map=shape_mapping)
        else:
            diffuser = TRTDiffuser(engine_path=full_path)

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
                config.unet_config["disable_unet_model_creation"] = True
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
                config.unet_config["disable_unet_model_creation"] = True # pyright: ignore[reportArgumentType]
                model = config.get_model({}) # pyright: ignore[reportUnknownMemberType]
            case SupportedModelType.SVD:
                config = comfy.supported_models.SVD_img2vid(model_type.value.config)
                config.unet_config["disable_unet_model_creation"] = True # pyright: ignore[reportUnknownMemberType]
                model = config.get_model({}) # pyright: ignore[reportUnknownMemberType]
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")

        model.diffusion_model = diffuser # pyright: ignore[reportAttributeAccessIssue]
        model.memory_required = lambda *args, **kwargs: 0 # type: ignore

        patcher = TRTModelPatcher(
            model,
            load_device=cast(torch.device, comfy.model_management.get_torch_device()),
            offload_device=cast(torch.device, comfy.model_management.unet_offload_device()),
            weight_mapping=weight_mapping,
            shape_mapping=shape_mapping,
            bundle_metadata=metadata
        )

        return (io.NodeOutput(patcher))

class TRTModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(self, model: comfy.model_base.BaseModel, load_device: torch.device, offload_device: torch.device, size: int = 0, weight_inplace_update: bool=False, weight_mapping: Optional[WeightsNameMap] = None, shape_mapping: Optional[ShapeMapType] = None, bundle_metadata: Optional[dict[str, Any]] = None) -> None:
        super().__init__(model, load_device, offload_device, size=size, weight_inplace_update=weight_inplace_update)
        self.weight_mapping = weight_mapping
        self.shape_mapping = shape_mapping
        self.original_weight_path = self._resolve_original_weight(bundle_metadata)
        if self.original_weight_path is not None:
            state_dict, cpkt_metadata = cast(tuple[dict[str, torch.Tensor], dict[str, Any]], comfy.utils.load_torch_file(self.original_weight_path, return_metadata=True)) # pyright: ignore[reportUnknownMemberType]
            self.dummy_state_dict = self._modify_state_dict(state_dict, cpkt_metadata) # pyright: ignore[reportUnknownMemberType]
        else:
            self.dummy_state_dict = None
        assert isinstance(self.model, comfy.model_base.BaseModel), "Model must be an instance of comfy.model_base.BaseModel"
        self.model.state_dict = MethodType(self.ret_dummy_state_dict, self.model)

    def _resolve_original_weight(self, bundle_metadata: Optional[dict[str, Any]]) -> Optional[str]:
        if bundle_metadata is None:
            return None
        metadata_key = "source_model"
        original_path = bundle_metadata.get(metadata_key)
        if isinstance(original_path, str):
            resolved = resolve_safe_model_metadata_path("original_weight_name", original_path)
            if resolved is not None:
                return resolved
        return None

    @no_type_check
    def _modify_state_dict(self, state_dict: dict[str, torch.Tensor], ckpt_metadata: Optional[dict[str, Any]]) -> dict[str, torch.Tensor]:
        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
        temp_sd = comfy.utils.state_dict_prefix_replace(state_dict, {diffusion_model_prefix: ""}, filter_keys=True)
        if len(temp_sd) > 0:
            state_dict = temp_sd
            model_config = comfy.model_detection.model_config_from_unet(state_dict, "", metadata=ckpt_metadata)

        if model_config is not None:
            new_sd = state_dict
        else:
            new_sd = comfy.model_detection.convert_diffusers_mmdit(state_dict, "") # pyright: ignore[reportUnknownMemberType]
            if new_sd is not None: #diffusers mmdit
                model_config = comfy.model_detection.model_config_from_unet(new_sd, "") # pyright: ignore[reportUnknownMemberType]
                if model_config is None:
                    return None
            else: #diffusers unet
                model_config = comfy.model_detection.model_config_from_diffusers_unet(state_dict) # pyright: ignore[reportUnknownMemberType]
                if model_config is None:
                    return None

                diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config) # pyright: ignore[reportUnknownMemberType]

                new_sd = {}
                for k in diffusers_keys:
                    if k in state_dict:
                        new_sd[diffusers_keys[k]] = state_dict.pop(k)
                    else:
                        print("{} {}".format(diffusers_keys[k], k))
        return new_sd

    def ret_dummy_state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        if self.dummy_state_dict is None:
            return {}
        diffusion_model_prefix = "diffusion_model."
        ret_state_dict = {k if k.startswith(diffusion_model_prefix) else diffusion_model_prefix+k : v for k, v in self.dummy_state_dict.items()}
        return ret_state_dict
    
    @override
    def clone(self, disable_dynamic: bool = False, model_override: Optional[Any] = None):
        weight_mapping = self.weight_mapping
        shape_mapping = self.shape_mapping
        original_weight_path = self.original_weight_path
        dummy_state_dict = self.dummy_state_dict
        n = cast(TRTModelPatcher, super().clone(disable_dynamic=disable_dynamic, model_override=model_override)) # pyright: ignore[reportUnknownMemberType]
        n.weight_mapping = weight_mapping
        n.shape_mapping = shape_mapping
        n.original_weight_path = original_weight_path
        n.dummy_state_dict = dummy_state_dict
        n.model.state_dict = MethodType(n.ret_dummy_state_dict, n.model) # pyright: ignore[reportUnknownArgumentType]
        return n

    @override
    def add_patches(self, patches: dict[str, Any], strength_patch: float = 1.0, strength_model: float = 1.0) -> list[str]:
        return cast(list[str], super().add_patches(patches, strength_patch=strength_patch, strength_model=strength_model)) # pyright: ignore[reportUnknownMemberType]

    @override
    def load(self, device_to: Optional[torch.device] = None, lowvram_model_memory: int = 0, force_patch_weights: bool = False, full_load: bool = False) -> None:
        if isinstance(self.model, comfy.model_base.BaseModel) and isinstance(self.model.diffusion_model, TRTDiffuser):
            diffuser = self.model.diffusion_model
            current_uuid = getattr(self.model, "current_weight_patches_uuid", None)
            needs_refit = (force_patch_weights or (current_uuid != self.patches_uuid)) and (hasattr(self, "patches") and len(cast(dict[str, list[PatchType]], getattr(self, "patches"))) > 0)

            if needs_refit:
                diffuser.set_source_state_dict(self.ret_dummy_state_dict())
                diffuser.patches = cast(dict[str, list[PatchType]], getattr(self, "patches"))
                diffuser.refit()

            setattr(self.model, "model_lowvram", False)
            setattr(self.model, "lowvram_patch_counter", 0)
            setattr(self.model, "device", device_to)
            setattr(self.model, "model_loaded_weight_memory", 0)
            setattr(self.model, "model_offload_buffer_memory", 0)
            setattr(self.model, "current_weight_patches_uuid", self.patches_uuid)

            for callback in cast(list[Callable[[TRTModelPatcher, Optional[torch.device], int, bool, bool], None]], self.get_all_callbacks(comfy.model_patcher.CallbacksMP.ON_LOAD)):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(cast(Any, self.forced_hooks), force_apply=True) # pyright: ignore[reportUnknownMemberType]
            return

        super().load(device_to=device_to, lowvram_model_memory=lowvram_model_memory, force_patch_weights=force_patch_weights, full_load=full_load) # pyright: ignore[reportUnknownMemberType]

    @override
    def unpatch_model(self, device_to: Optional[torch.device] = None, unpatch_weights: bool = True):
        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            self.unpin_all_weights()

            if device_to is not None:
                self.model.device = device_to

            self.model.model_loaded_weight_memory = 0
            self.model.model_offload_buffer_memory = 0

        object_patches_backup = cast(dict[str, Any], getattr(self, "object_patches_backup"))
        keys = list(object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(cast(Any, self.model), k, object_patches_backup[k]) # pyright: ignore[reportUnknownMemberType]
        object_patches_backup.clear()
