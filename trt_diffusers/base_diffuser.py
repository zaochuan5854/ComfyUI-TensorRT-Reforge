from typing_extensions import TypedDict
from typing import Any, Optional, cast, Literal

import numpy as np
from tqdm import tqdm

import torch
import tensorrt as trt

import comfy.lora
import comfy.model_management

from ..trt_exporter import SupportedModelType, WeightsNameMap, WeightsShapeMap
from ..trt_utils import trt_datatype_to_torch, torch_dtype_to_trt

SupportedModelName = [e.name for e in SupportedModelType]

# strength_patch, strength_model, (original_weight, lora_b, lora_a, alpha)
PatchType = tuple[float, Any, float, Any, Any]

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

logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, "") # pyright: ignore[reportArgumentType]
runtime = trt.Runtime(logger)

class TRTDiffuser:
    def __init__(self, engine_path: Optional[str] = None, engine: Optional[trt.ICudaEngine] = None, weight_map: Optional[WeightsNameMap] = None, shape_map: Optional[WeightsShapeMap] = None) -> None:        
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
        self.weight_mapping: Optional[WeightsNameMap] = None
        self.shape_mapping: Optional[WeightsShapeMap] = None
        self.patches: dict[str, list[PatchType]] = {}
        self.source_state_dict: dict[str, torch.Tensor] = {}
        self._last_refit_layers: set[str] = set()
        self.refit_compute_device: torch.device = cast(torch.device, comfy.model_management.get_torch_device())

        self._rename_inputs()
        if weight_map is not None:
            self.weight_mapping = weight_map
        if shape_map is not None:
            self.shape_mapping = shape_map

    def _rename_inputs(self):
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

        self._set_bindings_shape(model_inputs, curr_split_batch)

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

    def _set_bindings_shape(self, inputs: dict[str, torch.Tensor], split_batch: int) -> None:
        for k, tensor in inputs.items():
            shape = list(tensor.shape)
            shape[0] = shape[0] // split_batch
            self.context.set_input_shape(k, shape)

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

    def set_source_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.source_state_dict = state_dict

    def load_state_dict(self, sd: dict[str, Any], strict: bool = False) -> None:
        pass

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}
