from typing_extensions import override
from typing import Any, Optional, cast, Callable, no_type_check
from types import MethodType

import os

import torch

import comfy.utils
import comfy.model_base
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
from comfy_api.latest import io
import folder_paths

from .definitions import SupportedModelType, WeightsNameMap, WeightsShapeMap, PatchType
from .trt_utils import ModelBundle, BundleEntryType
from .trt_diffusers.base_diffuser import TRTDiffuser
from .trt_diffusers.anima_diffuser import TRTAnimaDiffuser

SupportedModelName = [e.name for e in SupportedModelType]

class TRTLoader(io.ComfyNode):
    """
    Loads a TensorRT engine file and its associated ONNX file (if available) from a zip archive.    
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_options = _ensure_tensorrt_search_paths()
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
        shape_mapping: Optional[WeightsShapeMap] = None
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

def _ensure_tensorrt_search_paths() -> list[str]:
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
    return folder_paths.get_filename_list("tensorrt")

class TRTModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(self, model: comfy.model_base.BaseModel, load_device: torch.device, offload_device: torch.device, size: int = 0, weight_inplace_update: bool=False, weight_mapping: Optional[WeightsNameMap] = None, shape_mapping: Optional[WeightsShapeMap] = None, bundle_metadata: Optional[dict[str, Any]] = None) -> None:
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
        self.model.state_dict = MethodType(self._ret_dummy_state_dict, self.model)

    def _resolve_original_weight(self, bundle_metadata: Optional[dict[str, Any]]) -> Optional[str]:
        if bundle_metadata is None:
            return None
        metadata_key = "source_model"
        original_path = bundle_metadata.get(metadata_key)
        if isinstance(original_path, str):
            if os.path.isfile(original_path):
                return original_path
            else:
                target_basename = os.path.basename(original_path)
                target_dir = folder_paths.models_dir
                for root, _, files in os.walk(target_dir):
                    if target_basename in files:
                        candidate_path = os.path.join(root, target_basename)
                        if os.path.isfile(candidate_path):
                            return candidate_path
        else:
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

    def _ret_dummy_state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
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
        n.model.state_dict = MethodType(n._ret_dummy_state_dict, n.model) # pyright: ignore[reportUnknownArgumentType]
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
                diffuser.set_source_state_dict(self._ret_dummy_state_dict())
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
