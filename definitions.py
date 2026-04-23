from typing import NamedTuple, Optional, TypedDict, Any
from enum import Enum

import tensorrt as trt

from comfy import model_base

# Global TensorRT runtime
trt_logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(trt_logger, "") # pyright: ignore[reportArgumentType]
trt_runtime = trt.Runtime(trt_logger)

# Exporter Definitions
WeightsNameMap = dict[str, str]
WeightsShapeMap = dict[str, tuple[tuple[int, ...], bool]]

class ModelMetaInfo(NamedTuple):
    model_type: type
    config: dict[str, Any]

class SupportedModelType(Enum):
    SD15 = ModelMetaInfo(model_base.BaseModel, {}) #Opset18 OK
    SDXL = ModelMetaInfo(model_base.SDXL, {"adm_in_channels": 2816}) #Opset18 OK
    AuraFlow = ModelMetaInfo(model_base.AuraFlow, {})
    Flux = ModelMetaInfo(model_base.Flux, {})
    SD3 = ModelMetaInfo(model_base.SD3, {}) #Medium Opset18 OK
    Anima = ModelMetaInfo(model_base.Anima, {}) #Opset25 OK
    SVD = ModelMetaInfo(model_base.SVD_img2vid, {})

    @classmethod
    def from_instance(cls, instance: object) -> "SupportedModelType":
        for model_type in cls:
            if type(instance) is model_type.value.model_type:
                return model_type
        raise NotImplementedError("Unsupported model type: {}".format(type(instance)))
    
    @staticmethod
    def semantic_list() -> list[str]:
        l = [model_type.name for model_type in SupportedModelType]
        l.reverse()
        l.append(l.pop(l.index(SupportedModelType.SVD.name)))
        return l

class TRTSpec(TypedDict):
    model_name: str
    filename_prefix: str
    enable_lora: bool

    opt_batch_size: int
    opt_width: int
    opt_height: int
    opt_context_mult: int

    min_batch_size: int
    max_batch_size: int
    
    min_width: int
    max_width: int
    
    min_height: int
    max_height: int
    
    min_context_mult: int
    max_context_mult: int
    
    num_video_frames: int

# Loader Definitions
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
    