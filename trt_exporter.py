from typing_extensions import TYPE_CHECKING
if TYPE_CHECKING or __name__ == "__main__":
    import sys
    from pathlib import Path
    comfy_path = Path(__file__).parent.parent.parent
    sys.path.append(str(comfy_path))

import os
import time

import torch
from torch import nn
import tensorrt as trt # pyright: ignore[reportMissingTypeStubs]

import folder_paths
from comfy_api.latest import io
from comfy.model_patcher import ModelPatcher
from comfy.ldm.anima.model import LLMAdapter
from comfy import model_base, model_management

from enum import Enum
from typing import Any, NamedTuple
from typing_extensions import override, cast, assert_never, no_type_check, TypedDict, Unpack

try:
    from .utils import UnifiedModel
except ImportError:
    # for debugging
    from utils import UnifiedModel # type: ignore

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
    model_patcher: ModelPatcher

    opt_batch_size: int
    opt_width: int
    opt_height: int
    opt_context_mult: int
    filename_prefix: str

    min_batch_size: int
    max_batch_size: int
    
    min_width: int #Width First for comfyui
    max_width: int #Width First for comfyui
    
    min_height: int
    max_height: int
    
    min_context_mult: int
    max_context_mult: int
    
    num_video_frames: int

def trt_spec_to_string(spec: TRTSpec) -> str:
    is_static = spec["min_batch_size"]  == spec["max_batch_size"]  == 0 and \
                spec["min_width"] == spec["max_width"] == 0 and \
                spec["min_height"] == spec["max_height"] == 0 and \
                spec["min_context_mult"] == spec["max_context_mult"] == 0
    if is_static:
        return f"W{spec['opt_width']}_H{spec['opt_height']}_BS{spec['opt_batch_size']}_CM{spec['opt_context_mult']}"
    else:
        return f"W{spec['min_width']}-{spec['opt_width']}-{spec['max_width']}_H{spec['min_height']}-{spec['opt_height']}-{spec['max_height']}_BS{spec['min_batch_size']}-{spec['opt_batch_size']}-{spec['max_batch_size']}_CM{spec['min_context_mult']}-{spec['opt_context_mult']}-{spec['max_context_mult']}"

class TRTExporter(io.ComfyNode):
    """
    Exports a TensorRT engine file and its associated ONNX file (if required) from a given model patcher and specifications.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TensorRTExporterNode",
            display_name="TensorRT Exporter Reforge",
            category="TensorRT",
            inputs=[
                io.Model.Input(id="model_patcher", display_name="MODEL"),
                io.Int.Input(id="opt_width", display_name="Opt Width", default=512, min=1), #Width First for comfyui
                io.Int.Input(id="opt_height", display_name="Opt Height", default=512, min=1),
                io.Int.Input(id="opt_batch_size", display_name="Opt Batch Size", default=1, min=1),

                io.Int.Input(id="opt_context_mult", display_name="Opt Context Multiplier", default=1, min=1),
                io.Int.Input(id="num_video_frames", display_name="Num Video Frames", default=1, min=1),
                io.String.Input(id="filename_prefix", display_name="Filename Prefix", default="tensorrt/"),

                io.Int.Input(id="min_width", display_name="Min Width", default=0, min=0),
                io.Int.Input(id="max_width", display_name="Max Width", default=0, min=0),

                io.Int.Input(id="min_height", display_name="Min Height", default=0, min=0),
                io.Int.Input(id="max_height", display_name="Max Height", default=0, min=0),

                io.Int.Input(id="min_batch_size", display_name="Min Batch Size", default=0, min=0),
                io.Int.Input(id="max_batch_size", display_name="Max Batch Size", default=0, min=0),

                io.Int.Input(id="min_context_mult", display_name="Min Context Multiplier", default=0, min=0),
                io.Int.Input(id="max_context_mult", display_name="Max Context Multiplier", default=0, min=0),
            ],
            is_output_node=True
        )

    @classmethod
    @override
    def execute(cls, **kwargs: Unpack[TRTSpec]) -> io.NodeOutput:
        model_patcher = kwargs["model_patcher"]
        min_batch_size, opt_batch_size, max_batch_size = kwargs["min_batch_size"], kwargs["opt_batch_size"], kwargs["max_batch_size"]
        min_height, opt_height, max_height = kwargs["min_height"], kwargs["opt_height"], kwargs["max_height"]
        min_width, opt_width, max_width = kwargs["min_width"], kwargs["opt_width"], kwargs["max_width"]
        min_context_mult, opt_context_mult, max_context_mult = kwargs["min_context_mult"], kwargs["opt_context_mult"], kwargs["max_context_mult"]

        def _adjust_range(min_val: int, max_val: int, opt_val: int) -> tuple[int, int]:
            new_min = opt_val if min_val == 0 else min(min_val, opt_val)
            new_max = opt_val if max_val == 0 else max(max_val, opt_val)
            return new_min, new_max

        min_batch_size, max_batch_size = _adjust_range(min_batch_size, max_batch_size, opt_batch_size)
        min_height, max_height = _adjust_range(min_height, max_height, opt_height)
        min_width, max_width = _adjust_range(min_width, max_width, opt_width)
        min_context_mult, max_context_mult = _adjust_range(min_context_mult, max_context_mult, opt_context_mult)

        num_video_frames = kwargs["num_video_frames"]

        model_management.unload_all_models()
        model_management.load_models_gpu([model_patcher], force_patch_weights=True, force_full_load=True) # pyright: ignore[reportUnknownMemberType]
        
        diffusion_model = cast(model_base.BaseModel, model_patcher.model)
        diffuser = diffusion_model.diffusion_model

        model_config: Any = diffusion_model.model_config
        unet_config: dict[str, Any]|None = None
        if hasattr(model_config, "unet_config"):
            unet_config = cast(dict[str, Any], model_config.unet_config)
        
        model_type = SupportedModelType.from_instance(diffusion_model)

        context_dim, context_len, context_len_min, dtype = get_context_features(model_type, unet_config)

        if model_type == SupportedModelType.SVD:
            min_batch_size *= num_video_frames
            max_batch_size *= num_video_frames

        is_svd = unet_config.get("use_temporal_resblock", False) == True if unet_config is not None else False
        y_dim = cast(int, diffusion_model.adm_channels) # pyright: ignore[reportUnknownMemberType]
        use_rope3d = unet_config.get("pos_emb_cls", "None") == "rope3d" if unet_config is not None else False

        transformer_options = cast(dict[str, Any], model_patcher.model_options['transformer_options'].copy()) # pyright: ignore[reportUnknownMemberType]

        inputs, inputs_shapes_min, inputs_shapes_opt, inputs_shapes_max, input_names, output_names, dynamic_shapes = build_onnx_tracing_input(
            unet_config, model_type,
            min_batch_size, opt_batch_size, max_batch_size,
            min_context_mult, opt_context_mult, max_context_mult,
            min_height, opt_height, max_height,
            min_width, opt_width, max_width,
            context_len_min, context_len, context_dim,
            use_rope3d, num_video_frames, y_dim, dtype,
        )

        tracing_model = build_onnx_tracing_model(is_svd, diffuser, input_names, transformer_options, num_video_frames)

        temp_dir = folder_paths.get_temp_directory()
        output_onnx = os.path.normpath(os.path.join(temp_dir, f"trt_{time.time()}", "model.onnx"))
        os.makedirs(os.path.dirname(output_onnx), exist_ok=True)

        print("[TensorRT] Exporting ONNX Model (Dynamo=True) ...")
        # Anima is supported in opset 25
        
        match model_type:
            case SupportedModelType.Anima:
                opset_version = 25
            case _:
                opset_version = 18

        torch.onnx.export( # pyright: ignore[reportUnknownMemberType]
            tracing_model.eval(),
            inputs,
            output_onnx,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_shapes=dynamic_shapes,
            dynamo=True,
        )

        model_management.unload_all_models()
        model_management.soft_empty_cache()

        if trt is None:
            raise RuntimeError("TensorRT is not installed but required for compilation.")

        print("[TensorRT] Building TensorRT Engine ...")
        
        output_dir = folder_paths.get_output_directory()
        output_dir = os.path.join(output_dir, os.path.dirname(kwargs["filename_prefix"]))
        basename = os.path.basename(kwargs["filename_prefix"]) + trt_spec_to_string(kwargs) + f".{model_type.name}"
        trt_output_path = os.path.join(output_dir, f"{basename}.engine")

        build_tensorrt_engine(
            output_onnx, input_names, inputs_shapes_min, inputs_shapes_opt, inputs_shapes_max, trt_output_path,
        )

        print(f"[TensorRT] Conversion completed. Saved to {trt_output_path}")

        if model_type == SupportedModelType.Anima:
            # Require llm_adapter
            print("[TensorRT] Exporting LLMAdapter to ONNX ...")
            llm_adapter = cast(LLMAdapter, diffuser.llm_adapter) # pyright: ignore[reportUnknownMemberType]
            output_onnx_llm_adapter = os.path.normpath(os.path.join(temp_dir, f"trt_{time.time()}", "llm_adapter.onnx"))
            os.makedirs(os.path.dirname(output_onnx_llm_adapter), exist_ok=True)
            export_anima_llmadapter_to_onnx(llm_adapter, output_onnx_llm_adapter, dtype)
            merged_output_path = os.path.join(output_dir, f"{basename}.onnx_and_engine")
            UnifiedModel.unify_onnx_and_trt_engine(output_onnx_llm_adapter, trt_output_path, merged_output_path) # pyright: ignore[reportUnknownMemberType]
            os.remove(trt_output_path)
            print(f"[TensorRT] LLMAdapter ONNX export completed. Saved to {output_onnx_llm_adapter}")

        return io.NodeOutput()

def get_context_features(model_type: SupportedModelType, unet_config: dict[str, Any]|None) -> tuple[int, int, int, torch.dtype]:
    match model_type:
        case SupportedModelType.SD3:
            if unet_config:
                context_embedder_config = unet_config.get("context_embedder_config", None)
                if context_embedder_config:
                    context_dim = context_embedder_config.get("params", {}).get("in_features", None)
                    context_len = 154
                    context_len_min = 77
                else:
                    _context_dim = unet_config.get("context_dim", None)
                    if _context_dim is None:
                        raise NotImplementedError("Unsupported model configuration: context_embedder_config or context_dim is required for SD3")
                    context_dim = _context_dim
                    context_len = 77
                    context_len_min = 77
                dtype = unet_config.get("dtype", torch.float32)
            else:
                raise NotImplementedError("Unsupported model configuration: context_embedder_config is required for SD3 if unet_config is not provided")
            
        case SupportedModelType.AuraFlow:
            context_dim = 2048
            context_len = 512
            context_len_min = 512
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32

        case SupportedModelType.Flux:
            _context_dim = unet_config.get("context_in_dim", None) if unet_config is not None else None
            if _context_dim is None:
                raise NotImplementedError("Unsupported model configuration: context_in_dim is required for Flux")
            context_dim = _context_dim
            context_len = 512
            context_len_min = 256
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32

        case SupportedModelType.Anima:
            _context_dim = unet_config.get("crossattn_emb_channels", None) if unet_config is not None else None
            if _context_dim is None:
                raise NotImplementedError("Unsupported model configuration: crossattn_emb_channels is required for Anima")
            context_dim = _context_dim
            context_len = 512
            context_len_min = 512
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32

        case SupportedModelType.SVD:
            _context_dim = unet_config.get("context_dim", None) if unet_config is not None else None
            if _context_dim is None:
                raise NotImplementedError("Unsupported model configuration: context_dim is required for SVD/SVD_XT")
            context_dim = _context_dim
            context_len = 1
            context_len_min = 1
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32

        case SupportedModelType.SD15 | SupportedModelType.SDXL:
            _context_dim = unet_config.get("context_dim", None) if unet_config is not None else None
            if _context_dim is None:
                raise NotImplementedError("Unsupported model configuration: context_dim is required for SD15/SDXL")
            context_dim = _context_dim
            context_len = 77
            context_len_min = 77
            dtype = unet_config.get("dtype", torch.float32) if unet_config is not None else torch.float32

        case _:
            assert_never(model_type)

    return context_dim, context_len, context_len_min, dtype

def build_onnx_tracing_input(
    unet_config: dict[str, Any]|None, model_type: SupportedModelType,
    bs_min: int, bs_opt: int, bs_max: int,
    ctx_min: int, ctx_opt: int, ctx_max: int,
    min_height: int, opt_height: int, max_height: int,
    min_width: int, opt_width: int, max_width: int,
    context_len_min: int, context_len: int, context_dim: int,
    use_rope3d: bool, num_video_frames: int, y_dim: int, dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], list[str], list[str], dict[str, dict[int, Any]]]:
    batch_size_dim = torch.export.Dim("batch_size")
    height_dim = torch.export.Dim("height")
    width_dim = torch.export.Dim("width")
    num_embeds_dim = torch.export.Dim("num_embeds")
    num_video_frames_dim = torch.export.Dim("num_video_frames")

    input_names = ["latent", "timestep", "context"]
    output_names = ["output"]
    dynamic_shapes: dict[str, dict[int, Any]] = {
        "latent": {0: batch_size_dim, 2: height_dim, 3: width_dim},
        "timestep": {0: batch_size_dim},
        "context": {0: batch_size_dim, 1: num_embeds_dim}
    }

    # Build shapes and dummy inputs
    input_channels: int = int(unet_config.get("in_channels", 4)) if unet_config is not None else 4
    
    inputs_shapes_min: tuple[tuple[int, ...], ...] = (
        (bs_min, input_channels, min_height // 8, min_width // 8),
        (bs_min,),
        (bs_min, context_len_min * ctx_min, context_dim),
    )
    inputs_shapes_opt: tuple[tuple[int, ...], ...] = (
        (bs_opt, input_channels, opt_height // 8, opt_width // 8),
        (bs_opt,),
        (bs_opt, context_len * ctx_opt, context_dim),
    )
    inputs_shapes_max: tuple[tuple[int, ...], ...] = (
        (bs_max, input_channels, max_height // 8, max_width // 8),
        (bs_max,),
        (bs_max, context_len * ctx_max, context_dim),
    )

    # Anima latent: (bs, input_channels, num_video_frames, h//8, w//8)
    if use_rope3d and num_video_frames >= 1:
        inputs_shapes_min = ((bs_min, input_channels, num_video_frames, min_height // 8, min_width // 8),) + inputs_shapes_min[1:]
        inputs_shapes_opt = ((bs_opt, input_channels, num_video_frames, opt_height // 8, opt_width // 8),) + inputs_shapes_opt[1:]
        inputs_shapes_max = ((bs_max, input_channels, num_video_frames, max_height // 8, max_width // 8),) + inputs_shapes_max[1:]
        dynamic_shapes["latent"] = {0: batch_size_dim, 2: num_video_frames_dim, 3: height_dim, 4: width_dim}

    # SDXL
    if y_dim > 0:
        input_names.append("vector_cond")
        dynamic_shapes["vector_cond"] = {0: batch_size_dim}
        inputs_shapes_min += ((bs_min, y_dim),)
        inputs_shapes_opt += ((bs_opt, y_dim),)
        inputs_shapes_max += ((bs_max, y_dim),)

    # FLUX
    extra_input: dict[str, tuple[int, ...]] = {}
    if model_type == SupportedModelType.Flux:
        extra_input = {"guidance": ()}

    for k, v in extra_input.items():
        input_names.append(k)
        dynamic_shapes[k] = {0: batch_size_dim}
        inputs_shapes_min += ((bs_min,) + v,)
        inputs_shapes_opt += ((bs_opt,) + v,)
        inputs_shapes_max += ((bs_max,) + v,)

    device = cast(torch.device, model_management.get_torch_device())
    inputs: tuple[torch.Tensor, ...] = tuple(torch.zeros(shape, device=device, dtype=dtype) for shape in inputs_shapes_opt)
    
    return inputs, inputs_shapes_min, inputs_shapes_opt, inputs_shapes_max, input_names, output_names, dynamic_shapes


def build_onnx_tracing_model(is_svd: bool, diffuser: nn.Module, input_names: list[str], transformer_options: dict[str, Any], num_video_frames: int|None) -> nn.Module:
    
    if is_svd:
        if num_video_frames is None:
            raise ValueError("num_video_frames is required for SVD models")
        class SVDTracingModel(nn.Module):
            @override
            def __init__(self, diffuser: nn.Module, num_video_frames: int, transformer_options: dict[str, Any]):
                super().__init__()
                self.diffuser = diffuser
                self.num_video_frames = num_video_frames
                self.transformer_options = transformer_options

            def forward(self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, vector_cond: torch.Tensor):
                return self.diffuser(latent, timestep, context, vector_cond, num_video_frames=self.num_video_frames, transformer_options=self.transformer_options)

        return SVDTracingModel(diffuser, num_video_frames, transformer_options).eval()

    else:
        class TracingModel(nn.Module):
            @override
            def __init__(self, diffuser: nn.Module, transformer_options: dict[str, Any]):
                super().__init__()
                self.diffuser = diffuser
                self.transformer_options = transformer_options

            def forward(self, latent: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, vector_cond: torch.Tensor|None = None, guidance: torch.Tensor|None = None) -> torch.Tensor:
                # """kwargs is not supported in torch.onnx.export""" with dynamo=True, so we need to handle optional inputs manually
                if vector_cond is not None and guidance is not None:
                    return self.diffuser(latent, timestep, context, vector_cond, guidance, transformer_options=self.transformer_options)
                elif vector_cond is not None:
                    return self.diffuser(latent, timestep, context, vector_cond, transformer_options=self.transformer_options)
                elif guidance is not None:
                    return self.diffuser(latent, timestep, context, guidance=guidance, transformer_options=self.transformer_options)
                else:
                    return self.diffuser(latent, timestep, context, transformer_options=self.transformer_options)

        return TracingModel(diffuser, transformer_options).eval()

@no_type_check
def build_tensorrt_engine(
    output_onnx: str,
    input_names: list[str],
    inputs_shapes_min: tuple[tuple[int, ...], ...],
    inputs_shapes_opt: tuple[tuple[int, ...], ...],
    inputs_shapes_max: tuple[tuple[int, ...], ...],
    output_path: str,
) -> tuple[str, str, str]:
    if trt is None:
        raise RuntimeError("TensorRT is not installed but required for compilation.")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    trt.init_libnvinfer_plugins(logger, "")

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(output_onnx)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        raise RuntimeError("ONNX parse ERROR")

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    
    timing_cache_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "timing_cache.trt"))
    buffer = b""
    if os.path.exists(timing_cache_path):
        with open(timing_cache_path, mode="rb") as timing_cache_file:
            buffer = timing_cache_file.read()
    timing_cache = config.create_timing_cache(buffer)
    config.set_timing_cache(timing_cache, ignore_mismatch=True)
    
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 * 1024 * 1024)

    for k in range(len(input_names)):
        profile.set_shape(input_names[k], inputs_shapes_min[k], inputs_shapes_opt[k], inputs_shapes_max[k])
        
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine building failed.")
    
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    with open(timing_cache_path, "wb") as timing_cache_file:
        timing_cache_file.write(memoryview(config.get_timing_cache().serialize()))

def export_anima_llmadapter_to_onnx(llm_adapter: LLMAdapter, output_path: str, dtype: torch.dtype):
    device = cast(torch.device, model_management.get_torch_device())
    tracing_model = llm_adapter.to(device).to(dtype).eval()

    batch = torch.export.Dim("batch")
    source_seq_len = torch.export.Dim("source_seq_len")
    target_seq_len = torch.export.Dim("target_seq_len")

    dynamic_shapes = (
        {0: batch, 1: source_seq_len}, # source_hidden_states
        {0: batch, 1: target_seq_len}, # target_input_ids
        {0: batch, 1: target_seq_len}, # target_attention_mask
        {0: batch, 1: source_seq_len}, # source_attention_mask
    )

    batch_s = 1 # batch
    S = 512 # source_seq_len
    T = 256 # target_seq_len

    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = (
        torch.randn(batch_s, S, 1024, device=device, dtype=dtype), # source_hidden_states
        torch.randint(0, 32128, (batch_s, T), device=device),      # target_input_ids (Long型)
        torch.ones(batch_s, T, device=device, dtype=dtype),        # target_attention_mask
        torch.ones(batch_s, S, device=device, dtype=dtype),        # source_attention_mask
    )
    # Anima LLMAdapter is supported in opset 25
    
    torch.onnx.export( # pyright: ignore[reportUnknownMemberType]
        tracing_model,
        inputs,
        output_path,
        input_names=["source_hidden_states", "target_input_ids", "target_attention_mask", "source_attention_mask"],
        output_names=["output"],
        opset_version=25,
        dynamic_shapes=dynamic_shapes,
    )
