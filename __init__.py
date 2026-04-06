from .trt_exporter import TRTExporter
from .trt_loader import TRTLoader

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "TensorRTExporterNode": TRTExporter,
    "TensorRTLoaderNode": TRTLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TensorRTExporterNode": "TensorRT Exporter Reforge",
    "TensorRTLoaderNode": "TensorRT Loader Reforge"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
