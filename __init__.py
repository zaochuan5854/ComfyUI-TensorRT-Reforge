from comfy_api.latest import ComfyExtension
from comfy_api.latest import io

from .trt_exporter import TRTExporter
from .trt_loader import TRTLoader

class ComfyUITensorRTReforge(ComfyExtension):

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TRTExporter,
            TRTLoader
        ]

def comfy_entrypoint() -> ComfyUITensorRTReforge:
    return ComfyUITensorRTReforge()
