from _typeshed import Incomplete

logger: Incomplete

def conv1d_to_linear(model) -> None: ...

class QuantizeHelper:
    @staticmethod
    def quantize_torch_model(model, dtype=...): ...
    @staticmethod
    def quantize_onnx_model(onnx_model_path, quantized_model_path, use_external_data_format: bool = False) -> None: ...
