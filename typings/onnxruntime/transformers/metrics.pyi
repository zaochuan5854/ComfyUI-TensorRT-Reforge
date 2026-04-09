from _typeshed import Incomplete

class BaseObject:
    customized: Incomplete
    def __init__(self) -> None: ...
    def to_dict(self): ...

class ModelInfo(BaseObject):
    full_name: Incomplete
    is_huggingface: Incomplete
    is_text_generation: Incomplete
    short_name: Incomplete
    input_shape: Incomplete
    def __init__(self, full_name: str | None = None, is_huggingface: bool | None = False, is_text_generation: bool | None = False, short_name: str | None = None) -> None: ...

class BackendOptions(BaseObject):
    enable_profiling: Incomplete
    execution_provider: Incomplete
    use_io_binding: Incomplete
    def __init__(self, enable_profiling: bool | None = False, execution_provider: str | None = None, use_io_binding: bool | None = False) -> None: ...

class Config(BaseObject):
    backend: Incomplete
    batch_size: Incomplete
    seq_length: Incomplete
    precision: Incomplete
    warmup_runs: Incomplete
    measured_runs: Incomplete
    model_info: Incomplete
    backend_options: Incomplete
    def __init__(self, backend: str | None = 'onnxruntime', batch_size: int | None = 1, seq_length: int | None = 0, precision: str | None = 'fp32', warmup_runs: int | None = 1, measured_runs: int | None = 10) -> None: ...

class Metadata(BaseObject):
    device: Incomplete
    package_name: Incomplete
    package_version: Incomplete
    platform: Incomplete
    python_version: Incomplete
    def __init__(self, device: str | None = None, package_name: str | None = None, package_version: str | None = None, platform: str | None = None, python_version: str | None = None) -> None: ...

class Metrics(BaseObject):
    latency_ms_mean: Incomplete
    throughput_qps: Incomplete
    max_memory_usage_GB: Incomplete
    def __init__(self, latency_ms_mean: float | None = 0.0, throughput_qps: float | None = 0.0, max_memory_usage_GB: float | None = 0.0) -> None: ...

class BenchmarkRecord:
    config: Incomplete
    metrics: Incomplete
    metadata: Incomplete
    trigger_date: Incomplete
    def __init__(self, model_name: str, precision: str, backend: str, device: str, package_name: str, package_version: str, batch_size: int | None = 1, warmup_runs: int | None = 1, measured_runs: int | None = 10, trigger_date: str | None = None) -> None: ...
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
    @classmethod
    def save_as_csv(cls, file_name: str, records: list) -> None: ...
    @classmethod
    def save_as_json(cls, file_name: str, records: list) -> None: ...
