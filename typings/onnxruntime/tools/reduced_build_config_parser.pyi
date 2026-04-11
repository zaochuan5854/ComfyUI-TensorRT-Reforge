from .ort_format_model import GloballyAllowedTypesOpTypeImplFilter as GloballyAllowedTypesOpTypeImplFilter, OperatorTypeUsageManager as OperatorTypeUsageManager

have_flatbuffers: bool

def parse_config(config_file: str, enable_type_reduction: bool = False): ...
