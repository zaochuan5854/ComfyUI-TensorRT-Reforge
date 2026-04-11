from onnx import AttributeProto as AttributeProto, FunctionProto as FunctionProto, GraphProto as GraphProto, ModelProto as ModelProto, NodeProto as NodeProto, SparseTensorProto as SparseTensorProto, TensorProto as TensorProto
from onnx.helper import make_attribute as make_attribute, make_function as make_function, make_graph as make_graph, make_model as make_model, make_node as make_node, make_tensor as make_tensor, make_tensor_value_info as make_tensor_value_info, set_model_props as set_model_props, tensor_dtype_to_np_dtype as tensor_dtype_to_np_dtype
from onnx.numpy_helper import from_array as from_array

def replace_initializer_by_constant_of_shape(onx: FunctionProto | GraphProto | ModelProto, threshold: int = 128, ir_version: int | None = None, use_range: bool = False, value_constant_of_shape: float = 0.5): ...
