import pydot
from _typeshed import Incomplete
from onnx import GraphProto as GraphProto, ModelProto as ModelProto, NodeProto as NodeProto
from typing import Any

OP_STYLE: Incomplete
BLOB_STYLE: Incomplete

def GetOpNodeProducer(embed_docstring: bool = False, **kwargs: Any) -> _NodeProducer: ...
def GetPydotGraph(graph: GraphProto, name: str | None = None, rankdir: str = 'LR', node_producer: _NodeProducer | None = None, embed_docstring: bool = False) -> pydot.Dot: ...
def main() -> None: ...
