import ort_flatbuffers_py.fbs as fbs
from _typeshed import Incomplete

class FbsTypeInfo:
    tensordatatype_to_string: Incomplete
    @staticmethod
    def typeinfo_to_str(type: fbs.TypeInfo): ...

def get_typeinfo(name: str, value_name_to_typeinfo: dict) -> fbs.TypeInfo: ...
def value_name_to_typestr(name: str, value_name_to_typeinfo: dict): ...
