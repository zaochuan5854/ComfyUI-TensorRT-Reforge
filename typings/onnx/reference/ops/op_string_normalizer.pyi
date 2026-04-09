from onnx.reference.op_run import OpRun as OpRun, RuntimeTypeError as RuntimeTypeError

class StringNormalizer(OpRun):
    @staticmethod
    def strip_accents_unicode(s): ...
