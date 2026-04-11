class Operators:
    ATTENTION: str
    LAYERNORM: str
    MULTI_HEAD_ATTENTION: str
    PACKEDATTENTION: str
    PACKED_MULTI_HEAD_ATTENTION: str
    REMOVEPADDING: str
    RESTOREPADDING: str
    SKIPLAYERNORM: str

class AttentionInputIDs:
    INPUT: int
    WEIGHTS: int
    BIAS: int
    MASK_INDEX: int
    PAST: int
    ATTENTION_BIAS: int
    PAST_SEQUENCE_LENGTH: int

class AttentionOutputIDs:
    OUTPUT: int
    PRESENT: int

class MultiHeadAttentionInputIDs:
    QUERY: int
    KEY: int
    VALUE: int
    BIAS: int
    KEY_PADDING_MASK: int
    ATTENTION_BIAS: int
    PAST_KEY: int
    PAST_VALUE: int

class MultiHeadAttentionOutputIDs:
    OUTPUT: int
    PRESENT_KEY: int
    PRESENT_VALUE: int
