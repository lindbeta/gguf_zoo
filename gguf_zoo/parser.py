"""GGUF v2/v3 header + tensor-info parser.

Parses only the metadata KV section and the tensor info table.
NEVER reads tensor weight data.
"""
import os
import re
import struct
from dataclasses import dataclass, field

GGUF_MAGIC = b"GGUF"

T_UINT8 = 0
T_INT8 = 1
T_UINT16 = 2
T_INT16 = 3
T_UINT32 = 4
T_INT32 = 5
T_FLOAT32 = 6
T_BOOL = 7
T_STRING = 8
T_ARRAY = 9
T_UINT64 = 10
T_INT64 = 11
T_FLOAT64 = 12

# llama.cpp general.file_type enum -> quant name (gguf-py FileType)
FILE_TYPE_NAMES = {
    0: "none", 1: "F32", 2: "F16", 3: "Q4_0", 4: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "Q8_K", 20: "IQ2_XXS", 21: "IQ2_XS", 22: "Q2_K_S", 23: "IQ3_XS",
    24: "IQ3_XXS", 25: "IQ1_S", 26: "IQ4_NL", 27: "IQ3_S", 28: "IQ3_M",
    29: "IQ2_S", 30: "IQ2_M", 31: "IQ4_XS", 32: "IQ1_M", 33: "BF16",
}

_SIMPLE_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}


@dataclass
class TensorInfo:
    name: str
    dims: list
    dtype: int
    offset: int


@dataclass
class ModelMeta:
    # structural
    architecture: str = ""
    name: str = ""
    base_model: str = ""
    language: str = ""
    quant_type: str = "unknown"
    quant_version: int = 0
    n_params: int = 0
    context_length: int = 0
    n_layers: int = 0
    d_model: int = 0
    d_ff: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    is_moe: bool = False
    is_multimodal: bool = False
    has_vision: bool = False
    has_audio: bool = False
    sliding_window: bool = False
    n_tensors: int = 0
    # cosmetic layer
    quant_block_style: str = "unknown"  # K-quant | legacy | lossless | unknown
    mixed_dtypes: bool = False
    has_ssm: bool = False
    rope_freq_base: float = 10000.0
    rope_scaling_type: str = "none"  # none|linear|dynamic|yarn
    partial_rope: bool = False
    mtp_tensors: bool = False
    n_special_tokens: int = 0
    add_bos: bool = False
    add_eos: bool = False
    byte_fallback: bool = False
    pooling_type: str = ""
    is_encoder_decoder: bool = False
    n_split_files: int = 1
    rms_epsilon: float = 1e-5
    # raw kv for debugging
    kv: dict = field(default_factory=dict, repr=False)


class _Reader:
    def __init__(self, f):
        self.f = f

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError(f"unexpected EOF reading {n} bytes")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        if n > 1 << 30:
            raise ValueError("implausible string length")
        return self.read(n).decode("utf-8", "replace")

    def value(self, vtype, depth=0):
        if depth > 4:
            raise ValueError("array nesting too deep")
        if vtype in _SIMPLE_FMT:
            return struct.unpack(_SIMPLE_FMT[vtype], self.read(struct.calcsize(_SIMPLE_FMT[vtype])))[0]
        if vtype == T_BOOL:
            return self.read(1)[0] != 0
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            if count > 1 << 26:
                raise ValueError("implausible array count")
            return [self.value(elem_type, depth + 1) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {vtype}")


def classify_block_style(quant: str) -> str:
    q = (quant or "").upper()
    if q in ("F16", "F32", "BF16", "F64"):
        return "lossless"
    if "_K" in q or q in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"):
        return "K-quant"
    if q.startswith("Q") or q.startswith("IQ"):
        return "legacy"
    return "unknown"


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def build_meta(kv: dict, tensors: list, filename: str = "") -> ModelMeta:
    """Build ModelMeta from a parsed kv dict + tensor info list."""
    m = ModelMeta(kv=kv)
    arch = kv.get("general.architecture", "")
    if isinstance(arch, list):
        arch = arch[0] if arch else ""
    m.architecture = str(arch).lower()
    a = m.architecture
    m.name = str(kv.get("general.name", ""))
    m.base_model = str(kv.get("general.base_model", ""))
    lang = kv.get("general.language", "")
    if isinstance(lang, list):
        lang = ",".join(str(x) for x in lang)
    m.language = str(lang).lower()

    ft = kv.get("general.file_type", None)
    if isinstance(ft, int) and ft in FILE_TYPE_NAMES:
        m.quant_type = FILE_TYPE_NAMES[ft]
    elif isinstance(ft, str) and ft:
        m.quant_type = ft
    else:
        m.quant_type = str(kv.get("general.quant_type", "unknown"))
    m.quant_version = int(kv.get("general.quantization_version", 0) or 0)
    m.quant_block_style = classify_block_style(m.quant_type)

    if "general.parameters.total" in kv:
        try:
            m.n_params = int(float(kv["general.parameters.total"]))
        except (TypeError, ValueError):
            m.n_params = 0
    if m.n_params <= 0 and tensors:
        total = 0
        for t in tensors:
            p = 1
            for d in t.dims:
                p *= max(int(d), 1)
            total += p
        m.n_params = total

    def _int(key, default=0):
        v = kv.get(key, default)
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    m.context_length = _int(f"{a}.context_length")
    m.n_layers = _int(f"{a}.block_count")
    m.d_model = _int(f"{a}.embedding_length")
    m.d_ff = _int(f"{a}.feed_forward_length")
    m.n_heads = _int(f"{a}.attention.head_count")
    kvh = kv.get(f"{a}.attention.head_count_kv", 0)
    if isinstance(kvh, list):
        kvh = kvh[0] if kvh else 0
    m.n_kv_heads = _int_val(kvh)
    m.head_dim = _int(f"{a}.attention.head_dim")
    if m.head_dim <= 0 and m.n_heads > 0 and m.d_model > 0:
        m.head_dim = m.d_model // m.n_heads

    tokens = kv.get("tokenizer.ggml.tokens")
    if isinstance(tokens, list):
        m.vocab_size = len(tokens)
    else:
        m.vocab_size = _int("tokenizer.ggml.tokens_length") or _int(f"{a}.vocab_size")

    names = [t.name for t in tensors]
    keys = list(kv.keys())
    lower_names = [n.lower() for n in names]

    m.is_moe = any(
        s in n for n in lower_names
        for s in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", ".experts.")
    )
    n_exp = kv.get(f"{a}.expert_count") or kv.get("general.n_experts") or 0
    try:
        if int(n_exp) > 1:
            m.is_moe = True
    except (TypeError, ValueError):
        pass

    m.has_vision = any(
        n.startswith("v.") or "mmproj" in n or "vision" in n or n.startswith("clip.") or "visual" in n
        for n in lower_names
    ) or any(k.lower().startswith(("clip.", "mmproj.")) or "vision" in k.lower() for k in keys)
    m.has_audio = any(
        "audio" in n or "whisper" in n or n.startswith("audio.") for n in lower_names
    ) or any("audio" in k.lower() for k in keys)
    m.is_multimodal = m.has_vision or m.has_audio

    atype = _as_list(kv.get(f"{a}.attention.type", []))
    layer_types = _as_list(kv.get(f"{a}.layer_types", []))
    lt = [str(x).lower() for x in layer_types]
    m.sliding_window = (
        any("sliding" in str(x).lower() for x in atype)
        or f"{a}.attention.sliding_window" in kv
        or any("swa" in x or "sliding" in x for x in lt)
    )
    m.has_ssm = (
        any(s in x for x in lt for s in ("mamba", "ssm", "conv"))
        or any(s in n for n in lower_names for s in ("mamba", "conv1d", "a_log", "dt_bias", "dt_proj"))
    )

    rb = kv.get(f"{a}.attention.rope.freq_base", kv.get("attention.rope.freq_base"))
    if rb is not None:
        try:
            m.rope_freq_base = float(rb)
        except (TypeError, ValueError):
            pass
    rs = kv.get(f"{a}.rope.scaling.type", kv.get("rope.scaling.type", "none"))
    rs = str(rs).lower() if rs else "none"
    m.rope_scaling_type = rs if rs in ("linear", "dynamic", "yarn") else "none"
    m.partial_rope = (
        f"{a}.rope.dimension_sections" in kv
        or "partial" in str(kv.get(f"{a}.rope.type", "")).lower()
    )
    m.mtp_tensors = any(
        n.startswith(("mtp.", "next_n.", "eagle.")) or ".mtp." in n for n in lower_names
    )

    tt = kv.get("tokenizer.ggml.token_type")
    if isinstance(tt, list):
        m.n_special_tokens = sum(1 for x in tt if int(x) == 3)
    m.add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", False))
    m.add_eos = bool(kv.get("tokenizer.ggml.add_eos_token", False))
    m.byte_fallback = bool(kv.get("tokenizer.ggml.byte_fallback", False))
    m.pooling_type = str(kv.get(f"{a}.pooling_type", kv.get("pooling_type", "")) or "")
    m.is_encoder_decoder = a in ("t5", "bart", "mt5", "t5gemma", "encoder-decoder") or any(
        "cross_attn" in n or "cross-attention" in n for n in lower_names
    )
    match = re.search(r"-(\d{5})-(\d{5})", filename or "")
    if match:
        m.n_split_files = int(match.group(2))
    eps = kv.get(f"{a}.attention.layer_norm_rms_epsilon")
    if eps is not None:
        try:
            m.rms_epsilon = float(eps)
        except (TypeError, ValueError):
            pass
    m.n_tensors = len(tensors)
    m.mixed_dtypes = len({t.dtype for t in tensors}) > 1
    return m


def _int_val(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def parse_gguf(path: str) -> ModelMeta:
    """Parse GGUF header + tensor info table. Never touches weight data."""
    tensors = []
    kv = {}
    with open(path, "rb") as f:
        r = _Reader(f)
        magic = r.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file (magic={magic!r})")
        version = r.u32()
        if version >= 3:
            n_tensors = r.u64()
            n_kv = r.u64()
        elif version >= 2:
            n_tensors = r.u32()
            n_kv = r.u32()
        else:
            raise ValueError(f"unsupported GGUF version {version}")
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            kv[key] = r.value(vtype)
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            dtype = r.u32()
            offset = r.u64()
            tensors.append(TensorInfo(name, dims, dtype, offset))
        # stop here: weight data is never read
    meta = build_meta(kv, tensors, os.path.basename(path))
    if meta.n_tensors == 0:
        meta.n_tensors = int(kv.get("general.tensor_count", 0) or 0)
    return meta
