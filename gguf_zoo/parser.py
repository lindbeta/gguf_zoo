"""GGUF header parser (metadata only) + ModelMeta normalization.

Reads ONLY the KV section; never touches tensor info or tensor data.
All integers little-endian. See SPEC.md §0.1 and FR-1.
"""
from __future__ import annotations

import mmap
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"

# type codes
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL = 0, 1, 2, 3, 4, 5, 6, 7
STRING, ARRAY, UINT64, INT64, FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FMT = {
    UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h",
    UINT32: "<I", INT32: "<i", FLOAT32: "<f", BOOL: "<?",
    UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}

_OPAQUE_ARRAY_LIMIT = 64


class GGUFError(Exception):
    pass


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise GGUFError("Truncated GGUF file")
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def unpack(self, fmt: str) -> Any:
        try:
            size = struct.calcsize(fmt)
            return struct.unpack(fmt, self.read(size))[0]
        except struct.error as exc:  # pragma: no cover - defensive
            raise GGUFError(f"Truncated GGUF file: {exc}") from exc

    def read_str(self) -> str:
        length = self.unpack("<Q")
        return self.read(length).decode("utf-8", errors="replace")


def _skip_elements(r: _Reader, elem_type: int, count: int) -> None:
    if elem_type in _SCALAR_FMT:
        r.read(count * struct.calcsize(_SCALAR_FMT[elem_type]))
    elif elem_type == STRING:
        for _ in range(count):
            r.read(r.unpack("<Q"))
    elif elem_type == ARRAY:
        for _ in range(count):
            _skip_value(r, ARRAY)
    else:
        raise GGUFError(f"Unknown GGUF type code {elem_type}")


def _skip_value(r: _Reader, typecode: int) -> None:
    if typecode == STRING:
        r.read(r.unpack("<Q"))
    elif typecode == ARRAY:
        elem_type = r.unpack("<I")
        count = r.unpack("<Q")
        _skip_elements(r, elem_type, count)
    elif typecode in _SCALAR_FMT:
        r.read(struct.calcsize(_SCALAR_FMT[typecode]))
    else:
        raise GGUFError(f"Unknown GGUF type code {typecode}")


def _read_value(r: _Reader, typecode: int) -> Any:
    if typecode == STRING:
        return r.read_str()
    if typecode == ARRAY:
        elem_type = r.unpack("<I")
        count = r.unpack("<Q")
        if count > _OPAQUE_ARRAY_LIMIT:
            marker = f"<array:{elem_type}:{count}>"
            _skip_elements(r, elem_type, count)
            return marker
        return [_read_value(r, elem_type) for _ in range(count)]
    if typecode in _SCALAR_FMT:
        return r.unpack(_SCALAR_FMT[typecode])
    raise GGUFError(f"Unknown GGUF type code {typecode}")


def parse_gguf_bytes(data: bytes) -> tuple[dict[str, Any], dict[str, int]]:
    """Parse the KV section only. Returns (kv_dict, header_info)."""
    r = _Reader(data)
    if r.read(4) != GGUF_MAGIC:
        raise GGUFError("Not a GGUF file (bad magic)")
    version = r.unpack("<I")
    if version < 2:
        raise GGUFError(f"Unsupported GGUF version {version} (need >= 2)")
    n_tensors = r.unpack("<Q")
    n_kv = r.unpack("<Q")
    kv: dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.read_str()
        typecode = r.unpack("<I")
        kv[key] = _read_value(r, typecode)
    header = {"version": version, "n_tensors": n_tensors, "n_kv": n_kv}
    return kv, header


def read_metadata(path: str | Path) -> tuple[dict[str, Any], dict[str, int]]:
    p = Path(path)
    try:
        with p.open("rb") as fh:
            if fh.seek(0, 2) == 0:
                raise GGUFError("Truncated GGUF file")
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return parse_gguf_bytes(mm)
    except OSError as exc:
        raise GGUFError(f"Cannot read file: {exc}") from exc


# ---------------------------------------------------------------------------
# ModelMeta
# ---------------------------------------------------------------------------
@dataclass
class ModelMeta:
    kv: dict[str, Any] = field(default_factory=dict)
    arch: str | None = None
    name: str | None = None
    file_type: int | str | None = None
    n_layers: int | None = None
    n_embd: int | None = None
    n_vocab: int | None = None
    d_ff: int | None = None
    n_expert: int | None = None
    n_expert_used: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    head_dim: int | None = None
    context_length: int | None = None
    rope_freq_base: float | None = None
    rope_scaling_type: str | None = None
    partial_rope: bool = False
    mtp_tensors: bool = False
    pooling_type: str | None = None
    is_encoder_decoder: bool = False
    add_bos: bool = False
    add_eos: bool = False
    has_ssm: bool = False
    is_moe: bool = False
    sliding_window: bool = False
    byte_fallback: bool = False
    language: str | None = None
    author: str | None = None
    base_model: str | None = None
    size_label: str | None = None
    year: int | None = None
    n_tensors: int = 0
    input_types: list[str] = field(default_factory=list)
    unknown: dict[str, Any] = field(default_factory=dict)


_ARCH_RE = re.compile(r"^general\.architecture\.(?P<arch>[^.]+)\.(?P<field>.+)$")

_ARCH_FIELD_MAP = {
    "block_length": "n_layers",
    "block_count": "n_layers",
    "embedding_length": "n_embd",
    "vocab_length": "n_vocab",
    "vocab_size": "n_vocab",
    "feed_forward_length": "d_ff",
    "expert_count": "n_expert",
    "expert_used_count": "n_expert_used",
    "attention.head_count": "n_heads",
    "attention.head_count_kv": "n_kv_heads",
    "attention.key_length": "head_dim",
    "rope.freq_base": "rope_freq_base",
    "rope.scaling.type": "rope_scaling_type",
    "context_length": "context_length",
}


def _as_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_meta(kv: dict[str, Any], year: int | None = None) -> ModelMeta:
    m = ModelMeta(kv=dict(kv), year=year)
    m.arch = kv.get("general.architecture")
    m.name = kv.get("general.name")
    m.file_type = kv.get("general.file_type")
    m.language = kv.get("general.language")
    m.author = kv.get("general.author")
    m.base_model = kv.get("general.base_model")
    m.add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", False))
    m.add_eos = "tokenizer.ggml.eos_token_id" in kv
    m.size_label = kv.get("general.size_label")

    arch_re = (re.compile(r"^" + re.escape(m.arch) + r"\.(?P<field>.+)$")
               if m.arch else None)
    for key, value in kv.items():
        mo = _ARCH_RE.match(key) or (arch_re.match(key) if arch_re else None)
        if mo:
            field_name = _ARCH_FIELD_MAP.get(mo.group("field"))
            if field_name == "rope_freq_base":
                setattr(m, field_name, _as_float(value))
            elif field_name:
                setattr(m, field_name, _as_int(value))
            if mo.group("field") == "rope.scaling.type" and "partial" in str(value).lower():
                m.partial_rope = True
            continue
        low = key.lower()
        if "partial_rope" in low:
            m.partial_rope = bool(value)
        if "mtp" in low:
            m.mtp_tensors = True
        if "pooling" in low:
            m.pooling_type = str(value)
        if "encoder_decoder" in low or "enc_dec" in low:
            m.is_encoder_decoder = bool(value)
        if "sliding_window" in low:
            m.sliding_window = bool(value)
        if "byte_fallback" in low:
            m.byte_fallback = bool(value)
        if "ssm" in low or "mamba" in low:
            m.has_ssm = True

    arch_l = (m.arch or "").lower()
    if "mamba" in arch_l:
        m.has_ssm = True
    if "moe" in arch_l or "mixtral" in arch_l or m.n_expert:
        m.is_moe = True
    if arch_l in ("t5", "enc-dec", "encoder-decoder"):
        m.is_encoder_decoder = True

    if m.n_vocab is None:
        tokens = kv.get("tokenizer.ggml.tokens")
        if isinstance(tokens, list):
            m.n_vocab = len(tokens)
        elif isinstance(tokens, str):
            mo = re.match(r"<array:\d+:(\d+)>", tokens)
            if mo:
                m.n_vocab = int(mo.group(1))

    from .senses import detect_input_types
    m.input_types = detect_input_types(m)
    return m
