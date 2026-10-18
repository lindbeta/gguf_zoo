"""Synthetic GGUF builder (used by --demo and the test fixtures)."""
from __future__ import annotations

import struct

GGUF_MAGIC = b"GGUF"


def enc_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def enc_value(typecode: int, value) -> bytes:
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if typecode == 8:
        return enc_str(value)
    if typecode == 9:
        elem, items = value
        out = struct.pack("<I", elem) + struct.pack("<Q", len(items))
        return out + b"".join(enc_value(elem, it) for it in items)
    return struct.pack(fmt[typecode], value)


def enc_kv(key: str, typecode: int, value) -> bytes:
    return enc_str(key) + struct.pack("<I", typecode) + enc_value(typecode, value)


def build_gguf(pairs, version: int = 3, n_tensors: int = 0) -> bytes:
    body = b"".join(enc_kv(k, t, v) for k, t, v in pairs)
    return (GGUF_MAGIC + struct.pack("<I", version)
            + struct.pack("<Q", n_tensors) + struct.pack("<Q", len(pairs)) + body)


def demo_pairs() -> list[tuple[str, int, object]]:
    """A qwen2-1.5b-like multimodal model with a couple of unknown fields."""
    return [
        ("general.architecture", 8, "qwen2"),
        ("general.name", 8, "MysteryBeast Qwen2 Demo"),
        ("general.file_type", 4, 15),  # Q4_K_M
        ("general.language", 8, "en,zh,ja"),
        ("general.author", 8, "ZooKeeper"),
        ("qwen2.block_count", 4, 28),
        ("qwen2.embedding_length", 4, 1536),
        ("qwen2.feed_forward_length", 4, 8960),
        ("qwen2.vocab_length", 4, 151936),
        ("qwen2.context_length", 4, 32768),
        ("qwen2.attention.head_count", 4, 12),
        ("qwen2.attention.head_count_kv", 4, 2),
        ("qwen2.attention.key_length", 4, 128),
        ("qwen2.rope.freq_base", 12, 1000000.0),
        ("qwen2.rope.scaling.type", 8, "yarn"),
        ("tokenizer.ggml.pre", 8, "qwen2-vision"),
        ("tokenizer.ggml.tokens", 9, (8, ["<image>", "<|end|>"])),
        ("tokenizer.ggml.eos_token_id", 4, 151645),
        ("tokenizer.ggml.add_bos_token", 7, False),
        # unknown fields -> anomalies
        ("general.mystery_field", 4, 42),
        ("general.crystal_resonance_hz", 6, 4321.0),
    ]


def demo_pairs_moe() -> list[tuple[str, int, object]]:
    """A mixtral-8x7B-like pristine F32 multimodal MoE with no unknown fields."""
    return [
        ("general.architecture", 8, "mixtral"),
        ("general.name", 8, "Pristine Colossus Demo"),
        ("general.file_type", 4, 0),  # ALL_F32 -> pristine
        ("general.size_label", 8, "47B"),
        ("general.language", 8, "en,de,fr,es,it"),
        ("general.author", 8, "Warden"),
        ("mixtral.block_count", 4, 32),
        ("mixtral.embedding_length", 4, 4096),
        ("mixtral.feed_forward_length", 4, 14336),
        ("mixtral.vocab_length", 4, 32000),
        ("mixtral.context_length", 4, 32768),
        ("mixtral.attention.head_count", 4, 32),
        ("mixtral.attention.head_count_kv", 4, 8),
        ("mixtral.rope.freq_base", 12, 1000000.0),
        ("mixtral.expert_count", 4, 8),
        ("mixtral.expert_used_count", 4, 2),
        ("tokenizer.ggml.pre", 8, "llama-bpe audio-clip"),
        ("tokenizer.ggml.tokens", 9, (8, ["<audio>", "<image>", "<|end|>"])),
        ("tokenizer.ggml.eos_token_id", 4, 2),
        ("tokenizer.ggml.add_bos_token", 7, True),
    ]


DEMO_MODELS = {"qwen2": demo_pairs, "mixtral": demo_pairs_moe}


def demo_quant_names() -> list[str]:
    """Accepted --demo-quant values (canonical names from the quant table)."""
    from .rules import QUANT_TABLE
    return sorted({name for name, _band in QUANT_TABLE.values()} | {"F32"})


def resolve_demo_quant(value: str) -> int:
    """Case-insensitive quant name -> general.file_type code (FR-10.8)."""
    from .rules import QUANT_TABLE
    want = value.strip().upper()
    if want == "F32":
        want = "ALL_F32"
    for code, (name, _band) in QUANT_TABLE.items():
        if name.upper() == want:
            return code
    raise ValueError(
        f"unknown quant {value!r}; accepted: {', '.join(demo_quant_names())}")


def normalize_demo_params(value: str) -> str:
    """Full number or k/m/b/t suffix (case-insensitive) -> size_label string."""
    import re
    s = value.strip()
    if re.fullmatch(r"\d+", s):
        return s
    if re.fullmatch(r"\d+(\.\d+)?[kmbtKMBT]", s):
        return s.upper()
    raise ValueError(
        f"unknown parameter count {value!r}; use a full number (7155000000) "
        "or a k/m/b/t suffix (300m, 7b, 1.5T), case-insensitive")


def demo_pairs_custom(which: str = "qwen2", name: str | None = None,
                      params: str | None = None,
                      quant: int | None = None) -> list[tuple[str, int, object]]:
    """Demo metadata with optional name / size_label / file_type overrides."""
    out: list[tuple[str, int, object]] = []
    for k, t, v in DEMO_MODELS[which]():
        if name is not None and k == "general.name":
            v = name
        if quant is not None and k == "general.file_type":
            t, v = 4, quant
        if params is not None and k == "general.size_label":
            continue
        out.append((k, t, v))
    if params is not None:
        out.append(("general.size_label", 8, params))
    return out


def demo_tensors() -> list[tuple[str, tuple[int, ...], int, bytes]]:
    """Small deterministic tensors so --demo --in-depth has data (FR-19)."""
    import numpy as np
    from .weights import F32, Q4_0, quantize
    rng = np.random.default_rng(0)
    tensors = []

    def add(name, dims, dtype, values):
        tensors.append((name, dims, dtype, quantize(values, dtype)))

    add("token_embd.weight", (128, 256), F32,
        rng.standard_normal(128 * 256) * 0.02)
    for b in range(4):
        add(f"blk.{b}.attn_q.weight", (128, 128), Q4_0,
            rng.standard_normal(128 * 128) * 0.02)
        add(f"blk.{b}.ffn_up.weight", (128, 384), Q4_0,
            rng.standard_normal(128 * 384) * 0.02)
        add(f"blk.{b}.attn_norm.weight", (128,), F32,
            rng.standard_normal(128) * 0.1)
    add("output.weight", (128, 256), F32, rng.standard_normal(128 * 256) * 0.02)
    return tensors


def build_gguf_with_tensors(pairs, tensors, version: int = 3,
                            alignment: int = 32) -> bytes:
    out = bytearray(GGUF_MAGIC + struct.pack("<I", version)
                    + struct.pack("<Q", len(tensors))
                    + struct.pack("<Q", len(pairs))
                    + b"".join(enc_kv(k, t, v) for k, t, v in pairs))
    offset = 0
    for name, dims, dtype, data in tensors:
        out += enc_str(name) + struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", dtype) + struct.pack("<Q", offset)
        offset += len(data)
    out += b"\0" * ((-len(out)) % alignment)
    for _, _, _, data in tensors:
        out += data
    return bytes(out)


def demo_bytes(which: str = "qwen2", with_tensors: bool = False,
               name: str | None = None, params: str | None = None,
               quant: int | None = None) -> bytes:
    pairs = demo_pairs_custom(which, name, params, quant)
    if with_tensors:
        return build_gguf_with_tensors(pairs, demo_tensors())
    return build_gguf(pairs)
