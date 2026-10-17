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


def demo_bytes() -> bytes:
    return build_gguf(demo_pairs())
