"""Build a realistic synthetic Qwen2-1.5B-style GGUF header for demos/tests."""
import struct
import sys

sys.path.insert(0, "tests")
from test_gguf_zoo import _enc_str, _kv_str, _kv_u32, _kv_bool, _kv_str_array, _kv_i32_array, _tensor


def build_qwen_gguf(path, vision=True):
    vocab = 151936
    d, ffn, layers = 1536, 8960, 28
    kvs = [
        _kv_str("general.architecture", "qwen2"),
        _kv_str("general.name", "Qwen2-1.5B-demo"),
        _kv_str("general.language", "zh,en"),
        _kv_u32("general.file_type", 15),          # Q4_K_M
        _kv_u32("general.quantization_version", 3),
        _kv_u32("qwen2.context_length", 32768),
        _kv_u32("qwen2.block_count", layers),
        _kv_u32("qwen2.embedding_length", d),
        _kv_u32("qwen2.feed_forward_length", ffn),
        _kv_u32("qwen2.attention.head_count", 12),
        _kv_i32_array("qwen2.attention.head_count_kv", [2]),
        _kv_u32("qwen2.attention.rope.freq_base", 1000000),
        _kv_str("qwen2.rope.scaling.type", "yarn"),
        _kv_str_array("tokenizer.ggml.tokens", [f"tok{i}" for i in range(vocab)]),
        _kv_i32_array("tokenizer.ggml.token_type",
                      [3 if i in (0, 1, 2) else 1 for i in range(vocab)]),
        _kv_bool("tokenizer.ggml.add_bos_token", False),
        _kv_bool("tokenizer.ggml.add_eos_token", True),
    ]
    tensors = [_tensor("token_embd.weight", [d, vocab], 12)]
    for b in range(layers):
        p = f"blk.{b}."
        tensors += [
            _tensor(p + "attn_q.weight", [d, d], 18),
            _tensor(p + "attn_k.weight", [256, d], 18),
            _tensor(p + "attn_v.weight", [256, d], 18),
            _tensor(p + "attn_o.weight", [d, d], 18),
            _tensor(p + "ffn_gate.weight", [ffn, d], 18),
            _tensor(p + "ffn_up.weight", [ffn, d], 18),
            _tensor(p + "ffn_down.weight", [d, ffn], 18),
            _tensor(p + "attn_norm.weight", [d], 8),
            _tensor(p + "ffn_norm.weight", [d], 8),
        ]
    if vision:
        tensors += [_tensor("v.blk.0.attn_q.weight", [d, d], 12),
                    _tensor("v.mmproj_w.weight", [d, d], 12)]
    tensors += [_tensor("output_norm.weight", [d], 8),
                _tensor("output.weight", [d, vocab], 12)]
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))
        fh.write(struct.pack("<Q", len(tensors)))
        fh.write(struct.pack("<Q", len(kvs)))
        for kv in kvs:
            fh.write(kv)
        for t in tensors:
            fh.write(t)
        fh.write(b"\x00" * 64)  # fake weight data, never read
    return path


if __name__ == "__main__":
    print(build_qwen_gguf("examples/qwen2-1.5b-demo.gguf"))
