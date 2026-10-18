import struct

import pytest

from gguf_zoo.demo import build_gguf, enc_kv, enc_str
from gguf_zoo.parser import GGUFError, build_meta, parse_gguf_bytes, read_metadata


def test_FR_1_1_bad_magic():
    with pytest.raises(GGUFError, match="bad magic"):
        parse_gguf_bytes(b"XXXX" + b"\x00" * 32)


def test_FR_1_2_version_too_old():
    data = b"GGUF" + struct.pack("<I", 1) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    with pytest.raises(GGUFError):
        parse_gguf_bytes(data)


def test_FR_1_3_all_types_roundtrip():
    pairs = [
        ("a", 0, 1), ("b", 1, -1), ("c", 2, 2), ("d", 3, -2), ("e", 4, 3),
        ("f", 5, -3), ("g", 6, 1.5), ("h", 7, True), ("i", 8, "hi"),
        ("j", 10, 10**12), ("k", 11, -10**12), ("l", 12, 3.25),
        ("m", 9, (8, ["x", "y"])), ("n", 9, (4, [1, 2, 3])),
    ]
    kv, _ = parse_gguf_bytes(build_gguf(pairs))
    assert kv["i"] == "hi" and kv["h"] is True and kv["l"] == 3.25
    assert kv["m"] == ["x", "y"] and kv["n"] == [1, 2, 3]


def test_FR_1_4_empty_file():
    kv, header = parse_gguf_bytes(build_gguf([]))
    assert kv == {} and header == {"version": 3, "n_tensors": 0, "n_kv": 0,
                                   "kv_end": 24}


def test_FR_1_5_truncated():
    data = build_gguf([("k", 8, "value")])
    with pytest.raises(GGUFError):
        parse_gguf_bytes(data[:-3])


def test_FR_1_6_bad_utf8_replaced():
    data = build_gguf([("k", 8, "ok")])
    # corrupt the string bytes to invalid utf-8
    idx = data.index(b"ok")
    data = data[:idx] + b"\xff\xfe" + data[idx + 2:]
    kv, _ = parse_gguf_bytes(data)
    assert isinstance(kv["k"], str)


def test_FR_1_7_header_info_fields():
    _, header = parse_gguf_bytes(build_gguf([("k", 4, 1)], n_tensors=7))
    assert set(header) == {"version", "n_tensors", "n_kv", "kv_end"}
    assert header["n_tensors"] == 7


def test_FR_1_8_build_meta_from_flat_dict():
    m = build_meta({"general.architecture": "llama", "general.name": "X",
                    "general.architecture.llama.block_length": 32})
    assert m.arch == "llama" and m.n_layers == 32


def test_FR_1_read_metadata_missing_file(tmp_path):
    with pytest.raises(GGUFError):
        read_metadata(tmp_path / "nope.gguf")
