import numpy as np

from gguf_zoo.demo import (build_gguf_with_tensors, demo_bytes, demo_pairs,
                           demo_tensors)
from gguf_zoo.parser import parse_gguf_bytes
from gguf_zoo.render import build_description
from gguf_zoo.rules import Die, neural_facets
from gguf_zoo.weights import (F32, IQ4_XS, Q4_0, Q8_0, block_size, dequant,
                              neural_profile, parse_tensor_infos, quantize,
                              _stats)


def _full(buf, dtype, n_values):
    bs, nb = block_size(dtype)
    return np.concatenate([dequant(buf[i * nb:(i + 1) * nb], dtype)
                           for i in range(n_values // bs)])


def test_FR_19_1_tensor_infos_parsed():
    data = demo_bytes(with_tensors=True)
    kv, header = parse_gguf_bytes(data)
    assert header["n_tensors"] == len(demo_tensors())
    infos, data_start = parse_tensor_infos(data, header["kv_end"],
                                           header["n_tensors"])
    names = [i.name for i in infos]
    assert "token_embd.weight" in names and "blk.0.attn_q.weight" in names
    assert all(i.offset >= 0 for i in infos)
    assert data_start > header["kv_end"]


def test_FR_19_1_neural_profile_from_demo():
    data = demo_bytes(with_tensors=True)
    kv, header = parse_gguf_bytes(data)
    p = neural_profile(data, kv, header)
    assert p is not None and p.n_supported > 0


def test_FR_19_2_dequant_roundtrip():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(32 * 64) * 0.02
    base = _stats(_full(quantize(w, F32), F32, len(w)))
    for dt in (Q8_0, Q4_0):
        s = _stats(_full(quantize(w, dt), dt, len(w)))
        assert abs(s[0] - base[0]) / base[0] < 0.05          # RMS within 5%
        assert abs(s[2] - base[2]) < 0.5                     # kurtosis close


def test_FR_19_2_unsupported_dtype_skipped():
    tensors = [("blk.0.attn_q.weight", (256,), IQ4_XS, b"\0" * 130)]
    data = build_gguf_with_tensors(demo_pairs(), tensors)
    kv, header = parse_gguf_bytes(data)
    assert neural_profile(data, kv, header) is None


def test_FR_19_3_aggregates_finite():
    p = neural_profile(demo_bytes(with_tensors=True))
    for v in (p.spread, p.heterogeneity, p.outlier, p.sparsity, p.trend):
        assert np.isfinite(v)


def test_FR_19_4_facets_complete():
    p = neural_profile(demo_bytes(with_tensors=True))
    facets = neural_facets(p, Die("test"))
    assert set(facets) == {"range", "variety", "molt", "temper", "activity",
                           "indiv", "migration", "mating", "calls", "social",
                           "home_range", "shelter", "organ", "organ_fire",
                           "role"}
    assert all(isinstance(v, str) and v for v in facets.values())


def _quant_gguf(dtype):
    rng = np.random.default_rng(1)
    tensors = []
    for b in range(4):
        w = rng.standard_normal(128 * 128) * 0.015 * (0.9 ** b)
        tensors.append((f"blk.{b}.attn_q.weight", (128, 128), dtype,
                        quantize(w, dtype)))
    return build_gguf_with_tensors(demo_pairs(), tensors)


def test_FR_19_5_quant_levels_same_facets():
    facets = []
    for dt in (F32, Q8_0, Q4_0):
        facets.append(neural_facets(neural_profile(_quant_gguf(dt)),
                                    Die("same-seed")))
    assert facets[0] == facets[1] == facets[2]


def test_FR_19_6_in_depth_deterministic():
    data = demo_bytes(with_tensors=True)
    kv, header = parse_gguf_bytes(data)
    a = build_description(kv, header=header, device="cpu", in_depth=True,
                          source=data)
    b = build_description(kv, header=header, device="cpu", in_depth=True,
                          source=data)
    assert a == b
    for title in ("APPEARANCE VARIATION", "TEMPERAMENT", "SPECIES BEHAVIOR",
                  "NATURAL HABITAT", "ORGAN VARIATION"):
        assert title in a
    assert "nervous system" not in a


def test_FR_19_no_in_depth_unchanged():
    data = demo_bytes(with_tensors=True)
    kv, header = parse_gguf_bytes(data)
    out = build_description(kv, header=header, device="cpu")
    assert "APPEARANCE VARIATION" not in out
