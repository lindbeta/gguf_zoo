"""Synthetic data generation: sample records, label with the teacher, save npz."""
import math
import os

import numpy as np

from .parser import ModelMeta, classify_block_style
from .rules import teacher, BODY_SIZE_EDGES
from .model import encode_features

ARCH_FAMILIES = [
    # name, p, dff_ratio_range, rope_bases  (ratios match real releases)
    ("llama", 0.18, (3.3, 3.7), [500000, 10000]),
    ("qwen2", 0.18, (5.0, 6.0), [1000000, 100000]),
    ("mistral", 0.12, (3.3, 3.7), [1000000, 10000]),
    ("mixtral", 0.06, (3.3, 3.7), [1000000, 10000]),
    ("gemma2", 0.10, (2.5, 3.2), [10000]),
    ("phi3", 0.08, (2.3, 2.8), [10000, 1000000]),
    ("deepseek2", 0.08, (1.9, 2.3), [10000]),
    ("mamba", 0.06, (2.5, 3.5), [10000]),
    ("falcon", 0.05, (3.8, 4.2), [100000, 10000]),
    ("command-r", 0.04, (3.2, 3.6), [10000]),
    ("gpt2", 0.03, (3.8, 4.2), [10000, 5000]),
    ("t5", 0.02, (3.8, 4.2), [10000]),
]
ARCH_P = np.array([a[1] for a in ARCH_FAMILIES], dtype=np.float64)
ARCH_P /= ARCH_P.sum()

QUANT_K = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q3_K_M", "Q3_K_L", "Q2_K"]
QUANT_LEGACY = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"]
QUANT_LOSSLESS = ["F16", "BF16", "F32"]

CONTEXTS = [2048, 4096, 8192, 32768, 131072, 262144]
CONTEXT_P = np.array([5, 15, 25, 30, 15, 10], dtype=np.float64)
CONTEXT_P /= CONTEXT_P.sum()

# vocab bands for the 6-class diet ladder: low (<32K), mid (32K-160K),
# high (200K+). Sampled correlated with language count: multilingual models
# carry big vocabularies because they cover many scripts.
VOCAB_LOW = [8192, 12000, 16384]
VOCAB_MID = [32000, 50257, 65536, 100352, 128256, 151936]
VOCAB_HIGH = [200000, 262144, 400000, 512000]
VOCAB_HIGH_P = [0.35, 0.35, 0.15, 0.15]  # ~20% in 200-300K, ~10% above 400K
LANG_CODES = ["en", "zh", "ja", "ko", "ru", "ar", "es", "de", "fr", "fi",
              "sv", "he"]
N_LANG_DRAWS = [1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 6, 10]


def _sample_language_and_vocab(rng):
    """Draw n_languages first, then a vocab band whose probability shifts
    upward with language count (diet_breadth and habitat_tint co-vary)."""
    n_lang = int(rng.choice(N_LANG_DRAWS))
    if n_lang == 1:
        band_p = (0.35, 0.55, 0.10)
    elif n_lang <= 3:
        band_p = (0.15, 0.60, 0.25)
    else:
        band_p = (0.05, 0.40, 0.55)
    band = int(rng.choice(3, p=band_p))
    if band == 2:
        vocab = int(rng.choice(VOCAB_HIGH, p=VOCAB_HIGH_P))
    else:
        vocab = int(rng.choice([VOCAB_LOW, VOCAB_MID][band]))
    if n_lang == 1:
        lang = str(rng.choice(LANG_CODES))
    elif n_lang <= 3:
        idx = rng.choice(len(LANG_CODES), size=n_lang, replace=False)
        lang = ",".join(LANG_CODES[i] for i in idx)
    else:
        lang = "multi"
    return lang, vocab


def _params_band(rng, cls):
    lo = 1e4 if cls == 0 else BODY_SIZE_EDGES[cls - 1]
    hi = BODY_SIZE_EDGES[cls] if cls < 11 else 1e14
    lo = max(lo, 1e4)
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))


def _dims_for_params(rng, n_params, family, ratio_range):
    if n_params < 1e9:
        n_layers = int(rng.integers(6, 29))
        d_model = int(rng.choice([512, 768, 896, 1024, 1536, 2048]))
    elif n_params < 1e10:
        n_layers = int(rng.integers(18, 49))
        d_model = int(rng.choice([2048, 2560, 3072, 3584, 4096]))
    elif n_params < 1e11:
        n_layers = int(rng.integers(28, 65))
        d_model = int(rng.choice([4096, 5120, 6144, 7168, 8192]))
    elif n_params < 1e12:
        n_layers = int(rng.integers(40, 81))
        d_model = int(rng.choice([7168, 8192, 12288]))
    else:
        n_layers = int(rng.integers(48, 129))
        d_model = int(rng.choice([8192, 12288, 16384]))
    ratio = rng.uniform(*ratio_range)
    d_ff = max(int(round(d_model * ratio / 128) * 128), d_model)
    head_dim = int(rng.choice([128, 128, 128, 128, 128, 128, 64, 64, 96, 80, 256]))
    n_heads = max(4, int(round(d_model / head_dim * rng.uniform(0.75, 1.0))))
    r = rng.random()
    if r < 0.05:
        n_kv = 1
    elif r < 0.35:
        n_kv = max(1, n_heads // 8)
    elif r < 0.65:
        n_kv = max(1, n_heads // 4)
    elif r < 0.85:
        n_kv = max(1, n_heads // 2)
    else:
        n_kv = n_heads
    return n_layers, d_model, d_ff, n_heads, n_kv, head_dim


def _sample_record(rng):
    fam_idx = int(rng.choice(len(ARCH_FAMILIES), p=ARCH_P))
    family, _, ratio_range, rope_bases = ARCH_FAMILIES[fam_idx]

    if rng.random() < 0.30:
        n_params = _params_band(rng, int(rng.integers(0, 12)))
    else:
        n_params = 10 ** rng.uniform(7.0, math.log10(5e12))

    n_layers, d_model, d_ff, n_heads, n_kv, head_dim = _dims_for_params(
        rng, n_params, family, ratio_range)

    r = rng.random()
    if r < 0.70:
        quant = str(rng.choice(QUANT_K))
    elif r < 0.85:
        quant = str(rng.choice(QUANT_LEGACY))
    else:
        quant = str(rng.choice(QUANT_LOSSLESS))

    is_moe = family in ("mixtral", "deepseek2") or rng.random() < 0.07
    has_ssm = family == "mamba" or rng.random() < 0.03
    has_vision = rng.random() < 0.20
    has_audio = rng.random() < 0.10
    sliding = family == "command-r" or rng.random() < 0.15
    enc_dec = family == "t5" or rng.random() < 0.02
    if enc_dec:
        has_vision = has_audio = False

    m = ModelMeta()
    m.architecture = family
    m.name = f"synthetic-{family}"
    lang, vocab = _sample_language_and_vocab(rng)
    m.language = lang
    m.quant_type = quant
    m.quant_block_style = classify_block_style(quant)
    m.quant_version = int(rng.choice([3, 3, 3, 2, 2, 1]))
    m.n_params = int(n_params)
    m.context_length = int(rng.choice(CONTEXTS, p=CONTEXT_P))
    m.n_layers = n_layers
    m.d_model = d_model
    m.d_ff = d_ff
    m.n_heads = n_heads
    m.n_kv_heads = n_kv
    m.head_dim = head_dim
    m.vocab_size = vocab
    m.is_moe = bool(is_moe)
    m.has_vision = bool(has_vision)
    m.has_audio = bool(has_audio)
    m.is_multimodal = has_vision or has_audio
    m.sliding_window = bool(sliding)
    m.has_ssm = bool(has_ssm)
    m.mixed_dtypes = bool(rng.random() < 0.20)
    m.rope_freq_base = float(rng.choice(rope_bases))
    m.rope_scaling_type = str(rng.choice(["none", "none", "none", "none",
                                          "none", "none", "none",
                                          "linear", "dynamic", "yarn"]))
    m.partial_rope = bool(rng.random() < 0.05)
    m.mtp_tensors = bool(rng.random() < 0.10)
    m.n_special_tokens = int(rng.integers(2, 13))
    m.add_bos = bool(rng.random() < 0.7)
    m.add_eos = bool(rng.random() < 0.6)
    m.byte_fallback = bool(rng.random() < 0.3)
    m.pooling_type = "mean" if rng.random() < 0.05 else ""
    m.is_encoder_decoder = bool(enc_dec)
    m.n_split_files = 1 if rng.random() < 0.85 else int(rng.integers(2, 9))
    m.rms_epsilon = float(rng.choice([1e-5, 1e-6]))
    m.n_tensors = n_layers * 6 + int(rng.integers(10, 40))
    return m


def generate_data(n: int = 200000, out_dir: str = "data", seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    from .model import N_FEATURES
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    from .rules import HEADS
    Y = {h: np.zeros(n, dtype=np.int64) for h in HEADS}
    for i in range(n):
        meta = _sample_record(rng)
        X[i] = encode_features(meta)
        prof = teacher(meta)
        for h in HEADS:
            Y[h][i] = prof[h]
    perm = rng.permutation(n)
    n_tr, n_va = int(n * 0.8), int(n * 0.1)
    splits = {"train.npz": perm[:n_tr], "val.npz": perm[n_tr:n_tr + n_va],
              "test.npz": perm[n_tr + n_va:]}
    for fname, idx in splits.items():
        arrays = {"X": X[idx]}
        for h in HEADS:
            arrays["Y_" + h] = Y[h][idx]
        np.savez_compressed(os.path.join(out_dir, fname), **arrays)
    counts = np.bincount(Y["body_size"], minlength=12)
    print(f"generated {n} records -> {out_dir}")
    print("body_size class counts:", counts.tolist())
    return out_dir
