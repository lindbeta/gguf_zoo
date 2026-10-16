"""Student network: feature encoding + small MLP with multi-head outputs."""
import math

import numpy as np
import torch
import torch.nn as nn

from . import rules
from .rules import HEAD_SIZES, HEADS


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(request: str):
    request = (request or "auto").lower()
    if request == "auto":
        return get_device()
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if request == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if request == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"unknown device '{request}' (use auto|cuda|mps|cpu)")


FEATURE_NAMES = [
    "log10_n_params", "log2_context", "n_layers", "log2_d_model", "dff_ratio",
    "log2_vocab", "quant_ord", "n_heads", "kv_ratio", "head_dim", "is_moe",
    "has_vision", "has_audio", "sliding_window", "block_style", "mixed_dtypes",
    "has_ssm", "log10_rope_base", "rope_scaling", "partial_rope", "mtp",
    "n_special", "add_bos", "add_eos", "pooling", "enc_dec", "arch_id",
    "lang_group",
]
N_FEATURES = len(FEATURE_NAMES)

# fixed (center, scale) per feature for normalization
_NORM = [
    (10.0, 1.0), (14.0, 3.0), (40.0, 15.0), (11.0, 2.0), (2.8, 0.6),
    (16.0, 1.5), (3.0, 2.0), (28.0, 20.0), (0.5, 0.4), (96.0, 48.0),
    (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (1.0, 1.0), (0.5, 0.5),
    (0.5, 0.5), (4.0, 1.0), (1.5, 1.2), (0.5, 0.5), (0.5, 0.5), (6.0, 4.0),
    (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (4.5, 3.0), (2.0, 1.5),
]


def encode_features(meta) -> np.ndarray:
    """ModelMeta -> normalized feature vector (N_FEATURES,)."""
    n_params = max(float(meta.n_params), 1.0)
    ctx = max(float(meta.context_length), 2.0)
    d_model = max(int(meta.d_model), 1)
    d_ff = max(int(meta.d_ff), 0)
    vocab = max(int(meta.vocab_size), 2)
    n_heads = max(int(meta.n_heads), 1)
    n_kv = max(int(meta.n_kv_heads), 0)
    ratio = d_ff / d_model if meta.d_model > 0 else 2.8
    raw = [
        math.log10(n_params),
        math.log2(ctx),
        float(meta.n_layers),
        math.log2(d_model),
        ratio,
        math.log2(vocab),
        float(rules.quant_ordinal(meta.quant_type)),
        float(meta.n_heads),
        (n_kv / n_heads) if n_heads else 1.0,
        float(meta.head_dim),
        float(meta.is_moe),
        float(meta.has_vision),
        float(meta.has_audio),
        float(meta.sliding_window),
        float(rules.block_style_id(meta.quant_block_style)),
        float(meta.mixed_dtypes),
        float(meta.has_ssm),
        math.log10(max(float(meta.rope_freq_base), 1.0)),
        float(rules.rope_scaling_id(meta.rope_scaling_type)),
        float(meta.partial_rope),
        float(meta.mtp_tensors),
        float(min(int(meta.n_special_tokens), 12)),
        float(meta.add_bos),
        float(meta.add_eos),
        float(bool(meta.pooling_type)),
        float(meta.is_encoder_decoder),
        float(rules.arch_id(meta.architecture)),
        float(rules.lang_group(meta.language)),
    ]
    out = np.empty(N_FEATURES, dtype=np.float32)
    for i, ((c, s), v) in enumerate(zip(_NORM, raw)):
        out[i] = (v - c) / s
    # clip so out-of-distribution metadata can't drive wild extrapolation
    return np.clip(out, -6.0, 6.0)


class CreatureNet(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, widths=(1024, 768, 640, 512, 512, 384)):
        super().__init__()
        layers = []
        prev = n_features
        for w in widths:
            layers += [nn.Linear(prev, w), nn.LayerNorm(w), nn.GELU(), nn.Dropout(0.1)]
            prev = w
        self.trunk = nn.Sequential(*layers)
        self.n_features = n_features
        # skip connection + small nonlinear bottleneck per head: sharp
        # threshold rules (gut, build, ...) can be carved directly from the
        # raw features without relying on the shared trunk.
        def make_head(n_out):
            return nn.Sequential(
                nn.Linear(prev + n_features, 128), nn.GELU(),
                nn.Linear(128, n_out))
        self.heads = nn.ModuleDict({h: make_head(HEAD_SIZES[h])
                                    for h in HEADS if h != "body_size"})
        # body_size is a sharp 1-D step function of log10(n_params) (feature 0);
        # give it a dedicated deep head on that feature alone so shared-trunk
        # smoothing does not blur the size-ladder boundaries (titan/kaiju).
        self.body_size_head = nn.Sequential(
            nn.Linear(1, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, HEAD_SIZES["body_size"]))

    def forward(self, x):
        z = torch.cat([self.trunk(x), x], dim=1)
        out = {h: head(z) for h, head in self.heads.items()}
        out["body_size"] = self.body_size_head(x[:, :1])
        return out

    def load_state_dict(self, state, strict=True, **kwargs):
        # v4 checkpoints have 3-class diet/marking heads (and the old
        # "heads.diet.*" key). Reject them loudly instead of mis-loading.
        legacy = "heads.diet.2.weight" in state
        wrong_width = any(
            f"heads.{h}.2.weight" in state
            and tuple(state[f"heads.{h}.2.weight"].shape)[0] != HEAD_SIZES[h]
            for h in ("diet_breadth", "marking_complexity")
        )
        if legacy or wrong_width:
            raise RuntimeError(
                "checkpoint is v4 (3-class diet/markings); "
                "regenerate data and retrain.")
        return super().load_state_dict(state, strict=strict, **kwargs)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
