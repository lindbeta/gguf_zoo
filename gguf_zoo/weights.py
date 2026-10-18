"""In-depth tensor reading (FR-19).

Parses the tensor info records and samples a bounded, deterministic subset of
weight blocks from the tensor data section, computing quant-robust statistics
(RMS, kurtosis, sparsity) from dequantized values. No RNG: blocks are chosen at
evenly spaced positions. See SPEC.md FR-19.
"""
from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .parser import GGUFError, _Reader

# GGML tensor dtype codes
F32, F16 = 0, 1
Q4_0, Q4_1 = 2, 3
Q5_0, Q5_1, Q8_0, Q8_1 = 6, 7, 8, 9
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K = 10, 11, 12, 13, 14, 15
BF16 = 24
IQ4_XS = 23

# dtype -> (values per block, bytes per block) for quantized types
_BLOCK = {
    Q4_0: (32, 18), Q4_1: (32, 20), Q8_0: (32, 34),
    Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210),
}
# plain (unquantized) dtypes: bytes per element
_PLAIN = {F32: 4, F16: 2, BF16: 2}
SUPPORTED = set(_BLOCK) | set(_PLAIN)

_MAX_BLOCKS = 64          # blocks sampled per tensor
_PLAIN_CHUNK = 256        # elements per sample chunk for plain dtypes


@dataclass
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    dtype: int
    offset: int


@dataclass
class NeuralProfile:
    n_tensors: int = 0
    n_supported: int = 0
    spread: float = 0.0          # mean log10 RMS across weight tensors
    heterogeneity: float = 0.0   # std of log10 RMS
    outlier: float = 0.0         # mean kurtosis
    sparsity: float = 0.0        # mean fraction of |v| < 1/4 RMS
    trend: float = 0.0           # slope of log10 RMS vs block index
    n_layers: int | None = None
    n_embd: int | None = None
    d_ff: int | None = None


# ---------------------------------------------------------------------------
# tensor info records
# ---------------------------------------------------------------------------
def parse_tensor_infos(buf, kv_end: int, n_tensors: int,
                       alignment: int = 32) -> tuple[list[TensorInfo], int]:
    r = _Reader(memoryview(buf)[kv_end:])
    infos: list[TensorInfo] = []
    for _ in range(n_tensors):
        name = r.read_str()
        n_dims = r.unpack("<I")
        dims = tuple(r.unpack("<Q") for _ in range(n_dims))
        dtype = r.unpack("<I")
        offset = r.unpack("<Q")
        infos.append(TensorInfo(name, dims, dtype, offset))
    data_start = kv_end + r.pos
    data_start = (data_start + alignment - 1) // alignment * alignment
    return infos, data_start


# ---------------------------------------------------------------------------
# dequantization (value order within a block is irrelevant for statistics)
# ---------------------------------------------------------------------------
def _f16(b: bytes) -> float:
    return float(np.frombuffer(b, dtype="<f2")[0])


def _nibbles(qs: bytes) -> np.ndarray:
    q = np.frombuffer(qs, dtype=np.uint8)
    out = np.empty(2 * len(q), dtype=np.uint8)
    out[0::2] = q & 0xF
    out[1::2] = q >> 4
    return out


def _qk_scales(buf: bytes):
    d = _f16(buf[0:2])
    dmin = _f16(buf[2:4])
    s = np.frombuffer(buf[4:16], dtype=np.uint8)
    sc = np.empty(8, dtype=np.float64)
    mn = np.empty(8, dtype=np.float64)
    for j in range(8):
        sc[j] = (s[j] & 0xF) | ((s[j + 4] >> 3) & 0x30)
        mn[j] = (s[j] >> 4) | ((s[j + 4] >> 1) & 0x30)
    return d, dmin, sc, mn


def dequant(block: bytes, dtype: int) -> np.ndarray:
    """Dequantize one block of tensor data into float64 values."""
    if dtype == F32:
        return np.frombuffer(block, dtype="<f4").astype(np.float64)
    if dtype == F16:
        return np.frombuffer(block, dtype="<f2").astype(np.float64)
    if dtype == BF16:
        u = np.frombuffer(block, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float64)
    if dtype == Q4_0:
        d = _f16(block[0:2])
        return d * (_nibbles(block[2:18]).astype(np.float64) - 8.0)
    if dtype == Q4_1:
        d = _f16(block[0:2])
        m = _f16(block[2:4])
        return d * _nibbles(block[4:20]).astype(np.float64) + m
    if dtype == Q8_0:
        d = _f16(block[0:2])
        return d * np.frombuffer(block[2:34], dtype=np.int8).astype(np.float64)
    if dtype == Q4_K:
        d, dmin, sc, mn = _qk_scales(block)
        q = _nibbles(block[16:144]).astype(np.float64)
        sub = np.repeat(np.arange(8), 32)
        return (d * sc)[sub] * q - (dmin * mn)[sub]
    if dtype == Q5_K:
        d, dmin, sc, mn = _qk_scales(block)
        qh = np.frombuffer(block[16:48], dtype=np.uint8)
        hi = np.unpackbits(qh, bitorder="little").astype(np.float64)
        q = _nibbles(block[48:176]).astype(np.float64) + 16.0 * hi[:256]
        sub = np.repeat(np.arange(8), 32)
        return (d * sc)[sub] * q - (dmin * mn)[sub]
    if dtype == Q6_K:
        lo = _nibbles(block[0:128])
        qh = np.frombuffer(block[128:192], dtype=np.uint8)
        h = np.empty(256, dtype=np.uint8)
        for k in range(4):
            h[k::4] = (qh >> (2 * k)) & 0x3
        q = (lo | (h << 4)).astype(np.float64)
        sc = np.frombuffer(block[192:208], dtype=np.int8).astype(np.float64)
        d = _f16(block[208:210])
        sub = np.repeat(np.arange(16), 16)
        return (d * sc)[sub] * (q - 32.0)
    raise GGUFError(f"Unsupported tensor dtype {dtype}")


def block_size(dtype: int) -> tuple[int, int]:
    """(values per block, bytes per block) for a supported dtype."""
    if dtype in _BLOCK:
        return _BLOCK[dtype]
    if dtype in _PLAIN:
        return _PLAIN_CHUNK, _PLAIN_CHUNK * _PLAIN[dtype]
    raise GGUFError(f"Unsupported tensor dtype {dtype}")


# ---------------------------------------------------------------------------
# quantization (used by the demo and the quant-robustness tests)
# ---------------------------------------------------------------------------
def quantize(values: np.ndarray, dtype: int) -> bytes:
    """Quantize a float array whose length is a multiple of the block size."""
    values = np.asarray(values, dtype=np.float64)
    if dtype == F32:
        return values.astype("<f4").tobytes()
    if dtype == F16:
        return values.astype("<f2").tobytes()
    if dtype == Q8_0:
        out = bytearray()
        for i in range(0, len(values), 32):
            blk = values[i:i + 32]
            amax = max(float(np.max(np.abs(blk))), 1e-30)
            d = np.float32(amax / 127.0)
            q = np.clip(np.round(blk / d), -127, 127).astype(np.int8)
            out += np.float16(d).tobytes() + q.tobytes()
        return bytes(out)
    if dtype == Q4_0:
        out = bytearray()
        for i in range(0, len(values), 32):
            blk = values[i:i + 32]
            amax = max(float(np.max(np.abs(blk))), 1e-30)
            d = np.float32(amax / 8.0)
            q = np.clip(np.round(blk / d) + 8.0, 0, 15).astype(np.uint8)
            packed = (q[1::2] << 4) | q[0::2]
            out += np.float16(d).tobytes() + packed.tobytes()
        return bytes(out)
    raise GGUFError(f"Cannot quantize to dtype {dtype}")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _stats(vals: np.ndarray) -> tuple[float, float, float, float] | None:
    if vals.size == 0:
        return None
    rms = float(np.sqrt(np.mean(vals * vals)))
    if rms == 0.0:
        return None
    mean = float(np.mean(vals))
    dev = vals - mean
    m2 = float(np.mean(dev * dev))
    m4 = float(np.mean(dev ** 4))
    kurt = m4 / (m2 * m2) if m2 > 0 else 0.0
    sparsity = float(np.mean(np.abs(vals) < 0.25 * rms))
    return rms, mean, kurt, sparsity


def _sample_blocks(data, offset: int, dtype: int, n_values: int):
    vals_per_block, bytes_per_block = block_size(dtype)
    n_blocks = n_values // vals_per_block
    if n_blocks <= 0:
        return None
    n_take = min(_MAX_BLOCKS, n_blocks)
    idx = np.unique(np.linspace(0, n_blocks - 1, n_take).round().astype(int))
    chunks = []
    for b in idx:
        start = offset + int(b) * bytes_per_block
        chunks.append(dequant(bytes(data[start:start + bytes_per_block]), dtype))
    return np.concatenate(chunks)


_SKIP_SUBSTR = ("norm", "rope", "freq", "lightnet")


def _is_norm(name: str) -> bool:
    low = name.lower()
    return any(s in low for s in _SKIP_SUBSTR)


def _layer_of(name: str) -> int | None:
    import re
    mo = re.match(r"^blk\.(\d+)\.", name)
    return int(mo.group(1)) if mo else None


def neural_profile(source, kv: dict[str, Any] | None = None,
                   header: dict[str, Any] | None = None) -> NeuralProfile | None:
    """Sample the tensor data and aggregate quant-robust statistics.

    `source` is a path to the .gguf file or the raw bytes (demo).
    Returns None when no tensor data is available or nothing is supported.
    """
    if kv is None or header is None or "kv_end" not in (header or {}):
        from .parser import parse_gguf_bytes, read_metadata
        if isinstance(source, (bytes, bytearray, memoryview)):
            kv, header = parse_gguf_bytes(bytes(source))
        else:
            kv, header = read_metadata(source)
    n_tensors = int(header.get("n_tensors", 0))
    if n_tensors <= 0:
        return None
    align = int(kv.get("general.alignment", 32))
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _profile(memoryview(source), kv, header, n_tensors, align)
    path = Path(source)
    try:
        with path.open("rb") as fh:
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return _profile(mm, kv, header, n_tensors, align)
    except OSError as exc:
        raise GGUFError(f"Cannot read file: {exc}") from exc


def _profile(data, kv, header, n_tensors, align) -> NeuralProfile | None:
    infos, data_start = parse_tensor_infos(data, int(header["kv_end"]),
                                           n_tensors, align)
    np_ = NeuralProfile(n_tensors=n_tensors)
    from .parser import build_meta
    meta = build_meta(dict(kv))
    np_.n_layers, np_.n_embd, np_.d_ff = meta.n_layers, meta.n_embd, meta.d_ff

    log_rms: list[float] = []
    kurtosities: list[float] = []
    sparsities: list[float] = []
    trend_x: list[float] = []
    trend_y: list[float] = []
    for info in infos:
        if info.dtype not in SUPPORTED or _is_norm(info.name):
            continue
        n_values = 1
        for d in info.dims:
            n_values *= int(d)
        vals = _sample_blocks(data, data_start + info.offset, info.dtype,
                              n_values)
        if vals is None:
            continue
        st = _stats(vals)
        if st is None:
            continue
        np_.n_supported += 1
        rms, _mean, kurt, sparsity = st
        log_rms.append(np.log10(rms))
        kurtosities.append(kurt)
        sparsities.append(sparsity)
        layer = _layer_of(info.name)
        if layer is not None:
            trend_x.append(float(layer))
            trend_y.append(np.log10(rms))
    if np_.n_supported == 0:
        return None
    arr = np.asarray(log_rms)
    np_.spread = float(np.mean(arr))
    np_.heterogeneity = float(np.std(arr))
    np_.outlier = float(np.mean(kurtosities))
    np_.sparsity = float(np.mean(sparsities))
    if len(trend_x) >= 2 and np.ptp(trend_x) > 0:
        np_.trend = float(np.polyfit(trend_x, np.asarray(trend_y), 1)[0])
    return np_
