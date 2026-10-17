"""Small trained input-type -> visible-sense classifier (the only ML component).

Hashed character n-gram bag + MLP. Stochastic, draw-driven inference + three-run
consensus (FR-6). GPU when available (FR-12). See SPEC.md FR-6, FR-12.
"""
from __future__ import annotations

import hashlib
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .senses import (REQUIRED_OVERRIDES, VISIBLE_SENSES, Sense, SenseResult,
                     build_dataset, keyword_sense, normalize)

CONF_THRESHOLD = 0.55
MAX_CONSENSUS_ROUNDS = 50
NB = 1024          # hashed n-gram buckets
ARTIFACT_ENV = "GGUF_ZOO_ARTIFACT"


def artifact_path() -> Path:
    env = os.environ.get(ARTIFACT_ENV)
    if env:
        return Path(env)
    return Path(__file__).parent / "artifacts" / "sense_clf.pt"


def resolve_device(request: str = "auto") -> torch.device:
    """FR-12.1: cuda -> mps -> cpu for auto; explicit unavailable raises."""
    request = (request or "auto").lower()
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
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


def _hash(s: str) -> int:
    h = 0
    for b in s.encode("utf-8"):
        h = (h * 131 + b) & 0xFFFFFFFF
    return h


def featurize(name: str) -> np.ndarray:
    """Hashed character 1/2/3-gram bag (deterministic, PYTHONHASHSEED-safe)."""
    v = np.zeros(NB, dtype=np.float32)
    s = (name or "").lower()
    for n in (1, 2, 3):
        for i in range(len(s) - n + 1):
            v[_hash(s[i:i + n]) % NB] += 1.0
    if s:
        v[_hash("^" + s + "$") % NB] += 2.0
    return v


class SenseNet(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(NB, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        return self.fc2(h)


def _grouped_split(rows, seed=0):
    groups = sorted({g for _, _, g in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    tr = set(groups[:int(n * 0.8)])
    va = set(groups[int(n * 0.8):int(n * 0.9)])
    te = set(groups[int(n * 0.9):])
    lab = {s: i for i, s in enumerate(VISIBLE_SENSES)}
    out = {"train": [], "val": [], "test": []}
    for name, sense, g in rows:
        if sense not in lab:
            continue
        split = "train" if g in tr else "val" if g in va else "test"
        out[split].append((featurize(name), lab[sense]))
    return out, lab


def _tensors(items, device):
    X = torch.tensor(np.stack([x for x, _ in items]), dtype=torch.float32, device=device)
    y = torch.tensor([y for _, y in items], dtype=torch.long, device=device)
    return X, y


def train_classifier(seed: int = 0, epochs: int = 40, device: str = "auto",
                     out_path: str | Path | None = None, batch_size: int = 256) -> Path:
    """FR-6.3 / FR-12.2: train on GPU when available, save CPU-tensor checkpoint."""
    dev = resolve_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rows = build_dataset(seed)
    splits, lab = _grouped_split(rows, seed)
    model = SenseNet(len(VISIBLE_SENSES)).to(dev)
    Xtr, ytr = _tensors(splits["train"], dev)
    Xva, yva = _tensors(splits["val"], dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    n = Xtr.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for b in range(0, n, batch_size):
            idx = perm[b:b + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == yva).float().mean().item()
        if acc >= best:
            best = acc
            best_state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    ds_hash = hashlib.sha256(repr(sorted(rows)).encode()).hexdigest()
    ckpt = {"state_dict": best_state, "labels": VISIBLE_SENSES,
            "val_acc": best, "dataset_hash": ds_hash, "seed": seed}
    path = Path(out_path) if out_path else artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    _MODEL_CACHE.clear()
    return path


def _load_or_train(device: str = "auto"):
    path = artifact_path()
    dev = resolve_device(device)
    if not path.exists():
        train_classifier(device=device, out_path=path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("labels") != VISIBLE_SENSES:
        raise RuntimeError("checkpoint label mismatch; retrain senses")
    model = SenseNet(len(VISIBLE_SENSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    return model, dev


_MODEL_CACHE: dict[str, tuple] = {}


def _model(device: str = "auto"):
    if device not in _MODEL_CACHE:
        _MODEL_CACHE[device] = _load_or_train(device)
    return _MODEL_CACHE[device]


def _predict(name: str, draw: int, device: str = "auto") -> tuple[str, float]:
    model, dev = _model(device)
    torch.manual_seed(draw)
    x = torch.tensor(featurize(name), dtype=torch.float32, device=dev).unsqueeze(0)
    with torch.no_grad():
        x = x + (torch.rand(x.shape, device=dev) - 0.5) * 0.1
        x = F.dropout(x, 0.1, training=True)
        h = F.gelu(model.fc1(x))
        h = F.dropout(h, 0.1, training=True)
        probs = F.softmax(model.fc2(h), dim=-1)[0]
    conf_t, idx_t = probs.max(0)
    return VISIBLE_SENSES[idx_t.item()], conf_t.item()


def classify_input_type(name: str, draw: int = 0, device: str = "auto") -> SenseResult:
    """FR-6.5: name + random number -> visible sense (stochastic)."""
    n = normalize(name)
    if n in REQUIRED_OVERRIDES:
        return SenseResult(REQUIRED_OVERRIDES[n], 1.0, "override")
    if not n:
        return SenseResult(Sense.SENSE_UNKNOWN.value, 0.0, "fallback")
    sense, conf = _predict(name, draw, device)
    if conf < CONF_THRESHOLD:
        return SenseResult(keyword_sense(name), conf, "fallback")
    return SenseResult(sense, conf, "model")


def classify_input_type_consensus(name: str, seed: int = 0,
                                  device: str = "auto") -> SenseResult:
    """FR-6.10: run 3x with 3 different draws until all three agree."""
    rng = random.Random(seed)
    last: list[SenseResult] = []
    for _ in range(MAX_CONSENSUS_ROUNDS):
        draws = [rng.getrandbits(64) for _ in range(3)]
        results = [classify_input_type(name, d, device) for d in draws]
        if results[0].sense == results[1].sense == results[2].sense:
            return results[0]
        last = results
    counts = Counter(r.sense for r in last)
    top = counts.most_common(1)[0][0]
    return SenseResult(top, 0.0, "fallback")


def eval_classifier(device: str = "auto") -> dict:
    """FR-6.7: alias-grouped accuracy + confusion over visible senses."""
    rows = build_dataset(0)
    splits, lab = _grouped_split(rows, 0)
    model, dev = _model(device)
    X, y = _tensors(splits["test"], dev)
    with torch.no_grad():
        pred = model(X).argmax(1)
    acc = (pred == y).float().mean().item()
    n = len(VISIBLE_SENSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
        cm[t][p] += 1
    return {"accuracy": acc, "confusion": cm, "labels": VISIBLE_SENSES}
