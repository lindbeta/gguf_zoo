"""Second trained model: unmapped metadata field key -> best-match attribute.

Recognized keys are mapped deterministically by rules.py (FR-2); every
*unrecognized* key is mapped by this classifier to its best-match physical
attribute (FR-16), so no field is left unmapped and no raw key/value is ever
printed (FR-3.4). The label space pairs the real FR-2 trait families with
fictional physical attributes, so unknown fields land on invented anatomy
instead of colliding with real attributes. Three-run consensus inference
(FR-6.10 protocol, FR-16.4): three seeded runs must agree, else re-run.
"""
from __future__ import annotations

import hashlib
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .classifier import (MAX_CONSENSUS_ROUNDS, SenseNet, _tensors, featurize,
                         resolve_device)

CONF_THRESHOLD = 0.55
ARTIFACT_ENV = "GGUF_ZOO_ATTR_ARTIFACT"

# FR-16.2 — one label per FR-2 trait family.
REAL_ATTRIBUTES: list[str] = [
    "size", "condition", "gut", "build", "eyes", "body_length", "diet",
    "integument", "palette", "fastness", "gradient", "headgear", "ears",
    "modality", "identity",
]

# FR-16.2 — fictional physical attributes: id -> (display name, kind).
FICTIONAL_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "spiracle_array": ("Spiracle Array", "structure"),
    "fermentation_crop": ("Fermentation Crop", "organ"),
    "resonance_bulla": ("Resonance Bulla", "organ"),
    "nuchal_sail": ("Nuchal Sail", "structure"),
    "tide_ganglion": ("Tide Ganglion", "organ"),
    "marrow_lattice": ("Marrow Lattice", "structure"),
    "shadow_membrane": ("Shadow Membrane", "structure"),
    "gizzard_stones": ("Gizzard Stones", "structure"),
}

ATTR_UNKNOWN = "attr_unknown"

ATTR_LABELS: list[str] = REAL_ATTRIBUTES + list(FICTIONAL_ATTRIBUTES) + [ATTR_UNKNOWN]

# Attribute-exclusive morphemes per label (FR-16.4, FR-6.2 method). Roots are
# chosen so no morpheme is a substring of another attribute's morphemes, and
# held-out aliases still share morphemes with training aliases.
ATTR_MORPHEMES: dict[str, list[str]] = {
    # real FR-2 trait families
    "size": ["param", "nparams", "modelsize", "weightcount", "bulkness"],
    "condition": ["quant", "ftype", "bits", "compression", "losslevel"],
    "gut": ["ctx", "context", "window", "seq", "gut"],
    "build": ["ffn", "hidden", "mlp", "expansion", "wiry"],
    "eyes": ["heads", "gaze", "ocelli", "kvheads"],
    "body_length": ["layers", "strata", "depth", "segments", "spine"],
    "diet": ["vocab", "lexicon", "intake", "diet", "wordlist"],
    "integument": ["ssm", "moe", "expert", "coat", "blocktype"],
    "palette": ["arch", "family", "palette", "color", "hue"],
    "fastness": ["precision", "dtype", "fade", "fastness", "tint"],
    "gradient": ["theta", "freqbase", "gradient", "ropefreq", "slowslope"],
    "headgear": ["yarn", "horn", "scaling", "antler", "spike"],
    "ears": ["ear", "pinna", "keylen", "aural", "hearer"],
    "modality": ["modal", "vision", "audio", "image", "sense"],
    "identity": ["name", "author", "license", "provenance", "homepage"],
    # fictional physical attributes
    "spiracle_array": ["spir", "spiracle", "stigma", "slitpore", "venthole"],
    "fermentation_crop": ["crop", "ferment", "rumen", "paunch", "zymo"],
    "resonance_bulla": ["bulla", "reson", "humchamber", "timpan", "vox"],
    "nuchal_sail": ["nuch", "sail", "crest", "dorsalfin", "keelplate"],
    "tide_ganglion": ["tide", "tidal", "lunar", "floodgate", "springtide"],
    "marrow_lattice": ["marrow", "lattice", "trabec", "reticula", "cancellous"],
    "shadow_membrane": ["umbra", "shadowfell", "velum", "pellicle", "murk"],
    "gizzard_stones": ["gizzard", "gastrolith", "grit", "millstone", "pulver"],
}

_KEY_PREFIXES = ["", "general."]
_SYNTH_PREFIXES = ["", "re", "pre", "auto", "micro", "multi", "crypto"]
_SYNTH_SUFFIXES = ["", "_count", "_type", "_length", "_id", "ia"]


def attr_artifact_path() -> Path:
    env = os.environ.get(ARTIFACT_ENV)
    if env:
        return Path(env)
    return Path(__file__).parent / "artifacts" / "field_attr_clf.pt"


def build_attr_dataset(seed: int = 0) -> list[tuple[str, str, int]]:
    """(key alias, attribute label, group) rows. Each synthesized alias is its
    own group, so held-out aliases still share morphemes with training
    aliases (FR-6.2 method, applied to field keys)."""
    rng = random.Random(seed)
    rows: list[tuple[str, str, int]] = []
    group = 0
    for attr, morphs in ATTR_MORPHEMES.items():
        for m in morphs:
            for pre in _SYNTH_PREFIXES:
                for suf in _SYNTH_SUFFIXES:
                    base = f"{pre}{m}{suf}"
                    for kpre in _KEY_PREFIXES:
                        key = f"{kpre}{base}"
                        rows.append((key, attr, group))
                        rows.append((key.upper(), attr, group))
                    group += 1
    rng.shuffle(rows)
    return rows


def _attr_grouped_split(rows, seed=0):
    groups = sorted({g for _, _, g in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    tr = set(groups[:int(n * 0.8)])
    va = set(groups[int(n * 0.8):int(n * 0.9)])
    te = set(groups[int(n * 0.9):])
    lab = {s: i for i, s in enumerate(ATTR_LABELS)}
    out = {"train": [], "val": [], "test": []}
    for name, attr, g in rows:
        split = "train" if g in tr else "val" if g in va else "test"
        out[split].append((featurize(name), lab[attr]))
    return out, lab


def train_attribute_classifier(seed: int = 0, epochs: int = 40, device: str = "auto",
                               out_path: str | Path | None = None,
                               batch_size: int = 256) -> Path:
    """FR-16.4: train the field->attribute model (GPU when available)."""
    dev = resolve_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rows = build_attr_dataset(seed)
    splits, _ = _attr_grouped_split(rows, seed)
    model = SenseNet(len(ATTR_LABELS)).to(dev)
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
    ckpt = {"state_dict": best_state, "labels": ATTR_LABELS,
            "val_acc": best, "dataset_hash": ds_hash, "seed": seed}
    path = Path(out_path) if out_path else attr_artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    _ATTR_MODEL_CACHE.clear()
    return path


def _load_or_train_attrs(device: str = "auto"):
    path = attr_artifact_path()
    dev = resolve_device(device)
    if not path.exists():
        train_attribute_classifier(device=device, out_path=path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("labels") != ATTR_LABELS:
        raise RuntimeError("checkpoint label mismatch; retrain attributes")
    model = SenseNet(len(ATTR_LABELS))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    return model, dev


_ATTR_MODEL_CACHE: dict[str, tuple] = {}


def _attr_model(device: str = "auto"):
    if device not in _ATTR_MODEL_CACHE:
        _ATTR_MODEL_CACHE[device] = _load_or_train_attrs(device)
    return _ATTR_MODEL_CACHE[device]


def _predict_attr(key: str, seed: int, device: str = "auto") -> tuple[str, float]:
    """One seeded stochastic run of the attribute model (FR-6.10 style)."""
    model, dev = _attr_model(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.tensor(featurize(key), dtype=torch.float32, device=dev).unsqueeze(0)
    with torch.no_grad():
        x = x + (torch.rand(x.shape, device=dev) - 0.5) * 0.1
        x = F.dropout(x, 0.1, training=True)
        h = F.gelu(model.fc1(x))
        h = F.dropout(h, 0.1, training=True)
        probs = F.softmax(model.fc2(h), dim=-1)[0]
    conf, idx = probs.max(0)
    return ATTR_LABELS[idx.item()], conf.item()


def classify_attribute(key: str, seed: int = 0, device: str = "auto") -> str:
    """FR-16.1/16.4: three-run consensus (FR-6.10) over attribute labels;
    abstains to attr_unknown below CONF_THRESHOLD."""
    key = (key or "").strip()
    if not key:
        return ATTR_UNKNOWN
    rng = random.Random(seed)
    last: list[tuple[str, float]] = []
    for _ in range(MAX_CONSENSUS_ROUNDS):
        seeds = [rng.getrandbits(64) for _ in range(3)]
        results = [_predict_attr(key, s, device) for s in seeds]
        if results[0][0] == results[1][0] == results[2][0]:
            label, conf = results[0]
            if conf < CONF_THRESHOLD:
                return ATTR_UNKNOWN
            return label
        last = results
    counts = Counter(lbl for lbl, _ in last)
    top, _ = counts.most_common(1)[0]
    return top


def eval_attribute_classifier(device: str = "auto") -> dict:
    """FR-16.5: alias-grouped accuracy + confusion over attribute labels."""
    rows = build_attr_dataset(0)
    splits, _ = _attr_grouped_split(rows, 0)
    model, dev = _attr_model(device)
    X, y = _tensors(splits["test"], dev)
    with torch.no_grad():
        pred = model(X).argmax(1)
    acc = (pred == y).float().mean().item()
    n = len(ATTR_LABELS)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
        cm[t][p] += 1
    return {"accuracy": acc, "confusion": cm, "labels": ATTR_LABELS}
