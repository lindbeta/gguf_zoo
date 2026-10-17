"""Unmapped metadata fields and uncovered senses -> named fictional anatomy.

Every anomaly is a fictional *organ* (behaviour description) or *bodily
structure* (structure + movement description), deterministically named
(FR-3.9-3.12). Unmapped fields are attributed to a best-match physical
attribute by the second trained model (FR-16); raw keys, values, and input
names are never printed (FR-3.4). Never uses builtin hash().
"""
from __future__ import annotations

import hashlib
import random

from .attributes import ATTR_UNKNOWN, FICTIONAL_ATTRIBUTES, classify_attribute
from .pools import (FICTIONAL_ANATOMY_NAMES, ORGAN_BEHAVIOURS, PERSON_NAMES,
                    STRUCTURE_MOVEMENTS)


def _rng(seed_str: str) -> random.Random:
    digest = hashlib.sha256(seed_str.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _desc(seed_str: str, kind: str) -> str:
    rng = _rng(seed_str)
    return rng.choice(ORGAN_BEHAVIOURS if kind == "organ" else STRUCTURE_MOVEMENTS)


def _anatomy(seed_str: str) -> str:
    """Return '<Name> (<organ|structure>) — <description>'."""
    rng = _rng(seed_str)
    word = rng.choice(FICTIONAL_ANATOMY_NAMES)
    kind = rng.choice(["organ", "structure"])
    name = f"{rng.choice(PERSON_NAMES)}'s {word}" if rng.random() < 0.4 else word
    desc = rng.choice(ORGAN_BEHAVIOURS) if kind == "organ" else rng.choice(STRUCTURE_MOVEMENTS)
    return f"{name} ({kind}) — {desc}"


def field_anatomy(key: str, value, device: str = "auto") -> str | None:
    """FR-16.3: attribute-named anatomy for one unmapped field, or None when
    the field maps to a real attribute already rendered from its source."""
    label = classify_attribute(key, device=device)
    seed = f"{key}::{value!r}"
    if label in FICTIONAL_ATTRIBUTES:
        display, kind = FICTIONAL_ATTRIBUTES[label]
        return f"{display} ({kind}) — {_desc(seed, kind)}"
    if label == ATTR_UNKNOWN:
        return _anatomy(seed)
    return None


def anomaly_organs(unknown: dict, extra_senses=None,
                   device: str = "auto") -> list[tuple[str, str]]:
    """Return ordered (anatomy body, internal source id) lines.

    The source id (field key or sense id) is for ordering/tests only and is
    never printed (FR-3.4). Metadata anomalies first (sorted by key), then
    uncovered-sense anomalies (sorted by sense id) per FR-3.8.
    ``extra_senses`` is a list of ``(sense_id, input_name)`` pairs.
    """
    out: list[tuple[str, str]] = []
    for key in sorted(unknown):
        body = field_anatomy(key, unknown[key], device)
        if body:
            out.append((body, key))
    for sense, inp in sorted(extra_senses or [], key=lambda x: (x[0], x[1])):
        out.append((_anatomy(f"sense::{sense}::{inp}"), sense))
    return out
