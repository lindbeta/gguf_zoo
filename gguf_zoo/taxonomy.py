"""Deterministic New-Latin binomial from the model name + traits (FR-7)."""
from __future__ import annotations

import re

from .parser import ModelMeta
from .rules import Profile

NOISE = {"gguf", "base", "instruct", "chat", "it", "ml", "model", "beta",
         "lite", "turbo", "pro", "plus", "finetune", "merged", "abliterated",
         "safetensors"}

_BANDS = ["micro", "insect", "small", "mammal", "mega", "colossal"]
_BAND_GENDER = {"micro": "n", "insect": "f", "small": "f",
                "mammal": "m", "mega": "m", "colossal": "m"}
_BAND_SUFFIX = {"micro": "um", "insect": "apis", "small": "iola",
                "mammal": "ferus", "mega": "odon", "colossal": "titan"}

_EPITHETS = {
    "exoskeleton": ("loricatus", "loricata", "loricatum"),
    "quills": ("setosus", "setosa", "setosum"),
    "fur": ("pilosus", "pilosa", "pilosum"),
    "scales": ("squamatus", "squamata", "squamatum"),
    "bare": ("nudus", "nuda", "nudum"),
    "patchy": ("deformis", "deformis", "deforme"),
    "immaculate": ("immaculatus", "immaculata", "immaculatum"),
    "corrupt": ("corruptus", "corrupta", "corruptum"),
    "faded": ("pallidus", "pallida", "pallidum"),
    "great_gut": ("magaventer",) * 3,
    "small_gut": ("parviventris",) * 3,
    "ruminant": ("ruminans",) * 3,
    "horned": ("cornutus", "cornuta", "cornutum"),
    "unicorn": ("unicornis",) * 3,
    "hornless": ("inermis",) * 3,
    "cyclops": ("cyclops",) * 3,
    "specialist": ("exiguus", "exigua", "exiguum"),
    "alleater": ("omnivorus", "omnivora", "omnivorum"),
    "jade": ("viridis",) * 3,
    "gold": ("aureus", "aurea", "aureum"),
    "dark": ("niger", "nigra", "nigrum"),
    "tan": ("fuscus", "fusca", "fuscum"),
    "violet": ("violaceus", "violacea", "violaceum"),
    "blue": ("caeruleus", "caerulea", "caeruleum"),
    "pearl": ("perlaceus", "perlacea", "perlaceum"),
    "rust": ("rubigineus", "rubiginea", "rubigineum"),
    "ignotus": ("ignotus", "ignota", "ignotum"),
}

_INTEG_MAP = {0: "exoskeleton", 1: "quills", 2: "fur", 3: "scales", 4: "bare", 5: "patchy"}
_COND_MAP = {10: "immaculate", 9: "immaculate", 4: "corrupt", 3: "corrupt"}
_HEADGEAR_MAP = {1: "horned", 2: "horned", 3: "horned", 4: "unicorn"}
_GUT_MAP = {0: "small_gut", 2: "ruminant", 3: "great_gut"}
_DIET_MAP = {0: "specialist", 5: "alleater"}
_PALETTE_MAP = {0: "tan", 1: "jade", 2: "blue", 3: "pearl", 4: "violet",
                5: "gold", 6: "rust"}


def _tokens(name: str | None) -> list[str]:
    if not name:
        return []
    toks = re.split(r"[^A-Za-z0-9]+", name.lower())
    out = []
    for t in toks:
        if not t or t in NOISE:
            continue
        if re.fullmatch(r"[\d.]+", t) or re.match(r"^\d", t):
            continue
        out.append(t)
    return out


def _band(body_size: int) -> str:
    return _BANDS[max(0, min(12, body_size)) // 2]


def _stem(token: str) -> str:
    s = re.sub(r"[^a-z]", "", token)
    return (s[:7] if len(s) > 7 else s) or "ignot"


def genus_for(profile: Profile, meta: ModelMeta) -> tuple[str, str, str]:
    toks = _tokens(meta.name)
    band = _band(profile.body_size)
    stem = _stem(toks[0]) if toks else "ignot"
    stem = stem.capitalize()
    genus = f"{stem}{_BAND_SUFFIX[band]}"
    return genus, _BAND_GENDER[band], band


def epithet_candidates(profile: Profile) -> list[str]:
    out: list[str] = []
    fired_integ = profile.integument in _INTEG_MAP
    if fired_integ:
        out.append(_INTEG_MAP[profile.integument])
    if profile.condition in _COND_MAP:
        out.append(_COND_MAP[profile.condition])
    elif profile.condition in (6, 5) and not fired_integ:
        out.append("faded")
    if profile.headgear in _HEADGEAR_MAP:
        out.append(_HEADGEAR_MAP[profile.headgear])
    if profile.eye == 3:
        out.append("cyclops")
    if profile.gut in _GUT_MAP:
        out.append(_GUT_MAP[profile.gut])
    if profile.diet_breadth in _DIET_MAP:
        out.append(_DIET_MAP[profile.diet_breadth])
    if profile.palette in _PALETTE_MAP:
        out.append(_PALETTE_MAP[profile.palette])
    if not out and profile.headgear == 0:
        out.append("hornless")
    if not out:
        out.append("ignotus")
    return out


def _inflect(key: str, gender: str) -> str:
    forms = _EPITHETS[key]
    return forms[{"m": 0, "f": 1, "n": 2}[gender]]


def _authority(meta: ModelMeta) -> str:
    author = meta.author
    if author:
        letters = re.sub(r"[^A-Za-z]", "", author)
        if letters:
            return letters.capitalize() + "ia" if letters[0].isupper() else letters.lower()
    return (meta.arch or "incertus").lower()


def binomial(profile: Profile, meta: ModelMeta) -> str:
    genus, gender, _ = genus_for(profile, meta)
    forms = [_inflect(k, gender) for k in epithet_candidates(profile)[:2]]
    return f"{genus} {' '.join(forms)} {_authority(meta)}".strip()


_EPITHET_GLOSS = {
    "exoskeleton": "armored", "quills": "quilled", "fur": "hairy",
    "scales": "scaled", "bare": "smooth-skinned", "patchy": "patchy-coated",
    "immaculate": "immaculate", "corrupt": "corrupted", "faded": "faded",
    "great_gut": "big-gutted", "small_gut": "slim-gutted", "ruminant": "ruminant",
    "horned": "horned", "unicorn": "one-horned", "hornless": "hornless",
    "cyclops": "one-eyed", "specialist": "hard-to-please", "alleater": "all-eating",
    "jade": "green", "gold": "gilded", "dark": "dark", "tan": "sandy",
    "violet": "violet", "ignotus": "unclassified",
}


def gloss(profile: Profile, meta: ModelMeta, phrase: str | None = None) -> str:
    _, gender, band = genus_for(profile, meta)
    keys = epithet_candidates(profile)[:2]
    parts = [_EPITHET_GLOSS[k] for k in keys]
    noun = {"micro": "microbe", "insect": "insect", "small": "small creature",
            "mammal": "mammal", "mega": "giant", "colossal": "titan"}[band]
    if phrase is None:
        from .pools import HABITAT_PHRASES
        phrase = HABITAT_PHRASES[
            ["terrestrial", "aerial", "aquatic"][profile.habitat]][0]
    return f"a {', '.join(parts)} {noun} {phrase}"
