"""Deterministic Linnaean nomenclature for gguf-zoo creatures.

These are New Latin coinages for a speculative zoo, NOT real taxonomy: the
names are playful Latinized descriptions of predicted traits, never claims
about actual biological classification.

Name anatomy:  Genus epithet [epithet2] Authority[, Year]
- genus: Latinized architecture stem fused with a size-band animal root;
  every genus carries a grammatical gender (m/f/n).
- epithets: trait-derived Latin adjectives inflected to agree with the
  genus gender, chosen by a fixed trait priority.
- authority: Latinized model author (or architecture), plus a `& basegenus`
  ampersand for finetunes.
- year: file mtime year when known; omitted, never fabricated.

Collisions: two different models sharing a binomial is legal - real
taxonomy distinguishes such homonyms by authority and year. `strict=True`
additionally keeps a names.json registry and forces uniqueness by walking
down the epithet candidate list, then appending a hash-derived trinomen.

Pure stdlib (hashlib, json, os, re). Deterministic: same metadata -> same
name, always.
"""
import hashlib
import json
import os
import re

from .render import meta_hash

TAXO_SALT = "taxonomy-v1"

# ---------------------------------------------------------------------------
# Genus: architecture stem x size band
# ---------------------------------------------------------------------------
# (architecture prefix, stem); first match wins. MoE models and mamba/SSM
# models override with the honeycomb / segmented-armor stems.
ARCH_STEMS = [
    ("mixtral", "Fav"),
    ("llama", "Llama"),
    ("qwen", "Qwen"),
    ("mistral", "Mistral"),
    ("gemma", "Gemma"),
    ("phi", "Phi"),
    ("deepseek", "Profund"),
    ("mamba", "Lorica"),
]

BANDS = ["micro", "insect", "small", "mammal", "mega", "colossal"]
BAND_GENDER = {"micro": "n", "insect": "f", "small": "f",
               "mammal": "m", "mega": "m", "colossal": "m"}
BAND_NOUN = {"micro": "microbe", "insect": "insect", "small": "small creature",
             "mammal": "mammal", "mega": "giant", "colossal": "titan"}

# (stem, band) -> 1-3 curated fused genus names. Every entry is pronounceable
# and free of 3+ consecutive consonants (see test suite).
GENUS_FUSIONS = {
    ("Llama", "micro"): ["Llamium", "Llamum"],
    ("Llama", "insect"): ["Llamaapis", "Llamapis"],
    ("Llama", "small"): ["Llamaiola", "Llamia"],
    ("Llama", "mammal"): ["Llamaferus", "Llamus"],
    ("Llama", "mega"): ["Llamaferodon", "Llamadon"],
    ("Llama", "colossal"): ["Llamatitan", "Llamarex"],

    ("Qwen", "micro"): ["Qwenum", "Qwenium"],
    ("Qwen", "insect"): ["Qwenapis", "Qwenis"],
    ("Qwen", "small"): ["Qweniola", "Qwenia"],
    ("Qwen", "mammal"): ["Qwenferus", "Qwenus"],
    ("Qwen", "mega"): ["Qwenodon"],
    ("Qwen", "colossal"): ["Qwentitan", "Qwenrex"],

    ("Mistral", "micro"): ["Mistralum", "Mistralium"],
    ("Mistral", "insect"): ["Mistralapis", "Mistralis"],
    ("Mistral", "small"): ["Mistraliola", "Mistralia"],
    ("Mistral", "mammal"): ["Mistralferus", "Mistralus"],
    ("Mistral", "mega"): ["Mistralodon"],
    ("Mistral", "colossal"): ["Mistralitan", "Mistralrex"],

    ("Gemma", "micro"): ["Gemmum", "Gemmium"],
    ("Gemma", "insect"): ["Gemmapis", "Gemmis"],
    ("Gemma", "small"): ["Gemmiola", "Gemmia"],
    ("Gemma", "mammal"): ["Gemmaferus", "Gemmus"],
    ("Gemma", "mega"): ["Gemmaferodon", "Gemmadon"],
    ("Gemma", "colossal"): ["Gemmatitan", "Gemmarex"],

    ("Phi", "micro"): ["Phium", "Phiium"],
    ("Phi", "insect"): ["Phiapis"],
    ("Phi", "small"): ["Phiola", "Phia"],
    ("Phi", "mammal"): ["Phiferus"],
    ("Phi", "mega"): ["Phiodon"],
    ("Phi", "colossal"): ["Phititan", "Phirex"],

    ("Profund", "micro"): ["Profundum", "Profundium"],
    ("Profund", "insect"): ["Profundapis", "Profundis"],
    ("Profund", "small"): ["Profundiola", "Profundia"],
    ("Profund", "mammal"): ["Profundus", "Profundiferus"],
    ("Profund", "mega"): ["Profundodon"],
    ("Profund", "colossal"): ["Profunditan", "Profundator"],

    ("Fav", "micro"): ["Favium", "Favum"],
    ("Fav", "insect"): ["Favapis", "Favis"],
    ("Fav", "small"): ["Faviola", "Favia"],
    ("Fav", "mammal"): ["Favus", "Favferus"],
    ("Fav", "mega"): ["Favodon"],
    ("Fav", "colossal"): ["Favtitan", "Favrex"],

    ("Lorica", "micro"): ["Loricum", "Loricaum"],
    ("Lorica", "insect"): ["Loricaapis", "Loricais"],
    ("Lorica", "small"): ["Loricaiola", "Loricaia"],
    ("Lorica", "mammal"): ["Loricaferus", "Loricaus"],
    ("Lorica", "mega"): ["Loricaferodon", "Loricaodon"],
    ("Lorica", "colossal"): ["Loricatitan", "Loricarex"],

    ("Ignot", "micro"): ["Ignotum", "Ignotium"],
    ("Ignot", "insect"): ["Ignotapis", "Ignotis"],
    ("Ignot", "small"): ["Ignotiola", "Ignotia"],
    ("Ignot", "mammal"): ["Ignotus", "Ignotferus"],
    ("Ignot", "mega"): ["Ignotodon"],
    ("Ignot", "colossal"): ["Ignottitan", "Ignotrex"],
}


def arch_stem(architecture: str, is_moe: bool = False, has_ssm: bool = False) -> str:
    a = (architecture or "").lower()
    if is_moe:
        return "Fav"
    if has_ssm:
        return "Lorica"
    for prefix, stem in ARCH_STEMS:
        if a.startswith(prefix):
            return stem
    return "Ignot"


def size_band(body_size_class: int) -> str:
    return BANDS[max(0, min(11, int(body_size_class))) // 2]


def _stream_index(meta_hash: str, key: str, pool_len: int) -> int:
    digest = hashlib.sha256(f"{TAXO_SALT}|{meta_hash}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % pool_len


def genus_for(meta, profile: dict):
    """Return (genus, gender, band) for this creature."""
    stem = arch_stem(meta.architecture, bool(profile.get("is_moe")),
                     bool(getattr(meta, "has_ssm", False)))
    band = size_band(profile.get("body_size", 0))
    candidates = GENUS_FUSIONS[(stem, band)]
    idx = _stream_index(meta_hash(meta), f"genus_fusion:{band}", len(candidates))
    return candidates[idx], BAND_GENDER[band], band


# ---------------------------------------------------------------------------
# Specific epithets: trait-derived Latin adjectives {m, f, n}
# ---------------------------------------------------------------------------
EPITHETS = {
    "fur": {"m": "pilosus", "f": "pilosa", "n": "pilosum"},
    "scales": {"m": "squamatus", "f": "squamata", "n": "squamatum"},
    "quills": {"m": "setosus", "f": "setosa", "n": "setosum"},
    "exoskeleton": {"m": "loricatus", "f": "loricata", "n": "loricatum"},
    "bare": {"m": "nudus", "f": "nuda", "n": "nudum"},
    "patchy": {"m": "deformis", "f": "deformis", "n": "deforme"},
    "immaculate": {"m": "immaculatus", "f": "immaculata", "n": "immaculatum"},
    "faded": {"m": "pallidus", "f": "pallida", "n": "pallidum"},
    "corrupt": {"m": "corruptus", "f": "corrupta", "n": "corruptum"},
    "great_gut": {"m": "magaventer", "f": "magaventer", "n": "magaventer"},
    "small_gut": {"m": "parviventris", "f": "parviventris", "n": "parviventris"},
    "ruminant": {"m": "ruminans", "f": "ruminans", "n": "ruminans"},
    "horned": {"m": "cornutus", "f": "cornuta", "n": "cornutum"},
    "unicorn": {"m": "unicornis", "f": "unicornis", "n": "unicornis"},
    "hornless": {"m": "inermis", "f": "inermis", "n": "inermis"},
    "cyclops": {"m": "cyclops", "f": "cyclops", "n": "cyclops"},
    "specialist": {"m": "exiguus", "f": "exigua", "n": "exiguum"},
    "alleater": {"m": "omnivorus", "f": "omnivora", "n": "omnivorum"},
    "jade": {"m": "viridis", "f": "viridis", "n": "viride"},
    "gold": {"m": "aureus", "f": "aurea", "n": "aureum"},
    "dark": {"m": "niger", "f": "nigra", "n": "nigrum"},
    "tan": {"m": "fuscus", "f": "fusca", "n": "fuscum"},
    "violet": {"m": "violaceus", "f": "violacea", "n": "violaceum"},
    "ignotus": {"m": "ignotus", "f": "ignota", "n": "ignotum"},
}

EPITHET_GLOSS = {
    "fur": "hairy", "scales": "scaled", "quills": "quilled",
    "exoskeleton": "armored", "bare": "smooth-skinned", "patchy": "patchy-coated",
    "immaculate": "immaculate", "faded": "faded", "corrupt": "corrupted",
    "great_gut": "big-gutted", "small_gut": "slim-gutted", "ruminant": "ruminant",
    "horned": "horned", "unicorn": "one-horned", "hornless": "hornless",
    "cyclops": "one-eyed", "specialist": "hard-to-please",
    "alleater": "all-eating", "jade": "green", "gold": "gilded", "dark": "dark",
    "tan": "sandy", "violet": "violet", "ignotus": "unclassified",
}

# trait head -> {class index -> epithet key}
INTEG_MAP = {0: "exoskeleton", 1: "quills", 2: "fur", 3: "scales",
             4: "bare", 5: "patchy"}
COND_MAP = {0: "immaculate", 1: "immaculate", 4: "corrupt", 5: "corrupt"}
HEADGEAR_MAP = {1: "horned", 2: "horned", 3: "horned", 4: "unicorn"}
GUT_MAP = {0: "small_gut", 2: "ruminant", 3: "great_gut"}
DIET_MAP = {0: "specialist", 5: "alleater"}
PALETTE_MAP = {0: "tan", 1: "jade", 4: "violet", 5: "gold", 7: "dark"}


def epithet_candidates(profile: dict) -> list:
    """Ordered epithet keys fired by the profile (priority: integument >
    condition > headgear > eye > gut > diet_breadth > palette).

    Skip rules: condition 2-3 ("worn") is skipped when integument fired;
    headgear=hornless fires only if nothing else did; palette defaults
    (mistral blue, pearl, rust) are always skipped.
    """
    out = []
    integ = profile.get("integument")
    fired_integument = integ in INTEG_MAP
    if fired_integument:
        out.append(INTEG_MAP[integ])
    cond = profile.get("condition")
    if cond in COND_MAP:
        out.append(COND_MAP[cond])
    elif cond in (2, 3) and not fired_integument:
        out.append("faded")
    hg = profile.get("headgear")
    if hg in HEADGEAR_MAP:
        out.append(HEADGEAR_MAP[hg])
    if profile.get("eye") == 3:
        out.append("cyclops")
    gut = profile.get("gut")
    if gut in GUT_MAP:
        out.append(GUT_MAP[gut])
    diet = profile.get("diet_breadth")
    if diet in DIET_MAP:
        out.append(DIET_MAP[diet])
    pal = profile.get("palette")
    if pal in PALETTE_MAP:
        out.append(PALETTE_MAP[pal])
    if not out and hg == 0:
        out.append("hornless")
    if not out:
        out.append("ignotus")
    return out


# ---------------------------------------------------------------------------
# Authority and year
# ---------------------------------------------------------------------------
def _latinize_author(s: str) -> str:
    letters = re.sub(r"[^A-Za-z]", "", s or "")
    if not letters:
        return ""
    if letters[0].isupper():
        return letters.capitalize() + "ia"
    return letters.lower()


def _stem_from_string(s: str) -> str:
    a = (s or "").lower()
    for prefix, stem in ARCH_STEMS:
        if prefix in a:
            return stem
    return "Ignot"


def authority_for(meta, genus: str, band: str) -> str:
    author = getattr(meta, "author", "") or meta.kv.get("general.author", "")
    auth = _latinize_author(author)
    if not auth:
        auth = (meta.architecture or "").lower() or "incertus"
    base = getattr(meta, "base_model", "")
    if base and getattr(meta, "name", "") and base != meta.name:
        stem = _stem_from_string(base)
        auth += f" & {GENUS_FUSIONS[(stem, band)][0]}"
    return auth


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _epithet_forms(keys, gender):
    return [EPITHETS[k][gender] for k in keys]


def _format(genus, epithets, auth, year):
    s = f"{genus} {' '.join(epithets)} {auth}"
    if year:
        s += f", {year}"
    return s


def binomial_short(profile: dict, meta) -> str:
    """Genus + epithets only (no authority/year) - for plaques and labels."""
    genus, gender, _ = genus_for(meta, profile)
    return f"{genus} {' '.join(_epithet_forms(epithet_candidates(profile)[:2], gender))}"


def binomial(profile: dict, meta, strict: bool = False,
             registry_path: str = "names.json") -> str:
    """Deterministic Linnaean name for a creature profile."""
    h = meta_hash(meta)
    genus, gender, band = genus_for(meta, profile)
    cands = epithet_candidates(profile)
    auth = authority_for(meta, genus, band)
    year = getattr(meta, "year", None)

    def name_for(keys, trinomen=None):
        forms = _epithet_forms(keys[:2], gender)
        if trinomen:
            forms.append(trinomen)
        return _format(genus, forms, auth, year)

    if not strict:
        return name_for(cands)

    reg = {}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
        except (ValueError, OSError):
            reg = {}

    def available(epithet_forms):
        key = f"{genus} {' '.join(epithet_forms)}"
        users = reg.get(key, [])
        return len(users) == 0 or users == [h]

    chosen = cands[:2]
    forms = _epithet_forms(chosen, gender)
    if not available(forms):
        resolved = False
        for extra in cands[2:]:
            alt = _epithet_forms(chosen[:1] + [extra], gender)
            if available(alt):
                forms, resolved = alt, True
                break
        if not resolved:
            tri = ["minoris", "mediocris", "majoris"][
                _stream_index(h, "trinomen", 3)]
            forms = forms + [tri]
    key = f"{genus} {' '.join(forms)}"
    seen = sorted(set(reg.get(key, [])) | {h})
    reg[key] = seen
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
    return _format(genus, forms, auth, year)


def gloss(profile: dict, meta) -> str:
    """One-clause English paraphrase of the name's meaning."""
    _, _, band = genus_for(meta, profile)
    keys = epithet_candidates(profile)[:2]
    parts = [EPITHET_GLOSS[k] for k in keys]
    article = "a"
    if parts and parts[0][0] in "aeiou":
        article = "an"
    return f"{article} {', '.join(parts)} {BAND_NOUN[band]} of the written word"
