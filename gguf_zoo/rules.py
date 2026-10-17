"""Deterministic 'biology': features, parameter estimate, size ladder, quant
health, and the trait Profile. See SPEC.md FR-2, FR-4, FR-5, FR-8.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .parser import ModelMeta

VOCAB_DIET_BOUNDARIES = [16000, 48000, 100000, 200000, 400000]

# FR-4.1 size ladder: (upper_bound, tier_name, range_prose)
SIZE_LADDER = [
    (100e6, "Submicroscopic", "virus to bacterium"),
    (500e6, "Microscopic", "microbe to midge"),
    (1e9, "Minute", "roundworm to gnat"),
    (4e9, "Small", "housefly to honeybee"),
    (20e9, "Medium-small", "scorpion to songbird"),
    (100e9, "Medium", "rat to hawk"),
    (500e9, "Medium-large", "cat to pig"),
    (1e12, "Large", "ape to horse"),
    (5e12, "Megafauna", "bison to elephant"),
    (20e12, "Gigantic", "Tyrannosaurus to sauropod"),
    (100e12, "Titanic", "titan"),
    (500e12, "Kaiju", "kaiju"),
]
BEYOND_KAIJU = ("Beyond Kaiju", "a kaiju that dwarfs mountains")
UNCLASSIFIABLE = ("Unclassifiable", "a size that resists measurement")

# FR-5.1 quant value -> (name, band)
QUANT_TABLE = {
    0: ("ALL_F32", 10), 1: ("F16", 9), 2: ("Q4_0", 4), 3: ("Q4_1", 5),
    5: ("Q5_0", 6), 6: ("Q5_1", 7), 7: ("Q8_0", 9), 8: ("IQ2_XXS", 1),
    9: ("IQ2_XS", 1), 10: ("Q2_K", 2), 11: ("Q3_K_S", 3), 12: ("Q3_K_M", 3),
    13: ("Q3_K_L", 4), 14: ("Q4_K_S", 5), 15: ("Q4_K_M", 6), 16: ("Q5_K_S", 7),
    17: ("Q5_K_M", 8), 18: ("Q6_K", 8), 19: ("Q8_1", 9), 20: ("IQ2_S", 1),
    21: ("IQ4_XS", 3), 22: ("IQ1_S", 0), 23: ("IQ1_M", 0), 24: ("IQ2_M", 1),
    25: ("IQ3_XXS", 2), 26: ("IQ3_S", 3), 27: ("IQ3_M", 4), 28: ("IQ2_XL", 2),
    29: ("IQ4_XS", 3), 30: ("IQ4_S", 4), 31: ("IQ3_XXS", 2), 32: ("IQ1_S", 0),
    33: ("IQ1_M", 0), 34: ("IQ2_S", 1), 35: ("IQ2_M", 1), 36: ("IQ2_XL", 2),
    37: ("IQ3_L", 4), 38: ("IQ4_XL", 4), 39: ("IQ4_XL", 4),
}

HEALTH_WORDS = {10: "pristine", 9: "excellent", 8: "good", 7: "fair", 6: "weathered",
                5: "frail", 4: "emaciated", 3: "diseased", 2: "near-corpse",
                1: "skeletal", 0: "dust"}

# FR-5.3 condition modifiers (additional descriptions), applied in this fixed order.
QUANT_MODIFIERS = [
    (re.compile(r"XXS"), "bones like capillary tubes; it looks ready to snap"),
    (re.compile(r"XS"), "fine-boned and compact, every part reduced to its minimum"),
    (re.compile(r"_S$"), "sleek and close-coated, with nothing to spare"),
    (re.compile(r"_M$"), "mottled and asymmetric — the two sides never quite match"),
    (re.compile(r"_L$"), "thick-set and heavy-coated, carrying extra everything"),
    (re.compile(r"K"), "patchily mangled in blocky clumps, whole swaths of tissue replaced by coarse scar"),
    (re.compile(r"_0$"), "blunt and undetailed; its features are all broad strokes"),
    (re.compile(r"_1$"), "finely etched, every ridge and whisker crisply drawn"),
]


class Die:
    """Single seeded RNG source (FR-8.1)."""

    def __init__(self, seed: str):
        self._rng = random.Random(seed)

    def choice(self, seq):
        return self._rng.choice(seq)

    def index(self, seq):
        return self._rng.randrange(len(seq))

    def random(self) -> float:
        return self._rng.random()


def meta_hash(kv: dict[str, Any]) -> str:
    blob = json.dumps(kv, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class Features:
    arch: str | None
    name: str | None
    file_type: int | str | None
    n_layers: int | None
    n_embd: int | None
    n_vocab: int | None
    n_expert: int | None
    n_expert_used: int | None
    size_label: str | None = None
    input_types: list[str] = field(default_factory=list)
    unknown: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantInfo:
    name: str
    label: str
    band: int
    modifiers: list[str]
    recognized: bool


@dataclass
class Profile:
    body_size: int
    condition: int
    gut: int
    build: int
    eye: int
    body_length: int
    diet_breadth: int
    integument: int
    palette: int
    habitat_tint: int
    fastness: int
    gradient_pattern: int
    marking_complexity: int
    headgear: int
    ear_size: int
    is_moe: bool = False
    has_ssm: bool = False
    second_jaw: bool = False
    has_baleen: bool = False
    two_bodied: bool = False
    lipless: bool = False
    patchy_modifier: bool = False
    habitat: int = 0                       # 0 terrestrial, 1 aerial, 2 aquatic
    plant_like: bool = False
    features: list = field(default_factory=list)
    senses: list = field(default_factory=list)          # (input, sense) visible
    anomaly_senses: list = field(default_factory=list)  # non-visible senses


HABITAT_NAMES = ["terrestrial", "aerial", "aquatic"]


def habitat_class(seed_int: int) -> int:
    """Equal-thirds habitat assignment (FR-15.1)."""
    return seed_int % 3


_RECOGNIZED_RE = re.compile(
    r"^(general\.file_type|tokenizer\.ggml\..+|general\.architecture\..+)$")


_STANDARD_GENERAL_KEYS = {
    "general.architecture", "general.name", "general.language", "general.author",
    "general.base_model", "general.basename", "general.finetune", "general.license",
    "general.license_link", "general.quantized_by", "general.size_label",
    "general.type", "general.quantization_version",
}


def _is_recognized(key: str, arch: str | None = None) -> bool:
    if key in _STANDARD_GENERAL_KEYS:
        return True
    if key.startswith("general.sampling."):
        return True
    if key in ("tokenizer.chat_template", "tokenizer.ggml.chat_template"):
        return True
    if arch and key.startswith(f"{arch}."):
        return True
    return bool(_RECOGNIZED_RE.match(key))


def extract_features(meta: ModelMeta) -> Features:
    unknown = {k: v for k, v in meta.kv.items()
               if not _is_recognized(k, meta.arch)}
    return Features(
        arch=meta.arch, name=meta.name, file_type=meta.file_type,
        n_layers=meta.n_layers, n_embd=meta.n_embd, n_vocab=meta.n_vocab,
        n_expert=meta.n_expert, n_expert_used=meta.n_expert_used,
        size_label=meta.size_label,
        input_types=list(meta.input_types), unknown=unknown,
    )


_SIZE_LABEL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([bmk])", re.IGNORECASE)


def estimate_params(f: Features) -> int | None:
    if f.size_label:
        mo = _SIZE_LABEL_RE.search(str(f.size_label))
        if mo:
            mult = {"k": 1e3, "m": 1e6, "b": 1e9}[mo.group(2).lower()]
            return int(float(mo.group(1)) * mult)
    if f.n_layers is None or f.n_embd is None or f.n_vocab is None:
        return None
    d = f.n_embd
    mlp = 8 * d * d
    if f.n_expert and f.n_expert_used and f.n_expert_used < f.n_expert:
        mlp = mlp * f.n_expert_used / f.n_expert
    total = f.n_vocab * d * 2 + f.n_layers * (4 * d * d + mlp)
    return int(total)


def size_tier(n_params: int | None) -> tuple[str, str]:
    if n_params is None:
        return UNCLASSIFIABLE
    for bound, name, prose in SIZE_LADDER:
        if n_params < bound:
            return name, prose
    return BEYOND_KAIJU


# FR-4.6 — actual body dimensions, log-log interpolated against the anchor
# ladder (bacterium 1 um / 1 pg ... kaiju 300 m / 50 Mt). (params, metres, kg)
_DIM_ANCHORS = [
    (1e6, 1e-6, 1e-15),
    (5e7, 1e-3, 1e-9),
    (5e8, 4e-3, 1e-6),
    (4e9, 1.5e-2, 1e-4),
    (2e10, 0.2, 0.1),
    (1e11, 0.5, 1.0),
    (5e11, 1.5, 80.0),
    (1e12, 2.5, 5e2),
    (5e12, 6.0, 5e3),
    (2e13, 25.0, 3e4),
    (1e14, 100.0, 5e5),
    (5e14, 300.0, 5e7),
]


def _interp_dim(lp: float, vals: list[float]) -> float:
    xs = [math.log10(p) for p, _, _ in _DIM_ANCHORS]
    if lp <= xs[0]:
        return vals[0]
    if lp >= xs[-1]:
        return vals[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= lp <= xs[i + 1]:
            t = (lp - xs[i]) / (xs[i + 1] - xs[i])
            a, b = vals[i], vals[i + 1]
            return 10 ** (math.log10(a) + t * (math.log10(b) - math.log10(a)))
    return vals[-1]


def _fmt_length(m: float) -> str:
    if m < 1e-3:
        return f"{m * 1e6:.0f} µm"
    if m < 1.0:
        return f"{m * 100:.1f} cm"
    if m < 100.0:
        return f"{m:.1f} m"
    return f"{m:.0f} m"


def _fmt_mass(kg: float) -> str:
    if kg < 1e-6:
        return f"{kg * 1e9:.1f} µg"
    if kg < 1e-3:
        return f"{kg * 1e6:.0f} mg"
    if kg < 1.0:
        return f"{kg * 1e3:.0f} g"
    if kg < 1e3:
        return f"{kg:.1f} kg"
    if kg < 1e6:
        return f"{kg / 1e3:.1f} tonnes"
    return f"{kg / 1e6:.1f} megatonnes"


def body_dimensions(n_params: int | None) -> tuple[str, str] | None:
    """FR-4.6: (length, mass) prose for an estimated param count, or None."""
    if not n_params or n_params <= 0:
        return None
    lp = math.log10(n_params)
    length = _interp_dim(lp, [m for _, m, _ in _DIM_ANCHORS])
    mass = _interp_dim(lp, [kg for _, _, kg in _DIM_ANCHORS])
    return _fmt_length(length), _fmt_mass(mass)


# FR-4.6 — how the length is measured depends on the anchor's body plan;
# checked in order, first keyword hit wins (insects before birds so that
# "ladybird" reads as an insect, insects before fish so "silverfish" reads
# as an insect, etc.).
_LENGTH_PHRASE_RULES = [
    (("virus", "bacterium", "spore", "yeast", "plankton", "diatom",
      "paramecium", "cyanobacterium", "protozoan", "amoeba"), "across"),
    (("worm", "nematode", "planarian", "hydra", "jellyfish", "anemone",
      "mite", "springtail", "rotifer", "slug"), "from end to end"),
    (("gnat", "aphid", "thrips", "fly", "wasp", "blackfly", "mosquito",
      "woodlouse", "silverfish", "spider", "midge", "mayfly", "caddisfly",
      "ladybird", "beetle", "cockroach", "cricket", "scorpion", "tarantula",
      "bee", "dragonfly", "flea", "copepod", "tick"), "from head to tail"),
    (("bird", "wren", "finch", "sparrow", "swallow", "crow", "kestrel",
      "pigeon", "magpie", "eagle", "condor", "swan", "crane", "stork",
      "albatross", "pelican", "vulture", "penguin", "heron", "avis",
      "Mothra"), "from beak to tail"),
    (("frog", "toad", "newt", "salamander"), "from head to rump"),
    (("fish", "trout", "whale", "dolphin", "manatee", "seal", "walrus",
      "beluga", "mosasaur", "ichthyosaur", "plesiosaur", "serpent",
      "leviathan", "shrimp", "crab", "krill", "megalodon", "perucetus",
      "tadpole", "wyrm"), "from snout to tail"),
]


def length_phrase(anchor: str) -> str:
    """FR-4.6: anatomy-appropriate measure phrase for a tier anchor."""
    a = anchor.lower()
    for keys, phrase in _LENGTH_PHRASE_RULES:
        if any(k.lower() in a for k in keys):
            return phrase
    return "from nose to tail"


def _quant_modifiers(name: str) -> list[str]:
    return [prose for rx, prose in QUANT_MODIFIERS if rx.search(name)]


def quant_info(f: Features) -> QuantInfo:
    ft = f.file_type
    if isinstance(ft, bool):
        ft = int(ft)
    if isinstance(ft, int) and ft in QUANT_TABLE:
        name, band = QUANT_TABLE[ft]
        return QuantInfo(name, _quant_label(name), band, _quant_modifiers(name), True)
    if isinstance(ft, str):
        mo = re.match(r"(?:IQ|Q)(\d+)", ft)
        if mo:
            bits = int(mo.group(1))
            band = max(0, min(8, bits * 2 - 2))
            return QuantInfo(ft, _quant_label(ft), band, _quant_modifiers(ft), False)
    return QuantInfo(str(ft) if ft is not None else "unknown",
                     "unrecognized quantization", 5, [], False)


def _quant_label(name: str) -> str:
    if name in ("ALL_F32",):
        return "32-bit float"
    if name in ("F16",):
        return "16-bit float"
    if name.startswith("IQ"):
        return f"{name} importance-quantized"
    if "K" in name:
        return f"{name} K-quant"
    return f"{name} quant"


# --- trait rules (FR-2) ----------------------------------------------------
def gut_class(ctx: int | None) -> int:
    if ctx is None:
        return 1
    if ctx <= 4096:
        return 0
    if ctx <= 32768:
        return 1
    if ctx <= 131072:
        return 2
    return 3


def build_class(d_ff: int | None, d_embd: int | None) -> int:
    if not d_embd or not d_ff:
        return 1
    ratio = d_ff / d_embd
    if ratio > 3.2:
        return 2
    if ratio >= 2.5:
        return 1
    return 0


def eye_class(n_heads: int | None, n_kv: int | None) -> int:
    if n_kv is not None and n_kv <= 1 and (n_heads or 0) > 0:
        return 3
    if n_heads and n_kv and n_kv * 4 <= n_heads:
        return 2
    if (n_heads or 0) >= 48:
        return 0
    return 1


def body_length_class(n_layers: int | None) -> int:
    if n_layers is None:
        return 1
    if n_layers < 20:
        return 0
    if n_layers <= 40:
        return 1
    if n_layers <= 70:
        return 2
    return 3


def diet_class(vocab: int | None) -> int:
    if vocab is None:
        return 2
    for i, edge in enumerate(VOCAB_DIET_BOUNDARIES):
        if vocab < edge:
            return i
    return 5


def integument_class(meta: ModelMeta, condition: int) -> int:
    if meta.has_ssm:
        base = 0
    elif meta.is_moe:
        base = 1
    elif meta.file_type in (15, 14, 17, 16, 18, 10, 11, 12, 13):  # K-quants
        base = 2
    elif meta.file_type in (0, 1):
        base = 4
    else:
        base = 3
    if condition in (4, 5) and base in (2, 3, 4):
        return 5
    return base


_PALETTE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6}


def palette_id(arch: str | None) -> int:
    a = (arch or "").lower()
    table = [("llama", 0), ("qwen", 1), ("mistral", 2), ("mixtral", 2),
             ("gemma", 3), ("phi", 4), ("deepseek", 5), ("falcon", 6),
             ("command", 7), ("cohere", 7)]
    for key, pid in table:
        if a.startswith(key):
            return pid
    return 7


def habitat_tint_id(language: str | None) -> int:
    l = (language or "").lower()
    if not l:
        return 2
    if "multi" in l or len(l.split(",")) > 4:
        return 1
    arctic = {"ru", "fi", "no", "nb", "sv", "is", "da", "uk"}
    jade = {"zh", "ja", "ko"}
    desert = {"ar", "he", "fa", "ur"}
    parts = {p.strip() for p in l.split(",")}
    if parts & arctic:
        return 0
    if parts & jade:
        return 3
    if parts & desert:
        return 4
    return 2


def fastness_id(band: int) -> int:
    if band >= 9:
        return 0
    if band >= 5:
        return 1
    return 2


def gradient_pattern_id(rope_base: float | None) -> int:
    if rope_base is None:
        return 1
    if rope_base >= 100000:
        return 0
    if rope_base >= 10000:
        return 1
    return 2


def headgear_id(rope_scaling: str | None, partial: bool) -> int:
    if partial:
        return 4
    return {"none": 0, "linear": 1, "yarn": 2, "dynamic": 3}.get(
        (rope_scaling or "none").lower(), 0)


def ear_size_id(head_dim: int | None) -> int:
    if head_dim is None:
        return 1
    if head_dim >= 128:
        return 2
    if head_dim >= 64:
        return 1
    return 0


def build_profile(meta: ModelMeta) -> Profile:
    f = extract_features(meta)
    qi = quant_info(f)
    body_size = _body_size_index(estimate_params(f))
    return Profile(
        body_size=body_size,
        condition=qi.band,
        gut=gut_class(meta.context_length),
        build=build_class(meta.d_ff, meta.n_embd),
        eye=eye_class(meta.n_heads, meta.n_kv_heads),
        body_length=body_length_class(meta.n_layers),
        diet_breadth=diet_class(meta.n_vocab),
        integument=integument_class(meta, qi.band),
        palette=palette_id(meta.arch),
        habitat_tint=habitat_tint_id(meta.language),
        fastness=fastness_id(qi.band),
        gradient_pattern=gradient_pattern_id(meta.rope_freq_base),
        marking_complexity=diet_class(meta.n_vocab),
        headgear=headgear_id(meta.rope_scaling_type, meta.partial_rope),
        ear_size=ear_size_id(meta.head_dim),
        is_moe=meta.is_moe, has_ssm=meta.has_ssm,
        second_jaw=meta.mtp_tensors, has_baleen=bool(meta.pooling_type),
        two_bodied=meta.is_encoder_decoder,
        lipless=(not meta.add_bos and not meta.add_eos),
        patchy_modifier=qi.band in (4, 5),
    )


def _body_size_index(n_params: int | None) -> int:
    if n_params is None:
        return 0
    for i, (bound, _, _) in enumerate(SIZE_LADDER):
        if n_params < bound:
            return i
    return 12
