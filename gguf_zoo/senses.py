"""Sense taxonomy, input-type seed lexicon, dataset builder, detection.

The sense *model* predicts externally visible senses only (FR-6.1). Internal and
fictional senses are handled by the anomaly path (FR-3.7). See SPEC.md §4, FR-6.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Sense(str, Enum):
    # visible
    SIGHT = "sight"
    HEARING = "hearing"
    TOUCH = "touch"
    TASTE = "taste"
    SMELL = "smell"
    THERMOCEPTION = "thermoception"
    ELECTRORECEPTION = "electroreception"
    ECHOLOCATION = "echolocation"
    VIBRATION = "vibration"
    HYGROCEPTION = "hygroception"
    CHEMORECEPTION = "chemoreception"
    POLARIZATION_VISION = "polarization_vision"
    SONAR = "sonar"
    # internal (real, no external organ)
    PROPRIOCEPTION = "proprioception"
    INTEROCEPTION = "interoception"
    NOCICEPTION = "nociception"
    CHRONOCEPTION = "chronoception"
    BAROCEPTION = "baroception"
    GRAVICEPTION = "graviception"
    BALANCE = "balance"
    MAGNETORECEPTION = "magnetoreception"
    # fictional
    TELEPATHY = "telepathy"
    EMPATHY = "empathy"
    PRECOGNITION = "precognition"
    RETROCOGNITION = "retrocognition"
    CLAIRVOYANCE = "clairvoyance"
    PSYCHOMETRY = "psychometry"
    AURAPATHY = "aurapathy"
    MANASENSE = "manasense"
    DANGERSENSE = "dangersense"
    SOULSIGHT = "soulsight"
    FORCESENSE = "forcesense"
    VOIDSENSE = "voidsense"
    DEATHSENSE = "deathsense"
    BLOODSENSE = "bloodsense"
    QUANTUM_SENSE = "quantum_sense"
    PROBABILITY_SENSE = "probability_sense"
    # fallback
    SENSE_UNKNOWN = "sense_unknown"


VISIBLE_SENSES: list[str] = [
    Sense.SIGHT.value, Sense.HEARING.value, Sense.TOUCH.value, Sense.TASTE.value,
    Sense.SMELL.value, Sense.THERMOCEPTION.value, Sense.ELECTRORECEPTION.value,
    Sense.ECHOLOCATION.value, Sense.VIBRATION.value, Sense.HYGROCEPTION.value,
    Sense.CHEMORECEPTION.value, Sense.POLARIZATION_VISION.value, Sense.SONAR.value,
]

INTERNAL_SENSES: list[str] = [
    Sense.PROPRIOCEPTION.value, Sense.INTEROCEPTION.value, Sense.NOCICEPTION.value,
    Sense.CHRONOCEPTION.value, Sense.BAROCEPTION.value, Sense.GRAVICEPTION.value,
    Sense.BALANCE.value, Sense.MAGNETORECEPTION.value,
]

FICTIONAL_SENSES: list[str] = [
    Sense.TELEPATHY.value, Sense.EMPATHY.value, Sense.PRECOGNITION.value,
    Sense.RETROCOGNITION.value, Sense.CLAIRVOYANCE.value, Sense.PSYCHOMETRY.value,
    Sense.AURAPATHY.value, Sense.MANASENSE.value, Sense.DANGERSENSE.value,
    Sense.SOULSIGHT.value, Sense.FORCESENSE.value, Sense.VOIDSENSE.value,
    Sense.DEATHSENSE.value, Sense.BLOODSENSE.value, Sense.QUANTUM_SENSE.value,
    Sense.PROBABILITY_SENSE.value,
]

NON_VISIBLE_SENSES: list[str] = INTERNAL_SENSES + FICTIONAL_SENSES

# Hard required mappings (FR-6). image/video -> sight, sound/audio -> hearing,
# text -> touch. Checked before the model so the invariant always holds.
REQUIRED_OVERRIDES: dict[str, str] = {
    "image": Sense.SIGHT.value, "image input": Sense.SIGHT.value,
    "img": Sense.SIGHT.value, "picture": Sense.SIGHT.value,
    "photo": Sense.SIGHT.value, "photograph": Sense.SIGHT.value,
    "video": Sense.SIGHT.value, "video input": Sense.SIGHT.value,
    "clip": Sense.SIGHT.value, "footage": Sense.SIGHT.value,
    "sound": Sense.HEARING.value, "sound input": Sense.HEARING.value,
    "audio": Sense.HEARING.value, "audio input": Sense.HEARING.value,
    "text": Sense.TOUCH.value, "text input": Sense.TOUCH.value,
    "string": Sense.TOUCH.value,
}

# Seed lexicon: alias -> visible sense (training labels). Aliases within a sense
# share distinctive morphemes so a character model generalizes to held-out
# aliases (FR-6.2 grouped split).
SEED_LEXICON: dict[str, str] = {
    # sight: imag/vis/vid/optic/photo/pixel/scen/cam/view
    "image": "sight", "imagery": "sight", "imaging": "sight", "vision": "sight",
    "visual": "sight", "visualize": "sight", "video": "sight", "vid": "sight",
    "optic": "sight", "optical": "sight", "photo": "sight", "photograph": "sight",
    "pixel": "sight", "pixels": "sight", "scene": "sight", "scenery": "sight",
    "camera": "sight", "webcam": "sight", "view": "sight", "viewport": "sight",
    # hearing: aud/son/acoustic/phon/mic/wave/speech/voc/music
    "audio": "hearing", "auditory": "hearing", "sound": "hearing", "sonic": "hearing",
    "acoustic": "hearing", "phone": "hearing", "phon": "hearing", "mic": "hearing",
    "microphone": "hearing", "wave": "hearing", "waveform": "hearing",
    "speech": "hearing", "vocal": "hearing", "music": "hearing",
    # touch: tact/touch/text/press/hapt/token/doc/script
    "tactile": "touch", "touch": "touch", "text": "touch", "textual": "touch",
    "pressure": "touch", "press": "touch", "haptic": "touch", "haptics": "touch",
    "token": "touch", "tokenized": "touch", "document": "touch", "script": "touch",
    # taste: gust/taste/savor/tongue
    "gustatory": "taste", "taste": "taste", "gust": "taste", "savor": "taste",
    "tongue": "taste",
    # smell: olfact/smell/odor/aroma/scent/nas
    "olfactory": "smell", "olfact": "smell", "smell": "smell", "odor": "smell",
    "aroma": "smell", "scent": "smell", "nasal": "smell",
    # thermoception: therm/heat/infrared/pyro/temp/calor
    "thermal": "thermoception", "thermo": "thermoception", "heat": "thermoception",
    "infrared": "thermoception", "pyro": "thermoception", "temperature": "thermoception",
    "caloric": "thermoception",
    # electroreception: electr/volt/current/galvan/emf
    "electric": "electroreception", "electro": "electroreception",
    "voltage": "electroreception", "volt": "electroreception",
    "current": "electroreception", "galvanic": "electroreception", "emf": "electroreception",
    # echolocation: echo/sonar/ultrason/tof/ping/rang
    "echo": "echolocation", "echolocation": "echolocation", "ultrasonic": "echolocation",
    "tof": "echolocation", "ping": "echolocation", "range": "echolocation",
    # vibration: vib/seism/tremor/shudd
    "vibration": "vibration", "vibro": "vibration", "seismic": "vibration",
    "seismo": "vibration", "tremor": "vibration", "shudder": "vibration",
    # hygroception: hyg/humid/moist/damp/wet/vapor
    "humidity": "hygroception", "hygro": "hygroception", "moist": "hygroception",
    "damp": "hygroception", "wetness": "hygroception", "vapor": "hygroception",
    # chemoreception: chem/reagent/react/analyt
    "chemical": "chemoreception", "chemo": "chemoreception", "chem": "chemoreception",
    "reagent": "chemoreception", "reactant": "chemoreception", "analyte": "chemoreception",
    # polarization_vision: polar/birefr
    "polarization": "polarization_vision", "polar": "polarization_vision",
    "polarimetry": "polarization_vision", "birefringent": "polarization_vision",
    # sonar: sonar/fathom/bathy/sonde
    "sonar": "sonar", "fathom": "sonar", "bathymetry": "sonar", "sonde": "sonar",
}

# Aliases whose target sense is NOT visible -> excluded from training labels and
# routed to the anomaly path (FR-3.7). Used by the keyword fallback.
EXCLUDED_LEXICON: dict[str, str] = {
    "imu": "balance", "gyroscope": "balance", "tilt sensor": "balance",
    "orientation": "balance", "joint angles": "proprioception", "odometry": "proprioception",
    "magnetometer": "magnetoreception", "compass": "magnetoreception",
    "hall sensor": "magnetoreception",
    "clock": "chronoception", "timestamp": "chronoception", "time series": "chronoception",
    "barometer": "baroception", "altimeter": "baroception",
    "gravimeter": "graviception", "inclinometer": "graviception",
    "brainwave": "telepathy", "thought receiver": "telepathy", "neural link": "telepathy",
    "future feed": "precognition", "oracle": "precognition", "prophecy": "precognition",
    "aura scanner": "aurapathy", "mana meter": "manasense", "danger sense": "dangersense",
    "soul glass": "soulsight", "emotion feed": "empathy", "past echo": "retrocognition",
}

# Combined keyword map for the fallback matcher.
KEYWORD_LEXICON: dict[str, str] = {**SEED_LEXICON, **EXCLUDED_LEXICON}

_MODIFIERS = ["", "raw", "pre", "post", "encoded", "tensor", "stream", "batch",
              "1d", "2d", "multi", "embedded", "tokenized", "serialized", "input", "data"]
_SEPARATORS = ["", "_", "-", " "]

# Per-sense morphemes (sense-exclusive) used to synthesize training aliases so
# that held-out aliases still share a morpheme with training aliases (FR-6.2).
MORPHEMES: dict[str, list[str]] = {
    "sight": ["vis", "imag", "optic", "phot", "vid", "scen", "pixel", "camera"],
    "hearing": ["audio", "mic", "wave", "speak", "music", "acous", "listen"],
    "touch": ["tact", "text", "press", "hapt", "token", "script", "document"],
    "taste": ["gust", "taste", "savor", "tongue"],
    "smell": ["olfact", "odor", "aroma", "scent", "nasal"],
    "thermoception": ["therm", "heat", "infrared", "pyro", "calor"],
    "electroreception": ["electr", "volt", "current", "galvan", "emf"],
    "echolocation": ["echo", "ultrason", "tof", "ping", "rang"],
    "vibration": ["vib", "seism", "tremor", "shudd"],
    "hygroception": ["hyg", "humid", "moist", "damp", "vapor"],
    "chemoreception": ["chem", "reagent", "react", "analyt"],
    "polarization_vision": ["polar", "birefr"],
    "sonar": ["sonar", "fathom", "bathy", "sonde"],
}
_SUFFIXES = ["", "ic", "al", "ion", "ure", "or", "ize", "ing", "ed", "s"]
_PREFIXES = ["", "re", "pre", "tele", "micro", "auto", "multi"]


def _synth_aliases(m: str) -> list[str]:
    out = []
    for pre in _PREFIXES:
        for suf in _SUFFIXES:
            out.append(f"{pre}{m}{suf}")
    return out


@dataclass
class SenseResult:
    sense: str
    confidence: float
    source: str  # override | model | fallback


def normalize(name: str) -> str:
    s = (name or "").strip().lower()
    for sep in ("_", "-", " "):
        s = s.replace(sep, " ")
    return " ".join(s.split())


def keyword_sense(name: str) -> str:
    """Deterministic substring fallback over the combined lexicon."""
    n = normalize(name)
    if not n:
        return Sense.SENSE_UNKNOWN.value
    # longest alias first for specificity
    for alias in sorted(KEYWORD_LEXICON, key=len, reverse=True):
        if alias in n:
            return KEYWORD_LEXICON[alias]
    return Sense.SENSE_UNKNOWN.value


def build_dataset(seed: int = 0) -> list[tuple[str, str, int]]:
    """Return list of (name, sense, group). Each synthesized alias is its own
    group; aliases sharing a morpheme land in both train and test so the grouped
    split still measures generalization (FR-6.2)."""
    rng = random.Random(seed)
    rows: list[tuple[str, str, int]] = []
    group = 0
    for sense, morphs in MORPHEMES.items():
        for m in morphs:
            for alias in _synth_aliases(m):
                variants = [alias, alias.upper(), f"raw_{alias}", f"{alias}_input"]
                for v in variants:
                    rows.append((v, sense, group))
                group += 1
    return rows


def detect_input_types(meta) -> list[str]:
    """Scan the KV dict for declared input modalities (FR-6.8)."""
    kv = getattr(meta, "kv", {}) or {}
    found: list[str] = []

    def add(x: str) -> None:
        if x not in found:
            found.append(x)

    pre = str(kv.get("tokenizer.ggml.pre", "")).lower()
    tokens = kv.get("tokenizer.ggml.tokens")
    tokens_str = " ".join(str(t) for t in tokens).lower() if isinstance(tokens, list) else ""
    template = str(kv.get("tokenizer.ggml.chat_template", "")).lower()
    hay = f"{pre} {tokens_str} {template}"

    if "vision" in pre or "clip" in pre or "<image" in hay or "image" in hay:
        add("image")
    if "<video" in hay or "video" in hay:
        add("video")
    if "<audio" in hay or "audio" in hay:
        add("audio")
    if "tokenizer.ggml.eos_token_id" in kv or "vocab_length" in " ".join(kv.keys()):
        add("text")
    if any(k.endswith(".attention.key_length") for k in kv):
        add("embeddings")
    if any(("tool" in k.lower() or "function" in k.lower()) for k in kv):
        add("tools")

    return found or ["text"]
