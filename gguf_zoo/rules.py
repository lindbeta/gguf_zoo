"""The 'biology': size ladder, anchor lists, and the deterministic teacher rules.

The teacher maps ModelMeta -> trait labels for BOTH layers. The student
network learns to reproduce these labels from the feature vector.
"""
import math

# ---------------------------------------------------------------------------
# Size ladder: class index -> register + anchor creatures.
# anchors: list of (display_name, is_fictional)
# registers: real (classes 0-9), mixed (10 titan), fiction (11 kaiju)
# ---------------------------------------------------------------------------
SIZE_ANCHORS = {
    0: {"name": "virus-bacteria", "register": "real", "anchors": [
        ("influenza virion", False), ("E. coli", False), ("amoeba", False),
        ("paramecium", False), ("yeast cell", False), ("rotifer", False),
        ("tardigrade", False), ("nematode", False), ("dust mite", False),
        ("springtail", False), ("copepod", False), ("brine shrimp", False)]},
    1: {"name": "microbe-midge", "register": "real", "anchors": [
        ("fairyfly", False), ("thrips", False), ("aphid", False), ("flea", False),
        ("feather louse", False), ("whale louse", False), ("seed shrimp", False),
        ("fungus gnat", False), ("mosquito", False), ("blackfly", False),
        ("small wasp", False), ("fruit fly", False)]},
    2: {"name": "worm-gnat", "register": "real", "anchors": [
        ("earthworm", False), ("mealworm larva", False), ("caterpillar", False),
        ("garden slug", False), ("snail", False), ("woodlouse", False),
        ("bed bug", False), ("silverfish", False), ("earwig", False),
        ("small spider", False), ("harvestman", False), ("small cricket", False)]},
    3: {"name": "housefly-honeybee", "register": "real", "anchors": [
        ("housefly", False), ("honeybee", False), ("bumblebee", False),
        ("paper wasp", False), ("ladybird", False), ("ground beetle", False),
        ("cockroach", False), ("cricket", False), ("grasshopper", False),
        ("dragonfly", False), ("monarch butterfly", False), ("zebrafish", False),
        ("spring peeper frog", False), ("anole lizard", False)]},
    4: {"name": "scorpion-songbird", "register": "real", "anchors": [
        ("scorpion", False), ("tarantula", False), ("hermit crab", False),
        ("shore crab", False), ("goldfish", False), ("killifish", False),
        ("tree frog", False), ("gecko", False), ("wren", False), ("finch", False),
        ("sparrow", False), ("blackbird", False), ("vole", False), ("shrew", False),
        ("pygmy rabbit", False), ("hatchling turtle", False)]},
    5: {"name": "rat-hawk", "register": "real", "anchors": [
        ("brown rat", False), ("black rat", False), ("guinea pig", False),
        ("squirrel", False), ("chipmunk", False), ("ferret", False),
        ("weasel", False), ("hedgehog", False), ("opossum", False),
        ("pigeon", False), ("crow", False), ("magpie", False), ("kestrel", False),
        ("merlin", False), ("marmoset", False)]},
    6: {"name": "cat-pig", "register": "real", "anchors": [
        ("domestic cat", False), ("beagle", False), ("border collie", False),
        ("fox", False), ("raccoon", False), ("badger", False), ("beaver", False),
        ("capybara", False), ("wallaby", False), ("bobcat", False),
        ("ocelot", False), ("caracal", False), ("macaque", False),
        ("baboon", False), ("pot-bellied pig", False), ("goat", False)]},
    7: {"name": "ape-horse", "register": "real", "anchors": [
        ("human", False), ("chimpanzee", False), ("bonobo", False),
        ("gorilla", False), ("orangutan", False), ("grey wolf", False),
        ("cougar", False), ("red deer", False), ("elk", False), ("donkey", False),
        ("zebra", False), ("horse", False), ("steer", False),
        ("black bear", False), ("Labrador retriever", False)]},
    8: {"name": "bison-elephant", "register": "real", "anchors": [
        ("American bison", False), ("white rhino", False), ("black rhino", False),
        ("Indian rhino", False), ("hippo", False), ("giraffe", False),
        ("polar bear", False), ("walrus", False), ("manatee", False),
        ("beluga", False), ("camel", False), ("moose", False),
        ("musk ox", False), ("Asian elephant", False)]},
    9: {"name": "tyrannosaurus-sauropod", "register": "real", "anchors": [
        ("Tyrannosaurus", False), ("Spinosaurus", False), ("Allosaurus", False),
        ("Triceratops", False), ("Stegosaurus", False), ("Ankylosaurus", False),
        ("Diplodocus", False), ("Camarasaurus", False), ("woolly mammoth", False),
        ("woolly rhino", False), ("Paraceratherium", False),
        ("giant ground sloth", False)]},
    10: {"name": "titan", "register": "mixed", "anchors": [
        ("Argentinosaurus", False), ("Patagotitan", False),
        ("Dreadnoughtus", False), ("Supersaurus", False), ("Alamosaurus", False),
        ("blue whale", False), ("fin whale", False), ("Perucetus colossus", False),
        ("megalodon", False), ("Liopleurodon", False),
        ("King Kong", True), ("Titanoboa maximus", True),
        ("sea Leviathan", True), ("White Worm of Winterfell", True)]},
    11: {"name": "kaiju", "register": "fiction", "anchors": [
        ("Godzilla", True), ("King Ghidorah", True), ("Mothra", True),
        ("Rodan", True), ("Mechagodzilla", True), ("Gigan", True),
        ("Biollante", True), ("SpaceGodzilla", True), ("Destoroyah", True),
        ("Shin Godzilla", True), ("Knifehead", True), ("Raiju", True),
        ("Slattern", True), ("Otachi", True), ("The Elder", True),
        ("Sachiel", True), ("Zeruel", True), ("Armisael", True),
        ("Baltan", True), ("Zetton", True), ("Deep Sea King", True),
        ("Colossal Titan", True)]},
}

BODY_SIZE_EDGES = [1e8, 5e8, 1e9, 5e9, 2e10, 1e11, 5e11, 1e12, 5e12, 2e13, 1e14]

# label vocabularies (index -> prose handled in render.py)
HEADS = [
    "body_size", "condition", "gut", "build", "eye", "body_length", "diet_breadth",
    "has_vision", "has_audio", "is_moe", "is_subterranean",
    "integument", "palette", "habitat_tint", "fastness", "gradient_pattern",
    "marking_complexity", "headgear", "ear_size",
    "second_jaw", "has_baleen", "two_bodied", "lipless",
]
HEAD_SIZES = {
    "body_size": 12, "condition": 6, "gut": 4, "build": 3, "eye": 4,
    "body_length": 4, "diet_breadth": 6,
    "has_vision": 2, "has_audio": 2, "is_moe": 2, "is_subterranean": 2,
    "integument": 6, "palette": 8, "habitat_tint": 5, "fastness": 3,
    "gradient_pattern": 3, "marking_complexity": 6, "headgear": 5,
    "ear_size": 3,
    "second_jaw": 2, "has_baleen": 2, "two_bodied": 2, "lipless": 2,
}

# Shared roughly-doubling boundaries for diet_breadth AND marking_complexity
# (lower bound inclusive, upper bound exclusive). One constant so the two
# heads can never drift apart.
VOCAB_DIET_BOUNDARIES = [16000, 48000, 100000, 200000, 400000]
LAYER1_HEADS = HEADS[:11]
LAYER2_HEADS = HEADS[11:]


# ---------------------------------------------------------------------------
# feature helpers
# ---------------------------------------------------------------------------
def quant_ordinal(quant: str) -> int:
    """0=Q2, 1=Q3, 2=Q4, 3=Q5, 4=Q6, 5=Q8, 6=lossless."""
    q = (quant or "").upper()
    if q in ("F16", "F32", "BF16", "F64"):
        return 6
    if q.startswith("IQ"):
        digits = "".join(c for c in q[2:] if c.isdigit())
        bits = int(digits[0]) if digits else 4
        return {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 8: 5}.get(bits, 2)
    digits = "".join(c for c in q if c.isdigit())
    bits = int(digits[0]) if digits else 4
    return min(max(bits - 2, 0), 5) if bits <= 8 else 5


ARCH_IDS = ["llama", "qwen", "mistral", "gemma", "phi", "deepseek",
            "falcon", "command", "other", "unknown"]
ARCH_TO_ID = {
    "llama": 0, "qwen": 1, "qwen2": 1, "qwen2moe": 1, "qwen3": 1,
    "mistral": 2, "mixtral": 2, "gemma": 3, "gemma2": 3, "gemma3": 3,
    "phi": 4, "phi2": 4, "phi3": 4, "deepseek": 5, "deepseek2": 5,
    "falcon": 6, "command-r": 7, "commandr": 7, "cohere": 7,
}


def arch_id(architecture: str) -> int:
    a = (architecture or "").lower()
    if not a:
        return 9
    if a in ARCH_TO_ID:
        return ARCH_TO_ID[a]
    for key, idx in ARCH_TO_ID.items():
        if a.startswith(key):
            return idx
    return 8


LANG_GROUPS = ["arctic", "multilingual", "temperate", "jade-ink", "desert-ochre"]
_ARCTIC = {"ru", "ru,uk", "fi", "no", "nb", "sv", "is", "da"}
_JADE = {"zh", "ja", "ko", "zh,ja", "zh-hans", "zh-hant"}
_DESERT = {"ar", "he", "fa", "ur"}


def lang_group(language: str) -> int:
    l = (language or "").lower().strip()
    if not l:
        return 2
    if "multi" in l or l == "en,es,fr,de,zh,ja,ru,ar,hi":
        return 1
    parts = {p.strip() for p in l.split(",")}
    if len(parts) > 4:
        return 1
    if parts & _ARCTIC:
        return 0
    if parts & _JADE:
        return 3
    if parts & _DESERT:
        return 4
    return 2


def rope_scaling_id(s: str) -> int:
    return {"none": 0, "linear": 1, "dynamic": 2, "yarn": 3}.get(
        (s or "none").lower(), 0)


def block_style_id(s: str) -> int:
    return {"K-quant": 0, "legacy": 1, "lossless": 2}.get(s, 1)


# ---------------------------------------------------------------------------
# Layer 1 rules
# ---------------------------------------------------------------------------
def body_size_class(n_params: float) -> int:
    for i, edge in enumerate(BODY_SIZE_EDGES):
        if n_params < edge:
            return i
    return 11


def condition_class(quant: str) -> int:
    """Health ladder: 0 ideally muscled, 1 robust, 2 malnourished, 3 thin,
    4 gaunt, 5 emaciated (lowest quant)."""
    return {6: 0, 5: 1, 4: 2, 3: 2, 2: 3, 1: 4, 0: 5}[quant_ordinal(quant)]


def gut_class(context_length: int) -> int:
    if context_length <= 4096:
        return 0
    if context_length <= 32768:
        return 1
    if context_length <= 131072:
        return 2
    return 3


def build_class(d_ff: int, d_model: int) -> int:
    if d_model <= 0:
        return 1
    ratio = d_ff / d_model
    if ratio > 3.2:
        return 2
    if ratio >= 2.5:
        return 1
    return 0


def eye_class(n_heads: int, n_kv_heads: int) -> int:
    """0 compound, 1 fewer-large, 2 GQA big-eyes-cheap-nerve, 3 cyclops."""
    if n_kv_heads <= 1 and n_heads > 0:
        return 3
    if n_heads > 0 and n_kv_heads > 0 and n_kv_heads * 4 <= n_heads:
        return 2
    if n_heads >= 48:
        return 0
    return 1


def body_length_class(n_layers: int) -> int:
    if n_layers < 20:
        return 0
    if n_layers <= 40:
        return 1
    if n_layers <= 70:
        return 2
    return 3


def diet_class(vocab_size: int) -> int:
    """6-class diet breadth from vocab size using the shared boundary ladder:
    0 obligate specialist, 1 picky browser, 2 generalist forager,
    3 broad-spectrum omnivore, 4 bulk browser, 5 all-eater/baleen strainer."""
    for i, edge in enumerate(VOCAB_DIET_BOUNDARIES):
        if vocab_size < edge:
            return i
    return 5


# ---------------------------------------------------------------------------
# Layer 2 rules
# ---------------------------------------------------------------------------
def integument_class(meta) -> int:
    """0 exoskeleton, 1 quills, 2 fur, 3 scales, 4 bare, 5 patchy.

    Priority: SSM > MoE > K-quant > legacy > lossless.
    Q2/Q3 condition turns cosmetic skins (fur/scales/bare) patchy;
    structural skins (exoskeleton/quills) keep their class with a
    patchy render modifier.
    """
    if meta.has_ssm:
        base = 0
    elif meta.is_moe:
        base = 1
    elif meta.quant_block_style == "K-quant":
        base = 2
    elif meta.quant_block_style == "legacy":
        base = 3
    elif meta.quant_block_style == "lossless":
        base = 4
    else:
        base = 3
    if condition_class(meta.quant_type) in (4, 5) and base in (2, 3, 4):
        return 5
    return base


def palette_id(architecture: str) -> int:
    a = arch_id(architecture)
    # 0 desert tan, 1 jade+vermilion, 2 mistral blue, 3 pearl-blue,
    # 4 deep violet, 5 ink+gold, 6 rust/amber, 7 slate grey
    return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 7, 9: 7}[a]


def habitat_tint_id(language: str) -> int:
    return lang_group(language)


def fastness_id(quant: str) -> int:
    o = quant_ordinal(quant)
    if o >= 5:
        return 0
    if o >= 2:
        return 1
    return 2


def gradient_pattern_id(rope_freq_base: float) -> int:
    if rope_freq_base >= 100000:
        return 0
    if rope_freq_base >= 10000:
        return 1
    return 2


def marking_complexity_id(vocab_size: int) -> int:
    """6-class marking complexity; shares VOCAB_DIET_BOUNDARIES with
    diet_class so the two ladders can never drift apart:
    0 plain coat, 1 faint markings, 2 modest markings, 3 bold patterning,
    4 intricate plumage, 5 extreme ornate display."""
    return diet_class(vocab_size)


def headgear_id(rope_scaling_type: str, partial_rope: bool) -> int:
    """0 hornless, 1 spikes, 2 antler crown, 3 asymmetric rack, 4 single horn."""
    if partial_rope:
        return 4
    return {"none": 0, "linear": 1, "yarn": 2, "dynamic": 3}.get(
        (rope_scaling_type or "none").lower(), 0)


def ear_size_id(head_dim: int) -> int:
    if head_dim >= 128:
        return 2
    if head_dim >= 64:
        return 1
    return 0


def teacher(meta) -> dict:
    """Full deterministic trait profile for both layers + render modifiers."""
    subterranean = 0 if (meta.has_vision or meta.has_audio) else 1
    return {
        "body_size": body_size_class(meta.n_params),
        "condition": condition_class(meta.quant_type),
        "gut": gut_class(meta.context_length),
        "build": build_class(meta.d_ff, meta.d_model),
        "eye": eye_class(meta.n_heads, meta.n_kv_heads),
        "body_length": body_length_class(meta.n_layers),
        "diet_breadth": diet_class(meta.vocab_size),
        "has_vision": int(meta.has_vision),
        "has_audio": int(meta.has_audio),
        "is_moe": int(meta.is_moe),
        "is_subterranean": subterranean,
        "integument": integument_class(meta),
        "palette": palette_id(meta.architecture),
        "habitat_tint": habitat_tint_id(meta.language),
        "fastness": fastness_id(meta.quant_type),
        "gradient_pattern": gradient_pattern_id(meta.rope_freq_base),
        "marking_complexity": marking_complexity_id(meta.vocab_size),
        "headgear": headgear_id(meta.rope_scaling_type, meta.partial_rope),
        "ear_size": ear_size_id(meta.head_dim),
        "second_jaw": int(meta.mtp_tensors),
        "has_baleen": int(bool(meta.pooling_type)),
        "two_bodied": int(meta.is_encoder_decoder),
        "lipless": int(not meta.add_bos and not meta.add_eos),
        # render-only modifiers (not model heads)
        "patchy_modifier": condition_class(meta.quant_type) in (4, 5),
    }
