"""The 'biology': size ladder, anchor lists, and the deterministic teacher rules.

The teacher maps ModelMeta -> trait labels for BOTH layers. The student
network learns to reproduce these labels from the feature vector.
"""
import math

# ---------------------------------------------------------------------------
# Size ladder: class index -> size-class name + anchor creatures.
# anchors: list of display names
# ---------------------------------------------------------------------------
SIZE_ANCHORS = {
    0: {"name": "virus-bacteria", "anchors": [
        "influenza virion", "E. coli", "amoeba", "paramecium", "yeast cell",
        "rotifer", "tardigrade", "nematode", "dust mite", "springtail",
        "copepod", "brine shrimp"]},
    1: {"name": "microbe-midge", "anchors": [
        "fairyfly", "thrips", "aphid", "flea", "feather louse", "whale louse",
        "seed shrimp", "fungus gnat", "mosquito", "blackfly", "small wasp",
        "fruit fly"]},
    2: {"name": "worm-gnat", "anchors": [
        "earthworm", "mealworm larva", "caterpillar", "garden slug", "snail",
        "woodlouse", "bed bug", "silverfish", "earwig", "small spider",
        "harvestman", "small cricket"]},
    3: {"name": "housefly-honeybee", "anchors": [
        "housefly", "honeybee", "bumblebee", "paper wasp", "ladybird",
        "ground beetle", "cockroach", "cricket", "grasshopper", "dragonfly",
        "monarch butterfly", "zebrafish", "spring peeper frog",
        "anole lizard"]},
    4: {"name": "scorpion-songbird", "anchors": [
        "scorpion", "tarantula", "hermit crab", "shore crab", "goldfish",
        "killifish", "tree frog", "gecko", "wren", "finch", "sparrow",
        "blackbird", "vole", "shrew", "pygmy rabbit", "hatchling turtle"]},
    5: {"name": "rat-hawk", "anchors": [
        "brown rat", "black rat", "guinea pig", "squirrel", "chipmunk",
        "ferret", "weasel", "hedgehog", "opossum", "pigeon", "crow", "magpie",
        "kestrel", "merlin", "marmoset"]},
    6: {"name": "cat-pig", "anchors": [
        "domestic cat", "beagle", "border collie", "fox", "raccoon", "badger",
        "beaver", "capybara", "wallaby", "bobcat", "ocelot", "caracal",
        "macaque", "baboon", "pot-bellied pig", "goat"]},
    7: {"name": "ape-horse", "anchors": [
        "human", "chimpanzee", "bonobo", "gorilla", "orangutan", "grey wolf",
        "cougar", "red deer", "elk", "donkey", "zebra", "horse", "steer",
        "black bear", "Labrador retriever"]},
    8: {"name": "bison-elephant", "anchors": [
        "American bison", "white rhino", "black rhino", "Indian rhino",
        "hippo", "giraffe", "polar bear", "walrus", "manatee", "beluga",
        "camel", "moose", "musk ox", "Asian elephant"]},
    9: {"name": "tyrannosaurus-sauropod", "anchors": [
        "Tyrannosaurus", "Spinosaurus", "Allosaurus", "Triceratops",
        "Stegosaurus", "Ankylosaurus", "Diplodocus", "Camarasaurus",
        "woolly mammoth", "woolly rhino", "Paraceratherium",
        "giant ground sloth"]},
    10: {"name": "titan", "anchors": [
        "Argentinosaurus", "Patagotitan", "Dreadnoughtus", "Supersaurus",
        "Alamosaurus", "blue whale", "fin whale", "Perucetus colossus",
        "megalodon", "Liopleurodon", "King Kong", "Titanoboa maximus",
        "sea Leviathan", "White Worm of Winterfell"]},
    11: {"name": "kaiju", "anchors": [
        "Godzilla", "King Ghidorah", "Mothra", "Rodan", "Mechagodzilla",
        "Gigan", "Biollante", "SpaceGodzilla", "Destoroyah", "Shin Godzilla",
        "Knifehead", "Raiju", "Slattern", "Otachi", "The Elder", "Sachiel",
        "Zeruel", "Armisael", "Baltan", "Zetton", "Deep Sea King",
        "Colossal Titan"]},
}

BODY_SIZE_EDGES = [1e8, 5e8, 1e9, 5e9, 2e10, 1e11, 5e11, 1e12, 5e12, 2e13, 1e14]

# Linnaean binomials for every anchor creature, keyed by anchor display name.
# Real (and extinct) animals get their real scientific names; legendary
# anchors get mock-Latin coinages. Used by the renderer in the image prompt.
ANCHOR_TAXONOMY = {
    # 0 virus-bacteria
    "influenza virion": "Alphainfluenzavirus influenzae",
    "E. coli": "Escherichia coli",
    "amoeba": "Amoeba proteus",
    "paramecium": "Paramecium caudatum",
    "yeast cell": "Saccharomyces cerevisiae",
    "rotifer": "Brachionus plicatilis",
    "tardigrade": "Ramazzottius varieornatus",
    "nematode": "Caenorhabditis elegans",
    "dust mite": "Dermatophagoides pteronyssinus",
    "springtail": "Folsomia candida",
    "copepod": "Calanus finmarchicus",
    "brine shrimp": "Artemia salina",
    # 1 microbe-midge
    "fairyfly": "Megaphragma caribea",
    "thrips": "Frankliniella occidentalis",
    "aphid": "Acyrthosiphon pisum",
    "flea": "Xenopsylla cheopis",
    "feather louse": "Menacanthus stramineus",
    "whale louse": "Cyamus scammoni",
    "seed shrimp": "Cypris subglobosa",
    "fungus gnat": "Lycoriella ingenua",
    "mosquito": "Aedes aegypti",
    "blackfly": "Simulium damnosum",
    "small wasp": "Trichogramma brassicae",
    "fruit fly": "Drosophila melanogaster",
    # 2 worm-gnat
    "earthworm": "Lumbricus terrestris",
    "mealworm larva": "Tenebrio molitor",
    "caterpillar": "Manduca sexta",
    "garden slug": "Arion hortensis",
    "snail": "Cornu aspersum",
    "woodlouse": "Oniscus asellus",
    "bed bug": "Cimex lectularius",
    "silverfish": "Lepisma saccharina",
    "earwig": "Forficula auricularia",
    "small spider": "Parasteatoda tepidariorum",
    "harvestman": "Leiobunum vittatum",
    "small cricket": "Acheta domesticus",
    # 3 housefly-honeybee
    "housefly": "Musca domestica",
    "honeybee": "Apis mellifera",
    "bumblebee": "Bombus terrestris",
    "paper wasp": "Polistes dominula",
    "ladybird": "Coccinella septempunctata",
    "ground beetle": "Carabus auratus",
    "cockroach": "Periplaneta americana",
    "cricket": "Gryllus campestris",
    "grasshopper": "Schistocerca gregaria",
    "dragonfly": "Anax junius",
    "monarch butterfly": "Danaus plexippus",
    "zebrafish": "Danio rerio",
    "spring peeper frog": "Pseudacris crucifer",
    "anole lizard": "Anolis carolinensis",
    # 4 scorpion-songbird
    "scorpion": "Pandinus imperator",
    "tarantula": "Theraphosa blondi",
    "hermit crab": "Pagurus bernhardus",
    "shore crab": "Carcinus maenas",
    "goldfish": "Carassius auratus",
    "killifish": "Fundulus heteroclitus",
    "tree frog": "Hyla arborea",
    "gecko": "Tarentola mauritanica",
    "wren": "Troglodytes troglodytes",
    "finch": "Fringilla coelebs",
    "sparrow": "Passer domesticus",
    "blackbird": "Turdus merula",
    "vole": "Microtus agrestis",
    "shrew": "Sorex araneus",
    "pygmy rabbit": "Brachylagus idahoensis",
    "hatchling turtle": "Chrysemys picta",
    # 5 rat-hawk
    "brown rat": "Rattus norvegicus",
    "black rat": "Rattus rattus",
    "guinea pig": "Cavia porcellus",
    "squirrel": "Sciurus carolinensis",
    "chipmunk": "Tamias striatus",
    "ferret": "Mustela furo",
    "weasel": "Mustela nivalis",
    "hedgehog": "Erinaceus europaeus",
    "opossum": "Didelphis virginiana",
    "pigeon": "Columba livia",
    "crow": "Corvus corone",
    "magpie": "Pica pica",
    "kestrel": "Falco tinnunculus",
    "merlin": "Falco columbarius",
    "marmoset": "Callithrix jacchus",
    # 6 cat-pig
    "domestic cat": "Felis catus",
    "beagle": "Canis lupus familiaris",
    "border collie": "Canis lupus familiaris",
    "fox": "Vulpes vulpes",
    "raccoon": "Procyon lotor",
    "badger": "Meles meles",
    "beaver": "Castor canadensis",
    "capybara": "Hydrochoerus hydrochaeris",
    "wallaby": "Notamacropus rufogriseus",
    "bobcat": "Lynx rufus",
    "ocelot": "Leopardus pardalis",
    "caracal": "Caracal caracal",
    "macaque": "Macaca mulatta",
    "baboon": "Papio anubis",
    "pot-bellied pig": "Sus scrofa domesticus",
    "goat": "Capra hircus",
    # 7 ape-horse
    "human": "Homo sapiens",
    "chimpanzee": "Pan troglodytes",
    "bonobo": "Pan paniscus",
    "gorilla": "Gorilla gorilla",
    "orangutan": "Pongo pygmaeus",
    "grey wolf": "Canis lupus",
    "cougar": "Puma concolor",
    "red deer": "Cervus elaphus",
    "elk": "Cervus canadensis",
    "donkey": "Equus asinus",
    "zebra": "Equus quagga",
    "horse": "Equus caballus",
    "steer": "Bos taurus",
    "black bear": "Ursus americanus",
    "Labrador retriever": "Canis lupus familiaris",
    # 8 bison-elephant
    "American bison": "Bison bison",
    "white rhino": "Ceratotherium simum",
    "black rhino": "Diceros bicornis",
    "Indian rhino": "Rhinoceros unicornis",
    "hippo": "Hippopotamus amphibius",
    "giraffe": "Giraffa camelopardalis",
    "polar bear": "Ursus maritimus",
    "walrus": "Odobenus rosmarus",
    "manatee": "Trichechus manatus",
    "beluga": "Delphinapterus leucas",
    "camel": "Camelus dromedarius",
    "moose": "Alces alces",
    "musk ox": "Ovibos moschatus",
    "Asian elephant": "Elephas maximus",
    # 9 tyrannosaurus-sauropod (extinct, real scientific names)
    "Tyrannosaurus": "Tyrannosaurus rex",
    "Spinosaurus": "Spinosaurus aegyptiacus",
    "Allosaurus": "Allosaurus fragilis",
    "Triceratops": "Triceratops horridus",
    "Stegosaurus": "Stegosaurus stenops",
    "Ankylosaurus": "Ankylosaurus magniventris",
    "Diplodocus": "Diplodocus carnegii",
    "Camarasaurus": "Camarasaurus lentus",
    "woolly mammoth": "Mammuthus primigenius",
    "woolly rhino": "Coelodonta antiquitatis",
    "Paraceratherium": "Paraceratherium bugtiense",
    "giant ground sloth": "Megatherium americanum",
    # 10 titan - real
    "Argentinosaurus": "Argentinosaurus huinculensis",
    "Patagotitan": "Patagotitan mayorum",
    "Dreadnoughtus": "Dreadnoughtus schrani",
    "Supersaurus": "Supersaurus vivianae",
    "Alamosaurus": "Alamosaurus sanjuanensis",
    "blue whale": "Balaenoptera musculus",
    "fin whale": "Balaenoptera physalus",
    "Perucetus colossus": "Perucetus colossus",
    "megalodon": "Otodus megalodon",
    "Liopleurodon": "Liopleurodon ferox",
    # 10 titan - legendary anchors (mock-Latin)
    "King Kong": "Megaprimatus kong",
    "Titanoboa maximus": "Titanoboa maximus",
    "sea Leviathan": "Leviathan maritimus",
    "White Worm of Winterfell": "Vermis albus",
    # 11 kaiju (mock-Latin)
    "Godzilla": "Godzillasaurus gojirae",
    "King Ghidorah": "Ghidorus tricephalus",
    "Mothra": "Mothra pulchella",
    "Rodan": "Rodan volans",
    "Mechagodzilla": "Mechagodzilla metallicus",
    "Gigan": "Gigan ensifer",
    "Biollante": "Biollante plantaris",
    "SpaceGodzilla": "SpaceGodzilla sidereus",
    "Destoroyah": "Destoroyah aquatica",
    "Shin Godzilla": "Godzillasaurus shinensis",
    "Knifehead": "Cultellum maris",
    "Raiju": "Raiurus electricus",
    "Slattern": "Slattern venenum",
    "Otachi": "Otachi volans",
    "The Elder": "Elder abyssalis",
    "Sachiel": "Sachiel angelicus",
    "Zeruel": "Zeruel angelicus",
    "Armisael": "Armisael angelicus",
    "Baltan": "Baltan stellis",
    "Zetton": "Zetton transcendens",
    "Deep Sea King": "Rex abyssorum",
    "Colossal Titan": "Titan colossus",
}


def anchor_taxonomy(name: str) -> str:
    """Linnaean binomial for an anchor creature."""
    return ANCHOR_TAXONOMY[name]

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
