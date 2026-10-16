"""Deterministic template renderer: profile + meta -> field description + image prompt."""
import hashlib
import json
import random

from .rules import SIZE_ANCHORS

CONDITION = [
    "pristine and freeze-dried", "immaculate", "well-preserved, slightly faded",
    "worn-plumaged with a matte texture", "patchy-furred with clouded eyes",
    "a shrunken-head museum relic",
]
GUT = [
    "a hummingbird crop that demands frequent small feedings",
    "a normal mammalian gut",
    "a ruminant fermentation chamber",
    "a whale gut for long, slow fermentations",
]
BUILD = ["wiry and streamlined", "balanced in build", "thick-bodied, a true bulk digestor"]
EYE = [
    "a glittering compound eye",
    "a few large, dark eyes",
    "big eyes fed by a suspiciously cheap optic nerve",
    "one great central eye dominating the face - a true cyclops",
]
BODY_LENGTH = [
    "Tadpole-short", "Compact-bodied", "Deep-bodied",
    "Centipede-long along a deliberate neural path",
]
DIET = ["a picky feeder", "a generalist", "an omnivore that eats anything"]
INTEGUMENT = [
    "chitinous exoskeleton plating - segmented, state-carrying plates",
    "spiny porcupine-quill coverage, only a few spines raised at once",
    "a layered, woven fur coat",
    "uniform, independent, overlapping scales",
    "smooth bare skin",
    "a patchy, mangy coat",
]
PALETTE = [
    "desert tan and sand", "jade green with vermilion markings",
    "pale mistral blue", "pearl and blue", "deep violet",
    "ink black with gold", "rust and amber", "slate grey",
]
TINT = ["arctic white and pale", "tropical vivid", "temperate green",
        "jade-and-ink", "desert ochre"]
FASTNESS = ["colors true to life", "sun-faded", "bleached museum-drawer pale"]
GRADIENT = ["a slow gradient from dark head to pale tail", "even coloration",
            "banded and striped"]
MARKING = ["a plain coat", "modest markings",
           "intricate, every-feather-different patterning"]
HEADGEAR = ["hornless", "a pair of simple unbranched spikes",
            "a full antler crown with many tines", "an asymmetric rack",
            "a single horn - a one-sided rack"]
EAR = ["sleek-headed", "moderate ears", "big tufted ears"]

SIZE_PHRASES = ["{a}-sized creature", "creature the size of a {a}",
                "creature sized like a {a}", "{a}-scale frame"]
PROMPT_SIZE = ["{a}-sized creature", "life-sized {a}", "{a}-scale creature"]
WEARS = ["wears", "sports", "is clad in", "carries"]
POSES = [
    "standing alert in three-quarter view", "resting, coiled", "mid-stride",
    "dorsal and ventral views", "head-and-shoulders portrait",
]
SCALE_STYLE = {
    (0, 3): "macro photography",
    (4, 7): "natural-history illustration",
    (8, 9): "epic scale, dramatic low-angle view",
}


def meta_hash(meta) -> int:
    canon = json.dumps({
        "a": meta.architecture, "n": meta.name, "p": meta.n_params,
        "q": meta.quant_type, "c": meta.context_length, "l": meta.n_layers,
        "d": meta.d_model, "v": meta.vocab_size, "h": meta.n_heads,
    }, sort_keys=True)
    return int(hashlib.md5(canon.encode("utf-8")).hexdigest(), 16)


def _size_style(cls: int) -> str:
    if cls <= 3:
        return "macro photography"
    if cls <= 7:
        return "natural-history illustration"
    if cls <= 9:
        return "epic scale, dramatic low-angle view"
    if cls == 10:
        return ("colossal scale, real-animal naturalism blended with "
                "creature-feature drama")
    return ("cinematic kaiju film still, city skyline for scale, dramatic "
            "low-angle, Toho/Pacific Rim creature-design style")


def _texture_short(integ: int) -> str:
    return ["chitinous segmented plating", "porcupine quills", "layered woven fur",
            "overlapping scales", "smooth bare skin", "patchy mangy fur"][integ]


def _headgear_short(h: int) -> str:
    return ["hornless head", "simple unbranched spikes", "full antler crown",
            "asymmetric horn rack", "single unicorn horn"][h]


def render(meta, profile: dict) -> tuple:
    """Return (field_description, image_prompt). Deterministic per metadata."""
    rng = random.Random(meta_hash(meta))
    cls = int(profile["body_size"])
    entry = SIZE_ANCHORS[cls]
    anchor, fictional = entry["anchors"][rng.randrange(len(entry["anchors"]))]
    size_phrase = rng.choice(SIZE_PHRASES).format(a=anchor)

    reg = entry["register"]
    if reg == "fiction":
        reg_note = f" ({entry['name']}-class, beyond any living creature, fictional register)"
    elif reg == "mixed":
        reg_note = (f" ({entry['name']}-class, mixed real-and-legendary register"
                    + (", a creature of legend" if fictional else "") + ")")
    else:
        reg_note = ""

    integ = int(profile["integument"])
    patchy = bool(profile.get("patchy_modifier", int(profile["condition"]) in (4, 5)))
    cond = CONDITION[int(profile["condition"])]

    s1 = f"A {size_phrase}{reg_note}, {cond}."

    integ_txt = INTEGUMENT[integ]
    if patchy and integ not in (5,):
        integ_txt += ", patchy and mangy in places"
    if meta.mixed_dtypes:
        integ_txt += "; a variegated integument, fur on the torso and scales on the limbs"
    s2 = (f"It {rng.choice(WEARS)} {integ_txt}, colored {PALETTE[int(profile['palette'])]}"
          f" washed with a {TINT[int(profile['habitat_tint'])]} habitat tint; "
          f"{FASTNESS[int(profile['fastness'])]}, {GRADIENT[int(profile['gradient_pattern'])]},"
          f" with {MARKING[int(profile['marking_complexity'])]}.")

    face_bits = [f"the face is {HEADGEAR[int(profile['headgear'])]} ({EAR[int(profile['ear_size'])]})"]
    if int(profile["second_jaw"]):
        face_bits.append("a spare second mandible folds beneath the first")
    if int(profile["has_baleen"]):
        face_bits.append("baleen plates fringe the mouth, a filter-feeder's face")
    n_br = min(int(meta.n_special_tokens), 8)
    if n_br:
        face_bits.append(f"{n_br} sensory whisker-bristles twitch on the muzzle")
    face_bits.append("defined lips and a beak tip" if not int(profile["lipless"])
                     else "a lipless mouth")
    s3 = "On the head, " + "; ".join(face_bits) + "."

    s4 = (f"{BODY_LENGTH[int(profile['body_length'])]} and {BUILD[int(profile['build'])]}, "
          f"it carries {GUT[int(profile['gut'])]}.")

    if int(profile["is_subterranean"]):
        s5 = (f"Blind and deaf, it is a vibration-sensing subterranean mammal, "
              f"foraging as {DIET[int(profile['diet'])]}.")
    else:
        sensory = []
        if int(profile["has_vision"]):
            sensory.append("large compound eyes with a complex retina")
        if int(profile["has_audio"]):
            sensory.append("tympanic membranes and feathery antennae")
        senses = EYE[int(profile["eye"])]
        if sensory:
            senses += ", augmented by " + " and ".join(sensory)
        s5 = f"Its senses: {senses}; it forages as {DIET[int(profile['diet'])]}."

    eco = []
    if int(profile["is_moe"]):
        eco.append("inside, hundreds of tiny hexagonal digestive chambers fire only a few at a time, a termite-mound beast")
    if meta.sliding_window:
        eco.append("mixed fast and slow digestion chambers line the gut")
    if int(profile["two_bodied"]):
        eco.append("thorax and abdomen are visibly distinct, two fused body plans")
    ratio = (meta.n_tensors / meta.n_params) if meta.n_params else 0
    if ratio > 1e-7:
        eco.append("lean and ropy in the frame")
    elif ratio and ratio < 1e-9:
        eco.append("smooth and fat in the frame")
    if meta.n_split_files > 1:
        eco.append(f"shipped in {meta.n_split_files} crates, drawn with a dotted-line reconstructed outline")
    if meta.rms_epsilon < 1e-5:
        eco.append("vestigial dewclaws on every foot")
    if meta.byte_fallback:
        eco.append("gizzard stones visible in the throat")
    s6 = "Ecology: " + "; ".join(eco) + "." if eco else ""

    field = " ".join(s for s in (s1, s2, s3, s4, s5, s6) if s)

    pose = rng.choice(POSES)
    style = [_size_style(cls)]
    if int(profile["is_moe"]):
        style.append("scientific cutaway plate")
    if meta.n_split_files > 1:
        style.append("dotted-line reconstruction")
    if meta.quant_version == 1:
        style.append("sepia aged engraving")
    if meta.quant_block_style == "lossless":
        style.append("museum diorama, hyperreal")
    sensory_bits = []
    if int(profile["has_vision"]):
        sensory_bits.append("complex compound eyes")
    if int(profile["has_audio"]):
        sensory_bits.append("tympanic membranes, antennae")
    if int(profile["is_subterranean"]):
        sensory_bits.append("blind vibration-sensing face")
    prompt_size = rng.choice(PROMPT_SIZE).format(a=anchor)
    prompt_parts = [
        prompt_size,
        PALETTE[int(profile["palette"])],
        f"{TINT[int(profile['habitat_tint'])]} habitat tint",
        _texture_short(integ),
        _headgear_short(int(profile["headgear"])),
        EAR[int(profile["ear_size"])],
        ", ".join(sensory_bits) if sensory_bits else EYE[int(profile["eye"])],
        pose,
    ] + style
    image_prompt = ", ".join(p for p in prompt_parts if p)
    return field, image_prompt
