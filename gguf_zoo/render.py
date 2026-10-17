"""Deterministic template renderer: profile + meta -> field description + image prompt.

Sentence templates are fixed (the field-guide voice); every descriptor slot is
filled from the per-class pools in pools.py via salted hash streams, so the
same file always renders identically while different files of the same class
vary their wording.
"""
import hashlib
import json

from .pools import pick, pick_index, pick_slot
from .rules import SIZE_ANCHORS, anchor_taxonomy

# Fixed short forms for the comma-separated image prompt (keyword style,
# not prose - these are intentionally NOT pooled).
PALETTE_SHORT = [
    "desert tan and sand", "jade green with vermilion markings",
    "pale mistral blue", "pearl and blue", "deep violet",
    "ink black with gold", "rust and amber", "slate grey",
]
TINT_SHORT = ["arctic white and pale", "tropical vivid", "temperate green",
              "jade-and-ink", "desert ochre"]
EAR_SHORT = ["sleek-headed", "moderate ears", "big tufted ears"]
MARKING_SHORT = ["plain coat", "faint markings", "modest markings",
                 "bold patterning", "intricate plumage",
                 "extreme ornate bird-of-paradise display"]
TEXTURE_SHORT = ["chitinous segmented plating", "porcupine quills",
                 "layered woven fur", "overlapping scales", "smooth bare skin",
                 "patchy mangy fur"]
HEADGEAR_SHORT = ["hornless head", "simple unbranched spikes",
                  "full antler crown", "asymmetric horn rack",
                  "single unicorn horn"]
EYE_SHORT = [
    "a glittering compound eye",
    "a few large, dark eyes",
    "big eyes fed by a suspiciously cheap optic nerve",
    "one great central eye dominating the face - a true cyclops",
]


def meta_hash(meta) -> str:
    canon = json.dumps({
        "a": meta.architecture, "n": meta.name, "p": meta.n_params,
        "q": meta.quant_type, "c": meta.context_length, "l": meta.n_layers,
        "d": meta.d_model, "v": meta.vocab_size, "h": meta.n_heads,
    }, sort_keys=True)
    return hashlib.md5(canon.encode("utf-8")).hexdigest()


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


PLAQUE_STYLES = ("scientific cutaway plate", "sepia aged engraving",
                 "museum diorama, hyperreal")


def render(meta, profile: dict, name: str = None) -> tuple:
    """Return (field_description, image_prompt). Deterministic per metadata."""
    from . import taxonomy
    if name is None:
        name = taxonomy.binomial(profile, meta)
    h = meta_hash(meta)
    cls = int(profile["body_size"])
    entry = SIZE_ANCHORS[cls]
    anchors = entry["anchors"]
    anchor = anchors[pick_index(h, f"anchor:{cls}", len(anchors))]
    size_phrase = pick_slot(h, "size_phrase").format(a=anchor)

    reg_note = f" ({entry['name']}-class)" if cls >= 10 else ""

    integ = int(profile["integument"])
    patchy = bool(profile.get("patchy_modifier", int(profile["condition"]) in (4, 5)))

    s1 = f"A {size_phrase}{reg_note}, {pick(h, 'condition', int(profile['condition']))}."

    integ_txt = pick(h, "integument", integ)
    if patchy and integ not in (5,):
        integ_txt += ", patchy and mangy in places"
    if meta.mixed_dtypes:
        integ_txt += "; a variegated integument, fur on the torso and scales on the limbs"
    s2 = (f"It {pick_slot(h, 'wear')} {integ_txt}, "
          f"colored {pick(h, 'palette', int(profile['palette']))} "
          f"washed with {pick(h, 'habitat_tint', int(profile['habitat_tint']))}; "
          f"{pick(h, 'fastness', int(profile['fastness']))}, "
          f"{pick(h, 'gradient_pattern', int(profile['gradient_pattern']))}, "
          f"with {pick(h, 'marking_complexity', int(profile['marking_complexity']))}.")

    face_bits = [f"the face is {pick(h, 'headgear', int(profile['headgear']))} "
                 f"({pick(h, 'ear_size', int(profile['ear_size']))})"]
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

    s4 = (f"{pick(h, 'body_length', int(profile['body_length'])).capitalize()}, "
          f"{pick(h, 'build', int(profile['build']))}, "
          f"it carries {pick(h, 'gut', int(profile['gut']))}.")

    diet = pick(h, "diet_breadth", int(profile["diet_breadth"]))
    if int(profile["is_subterranean"]):
        s5 = (f"Blind and deaf, it is a vibration-sensing subterranean mammal, "
              f"foraging as {diet}.")
    else:
        sensory = []
        if int(profile["has_vision"]):
            sensory.append("large compound eyes with a complex retina")
        if int(profile["has_audio"]):
            sensory.append("tympanic membranes and feathery antennae")
        senses = pick(h, "eye", int(profile["eye"]))
        if sensory:
            senses += ", augmented by " + " and ".join(sensory)
        s5 = f"Its senses: {senses}; it forages as {diet}."

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

    body = " ".join(s for s in (s1, s2, s3, s4, s5, s6) if s)
    field = f"*{name}* \u2014 {taxonomy.gloss(profile, meta)}. {body}"

    pose = pick_slot(h, "pose")
    style = [_size_style(cls)]
    if int(profile["is_moe"]):
        style.append("scientific cutaway plate")
    if meta.n_split_files > 1:
        style.append("dotted-line reconstruction")
    if meta.quant_version == 1:
        style.append("sepia aged engraving")
    if meta.quant_block_style == "lossless":
        style.append("museum diorama, hyperreal")
    if any(s in style for s in PLAQUE_STYLES):
        style.append(f'museum specimen plaque reading '
                     f'"{taxonomy.binomial_short(profile, meta)}" in serif type')
    sensory_bits = []
    if int(profile["has_vision"]):
        sensory_bits.append("complex compound eyes")
    if int(profile["has_audio"]):
        sensory_bits.append("tympanic membranes, antennae")
    if int(profile["is_subterranean"]):
        sensory_bits.append("blind vibration-sensing face")
    prompt_size = pick_slot(h, "prompt_size").format(a=anchor)
    prompt_parts = [
        prompt_size,
        anchor_taxonomy(anchor),
        PALETTE_SHORT[int(profile["palette"])],
        f"{TINT_SHORT[int(profile['habitat_tint'])]} habitat tint",
        TEXTURE_SHORT[integ],
        MARKING_SHORT[int(profile["marking_complexity"])],
        HEADGEAR_SHORT[int(profile["headgear"])],
        EAR_SHORT[int(profile["ear_size"])],
        ", ".join(sensory_bits) if sensory_bits else EYE_SHORT[int(profile["eye"])],
        pose,
    ] + style
    image_prompt = ", ".join(p for p in prompt_parts if p)
    return field, image_prompt
