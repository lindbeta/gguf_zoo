"""Deterministic field-guide renderer (FR-9) + image-generator prompt (FR-17)."""
from __future__ import annotations

import re

from . import pools
from .anomalies import anomaly_organs
from .classifier import classify_input_type_consensus
from .parser import ModelMeta, build_meta
from .rules import (HABITAT_NAMES, HEALTH_WORDS, Die, QuantInfo,
                    body_dimensions, build_profile, estimate_params,
                    extract_features, length_phrase, meta_hash, neural_facets,
                    quant_info, quant_modifiers, size_tier)
from .senses import VISIBLE_SENSES
from .taxonomy import binomial, gloss

WIDTH = 72
RULE = "=" * WIDTH

STYLE_TAGS = ("natural history field-guide plate, detailed wildlife "
              "illustration, watercolor and ink, soft diffused daylight, "
              "plain off-white background, full body in profile, "
              "scientifically meticulous")


def _fmt_params(n: int | None) -> str:
    if n is None:
        return "~unknown"
    if n >= 1e12:
        return f"~{n / 1e12:.1f}T"
    return f"~{n / 1e9:.1f}B"


def _classify_inputs(meta: ModelMeta, seed: int, device: str):
    visible, anomaly = [], []
    for name in meta.input_types:
        res = classify_input_type_consensus(name, seed=seed, device=device)
        if res.sense in VISIBLE_SENSES:
            visible.append((name, res.sense))
        else:
            anomaly.append((res.sense, name))
    return visible, anomaly


def _collect(kv: dict, device: str, params_override: int | None,
             in_depth: bool = False, source=None, header=None) -> dict:
    """Run every deterministic draw once (fixed order) and keep the parts."""
    meta = build_meta(kv)
    f = extract_features(meta)
    qi: QuantInfo = quant_info(f)
    n_params = params_override if params_override is not None else estimate_params(f)
    tier_name, tier_prose = size_tier(n_params)
    profile = build_profile(meta)
    seed_str = f"{meta.name}|{meta.arch}|{qi.name}"
    die = Die(seed_str)
    qi.modifiers = quant_modifiers(qi.name, die)
    seed = int(meta_hash(kv)[:8], 16)
    visible, anomaly = _classify_inputs(meta, seed, device)
    profile.senses = visible
    profile.anomaly_senses = anomaly

    anchor = die.choice(pools.TIER_ANCHORS.get(profile.body_size,
                                               pools.TIER_ANCHORS[6]))
    creature = pools.anchor_creature(anchor)
    clade = pools.ANCHOR_CLADE.get(creature, "mythical")
    hab = pools.ANCHOR_HAB[creature]
    profile.habitat = hab
    profile.plant_like = die.random() < 0.34
    feat_name = ["TERRRESTRIAL_FEATURES", "AERIAL_FEATURES",
                 "AQUATIC_FEATURES"][hab]
    feat_pool = pools.pool_for(feat_name, [pools.TERRRESTRIAL_FEATURES,
                                           pools.AERIAL_FEATURES,
                                           pools.AQUATIC_FEATURES][hab], clade)
    feats = [die.choice(feat_pool) for _ in range(2)]
    if profile.plant_like:
        feats.append(die.choice(pools.PLANT_FEATURES))
    profile.features = feats
    hab_phrase = die.choice(pools.HABITAT_PHRASES[
        ["terrestrial", "aerial", "aquatic"][hab]])

    def pick(name):
        return die.choice(pools.pool_for(name, getattr(pools, name), clade))

    s = {
        "meta": meta, "f": f, "qi": qi, "n_params": n_params,
        "tier_name": tier_name, "tier_prose": tier_prose, "profile": profile,
        "dims": body_dimensions(n_params),
        "anchor": anchor,
        "hab_phrase": hab_phrase,
        "integument": pick("INTEGUMENT"),
        "palette": die.choice(pools.PALETTE),
        "tint": die.choice(pools.HABITAT_TINT),
        "fastness": die.choice(pools.FASTNESS),
        "gradient": die.choice(pools.GRADIENT),
        "markings": die.choice(pools.MARKINGS),
        "headgear": pick("HEADGEAR"),
        "ears": pick("EARS"),
        "eyes": pick("EYES"),
        "body_length": die.choice(pools.BODY_LENGTH),
        "build": die.choice(pools.BUILD),
        "gut": pick("GUT"),
        "diet": die.choice(pools.DIET),
    }
    appended = []
    if profile.is_moe:
        appended.append("inside, hundreds of tiny hexagonal chambers fire a few "
                        "at a time — a termite-mound beast")
    if profile.second_jaw:
        appended.append("a second jaw works beneath the first")
    if profile.has_baleen:
        appended.append("baleen plates fringe its face")
    if profile.two_bodied:
        appended.append("its body is two distinct bodies, thorax and abdomen")
    if profile.lipless:
        appended.append("its mouth is lipless")
    s["appended"] = appended

    senses: list[tuple[str, str]] = []
    seen: list[str] = []
    for _, sense in visible:
        if sense not in seen:
            seen.append(sense)
    for sense in seen:
        large, small = pools.SENSE_PROSE.get(sense, pools.SENSE_PROSE["sight"])
        senses.append((sense, die.choice(small if profile.body_size <= 2 else large)))
    s["senses"] = senses
    s["health"] = die.choice(pools.HEALTH_PHRASES[qi.band])
    bodies: list[str] = []
    for body, _source in anomaly_organs(f.unknown, anomaly, device):
        if body not in bodies:
            bodies.append(body)
    s["traits"] = bodies
    s["behavior"] = die.choice(pools.BEHAVIOR)
    s["neural"] = None
    s["sections"] = []
    if in_depth and source is not None:
        from .weights import neural_profile
        np_ = neural_profile(source, kv, header)
        if np_ is not None:
            # seeded on name+arch only (never quant) so quant levels of the
            # same model pick identical phrasings (FR-19.5)
            ndie = Die(f"{meta.name}|{meta.arch}|neural")
            s["neural"] = neural_facets(np_, ndie)
            s["sections"] = _neural_sections(s, anomaly, bodies)
    return s


def _neural_sections(s: dict, anomaly: list, bodies: list):
    """Split in-depth biology write-up: appearance, temperament, behavior,
    habitat, and how the creature's odd organs vary among individuals."""
    f = s["neural"]
    secs = [
        ("APPEARANCE VARIATION", [f["range"], f["variety"], f["molt"]]),
        ("TEMPERAMENT", [f["temper"], f["activity"], f["indiv"]]),
        ("SPECIES BEHAVIOR",
         [f["migration"], f["mating"], f["calls"], f["social"]]),
        ("NATURAL HABITAT", [f["home_range"], f["shelter"]]),
    ]
    organ = None
    if bodies:
        organ = re.split(r"\s*[—(]", bodies[0], maxsplit=1)[0]
    sense = anomaly[0][0].replace("_", " ") if anomaly else None
    if organ is None and sense:
        organ = f"{sense} organ"
    if organ:
        secs.append(("ORGAN VARIATION", [
            f["organ"].format(organ=organ, sense=sense or "distant things",
                              role=f["role"], roles=f["role"] + "s"),
            f["organ_fire"].format(organ=organ)]))
    return secs


def _intro_text(s: dict) -> str:
    profile, meta, dims = s["profile"], s["meta"], s["dims"]
    anchor = s["anchor"]
    if dims:
        size_sentence = (f"It measures about {dims[0]} {length_phrase(anchor)} "
                         f"and weighs about {dims[1]}")
    else:
        size_sentence = "Its size and weight resist measurement"
    parts = [
        f"{anchor[0].upper() + anchor[1:]} — {gloss(profile, meta, s['hab_phrase'])}",
        size_sentence,
        f"It wears {s['integument']}, colored {s['palette']} under {s['tint']}; "
        f"{s['fastness']}, {s['gradient']}, {s['markings']}",
        f"On the head: {s['headgear']} with {s['ears']}; {s['eyes']}",
        f"{s['body_length'].capitalize()}, {s['build']}, "
        f"it carries {s['gut']} and forages as {s['diet']}",
        f"It is {HABITAT_NAMES[profile.habitat]}, bearing {', '.join(profile.features)}",
    ]
    if s["appended"]:
        parts.append("; ".join(s["appended"]).capitalize())
    return ". ".join(parts) + "."


def _prompt_text(s: dict) -> str:
    profile, qi, dims = s["profile"], s["qi"], s["dims"]
    anchor = s["anchor"]
    segs = [f"Wildlife field-guide illustration of {anchor}, "
            f"{gloss(profile, s['meta'], s['hab_phrase'])}"]
    if dims:
        segs.append(f"measuring about {dims[0]} {length_phrase(anchor)} and "
                    f"weighing about {dims[1]}")
    segs.append(f"wearing {s['integument']}, colored {s['palette']} under "
                f"{s['tint']}")
    segs.append(f"{s['fastness']}, {s['gradient']}, {s['markings']}")
    segs.append(f"on the head: {s['headgear']} with {s['ears']}; {s['eyes']}")
    segs.append(f"{s['body_length']}, {s['build']}, carrying {s['gut']}, "
                f"foraging as {s['diet']}")
    segs.append(f"{HABITAT_NAMES[profile.habitat]} animal with "
                f"{', '.join(profile.features)}")
    if s["appended"]:
        segs.append("; ".join(s["appended"]))
    cond = f"condition: {HEALTH_WORDS[qi.band].lower()}, the specimen is {s['health']}"
    if qi.modifiers:
        cond += "; " + "; ".join(qi.modifiers)
    segs.append(cond)
    if s["senses"]:
        segs.append("senses shown by " + "; ".join(
            f"{sense} organs — {prose}" for sense, prose in s["senses"]))
    if s["traits"]:
        segs.append("unexplained anatomy: " + "; ".join(
            t.rstrip(".") for t in s["traits"]))
    if s.get("neural"):
        n = s["neural"]
        segs.append(f"in-depth: {n['range']}; {n['variety']}; {n['temper']}; "
                    f"{n['activity']}; {n['molt']}; {n['social']}; "
                    f"{n['mating']}")
    segs.append(f"behaving thus: {s['behavior']}")
    segs.append(STYLE_TAGS)
    return "  " + ", ".join(segs)


def build_description(kv: dict, header="", device: str = "auto",
                      params_override: int | None = None,
                      in_depth: bool = False, source=None) -> str:
    s = _collect(kv, device, params_override, in_depth=in_depth,
                 source=source, header=header)
    meta, qi, profile = s["meta"], s["qi"], s["profile"]

    lines: list[str] = []
    lines.append(RULE)
    lines.append(f"FIELD GUIDE ENTRY — {binomial(profile, meta)} "
                 f"({pools.COMMON_EPITHETS[qi.band]})")
    lines.append(f"Source model : {meta.name or 'unknown'} ({meta.arch or 'unknown'})")
    lines.append(f"Parameters   : {_fmt_params(s['n_params'])}  ->  size class: "
                 f"{s['tier_name']} ({s['tier_prose']})")
    quant_line = (f"Quantization : {qi.name} ({qi.label})  ->  "
                  f"condition: {HEALTH_WORDS[qi.band]}")
    if not qi.recognized:
        quant_line += "  [UNKNOWN QUANT — condition estimated]"
    lines.append(quant_line)
    lines.append(RULE)
    lines.append("")
    lines.append(_intro_text(s))

    if s["senses"]:
        lines.append("")
        lines.append("SENSES:")
        for sense, prose in s["senses"]:
            lines.append(f"  • {sense.capitalize():<14} {prose}")

    lines.append("")
    lines.append(f"CONDITION — {HEALTH_WORDS[qi.band].upper()}: the specimen is "
                 f"{s['health']}.")
    for mod in qi.modifiers:
        lines.append(f"  • {mod}")

    if s["traits"]:
        lines.append("")
        lines.append("UNIQUE TRAITS:")
        for body in s["traits"]:
            lines.append(f"  • {body}.")

    for title, bullets in s.get("sections", []):
        lines.append("")
        lines.append(title + ":")
        for text in bullets:
            lines.append(f"  • {text[0].upper() + text[1:]}.")

    lines.append("")
    lines.append(s["behavior"])

    lines.append("")
    lines.append("IMAGE PROMPT:")
    lines.append(_prompt_text(s))

    return "\n".join(lines) + "\n"


def render_prompt(kv: dict, device: str = "auto", in_depth: bool = False,
                  source=None, header=None) -> str:
    """FR-17: the detailed image-generator prompt for a KV mapping."""
    return _prompt_text(_collect(kv, device, None, in_depth=in_depth,
                                 source=source, header=header)).strip()
