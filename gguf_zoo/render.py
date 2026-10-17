"""Deterministic field-guide renderer (FR-9)."""
from __future__ import annotations

from . import pools
from .anomalies import anomaly_organs
from .classifier import classify_input_type_consensus
from .parser import ModelMeta, build_meta
from .rules import (HEALTH_WORDS, HABITAT_NAMES, Die, QuantInfo, body_dimensions,
                    build_profile, estimate_params, extract_features,
                    habitat_class, length_phrase, meta_hash, quant_info,
                    size_tier)
from .senses import VISIBLE_SENSES
from .taxonomy import binomial, gloss

WIDTH = 72
RULE = "=" * WIDTH


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


def _intro(meta, profile, qi, dims, die) -> str:
    anchor = die.choice(pools.TIER_ANCHORS.get(profile.body_size, pools.TIER_ANCHORS[6]))
    if dims:
        size_sentence = (f"It measures about {dims[0]} {length_phrase(anchor)} "
                         f"and weighs about {dims[1]}")
    else:
        size_sentence = "Its size and weight resist measurement"
    parts = [
        f"{anchor[0].upper() + anchor[1:]} — {gloss(profile, meta)}",
        size_sentence,
        f"It wears {die.choice(pools.INTEGUMENT)}, colored "
        f"{die.choice(pools.PALETTE)} under {die.choice(pools.HABITAT_TINT)}; "
        f"{die.choice(pools.FASTNESS)}, {die.choice(pools.GRADIENT)}, "
        f"{die.choice(pools.MARKINGS)}",
        f"On the head: {die.choice(pools.HEADGEAR)} with {die.choice(pools.EARS)}; "
        f"{die.choice(pools.EYES)}",
        f"{die.choice(pools.BODY_LENGTH).capitalize()}, {die.choice(pools.BUILD)}, "
        f"it carries {die.choice(pools.GUT)} and forages as {die.choice(pools.DIET)}",
        f"It is {HABITAT_NAMES[profile.habitat]}, bearing {', '.join(profile.features)}",
    ]
    appended = []
    if profile.is_moe:
        appended.append("Inside, hundreds of tiny hexagonal chambers fire a few at a "
                        "time — a termite-mound beast")
    if profile.second_jaw:
        appended.append("a second jaw works beneath the first")
    if profile.has_baleen:
        appended.append("baleen plates fringe its face")
    if profile.two_bodied:
        appended.append("its body is two distinct bodies, thorax and abdomen")
    if profile.lipless:
        appended.append("its mouth is lipless")
    if appended:
        parts.append("; ".join(appended).capitalize())
    return ". ".join(parts) + "."


def build_description(kv: dict, header: str = "", device: str = "auto",
                      params_override: int | None = None) -> str:
    meta = build_meta(kv)
    f = extract_features(meta)
    qi: QuantInfo = quant_info(f)
    n_params = params_override if params_override is not None else estimate_params(f)
    tier_name, tier_prose = size_tier(n_params)
    profile = build_profile(meta)
    seed_str = f"{meta.name}|{meta.arch}|{qi.name}"
    die = Die(seed_str)
    seed = int(meta_hash(kv)[:8], 16)
    visible, anomaly = _classify_inputs(meta, seed, device)
    profile.senses = visible
    profile.anomaly_senses = anomaly

    hab = habitat_class(die.index([0, 1, 2]))
    profile.habitat = hab
    profile.plant_like = die.random() < 0.34
    feat_pool = [pools.TERRRESTRIAL_FEATURES, pools.AERIAL_FEATURES,
                 pools.AQUATIC_FEATURES][hab]
    feats = [die.choice(feat_pool) for _ in range(2)]
    if profile.plant_like:
        feats.append(die.choice(pools.PLANT_FEATURES))
    profile.features = feats

    lines: list[str] = []
    lines.append(RULE)
    lines.append(f"FIELD GUIDE ENTRY — {binomial(profile, meta)} "
                 f"({pools.COMMON_EPITHETS[qi.band]})")
    lines.append(f"Source model : {meta.name or 'unknown'} ({meta.arch or 'unknown'})")
    lines.append(f"Parameters   : {_fmt_params(n_params)}  ->  size class: "
                 f"{tier_name} ({tier_prose})")
    quant_line = (f"Quantization : {qi.name} ({qi.label})  ->  "
                  f"condition: {HEALTH_WORDS[qi.band]}")
    if not qi.recognized:
        quant_line += "  [UNKNOWN QUANT — condition estimated]"
    lines.append(quant_line)
    lines.append(RULE)
    lines.append("")
    lines.append(_intro(meta, profile, qi, body_dimensions(n_params), die))

    if visible:
        lines.append("")
        lines.append("SENSES:")
        seen: list[str] = []
        for _, sense in visible:
            if sense not in seen:
                seen.append(sense)
        for sense in seen:
            large, small = pools.SENSE_PROSE.get(sense, pools.SENSE_PROSE["sight"])
            prose = die.choice(small if profile.body_size <= 2 else large)
            lines.append(f"  • {sense.capitalize():<14} {prose}")

    lines.append("")
    health = pools.HEALTH_PHRASES[qi.band]
    lines.append(f"CONDITION — {HEALTH_WORDS[qi.band].upper()}: the specimen is "
                 f"{die.choice(health)}.")
    for mod in qi.modifiers:
        lines.append(f"  • {mod}")

    anomalies = anomaly_organs(f.unknown, anomaly, device)
    if anomalies:
        lines.append("")
        lines.append("UNIQUE TRAITS:")
        seen_bodies: set[str] = set()
        for body, _source in anomalies:
            if body in seen_bodies:
                continue
            seen_bodies.add(body)
            lines.append(f"  • {body}.")

    lines.append("")
    lines.append(die.choice(pools.BEHAVIOR))

    return "\n".join(lines) + "\n"


def render_prompt(kv: dict, device: str = "auto") -> str:
    meta = build_meta(kv)
    profile = build_profile(meta)
    die = Die(f"{meta.name}|prompt")
    return (f"{pools.TIER_ANCHORS.get(profile.body_size, pools.TIER_ANCHORS[6])[0]} "
            f"creature, {die.choice(pools.PALETTE)}, {die.choice(pools.INTEGUMENT)}, "
            f"{die.choice(pools.MARKINGS)}, natural-history illustration")
