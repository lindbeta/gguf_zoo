import pytest

from gguf_zoo import pools

# Module-level list pools (FR-14.1).
LIST_POOLS = [
    "INTEGUMENT", "PALETTE", "HABITAT_TINT", "FASTNESS", "GRADIENT", "MARKINGS",
    "EYES", "GUT", "BUILD", "BODY_LENGTH", "DIET", "HEADGEAR", "EARS",
    "BEHAVIOR", "FICTIONAL_ANATOMY_NAMES", "PERSON_NAMES", "ORGAN_BEHAVIOURS",
    "STRUCTURE_MOVEMENTS", "TERRRESTRIAL_FEATURES", "AERIAL_FEATURES",
    "AQUATIC_FEATURES", "PLANT_FEATURES",
]
# Pools whose entries are complete sentences (exempt from punctuation rule).
SENTENCE_POOLS = {"BEHAVIOR"}


def _check(pool, name):
    assert len(pool) >= 20, f"{name} has {len(pool)} (<20)"
    assert len(set(pool)) == len(pool), f"{name} has duplicates"
    for e in pool:
        assert isinstance(e, str) and e.strip(), f"{name} has empty entry"
        if name not in SENTENCE_POOLS:
            assert e[-1] not in ".!?", f"{name}: {e!r} ends in punctuation"


@pytest.mark.parametrize("name", LIST_POOLS)
def test_FR_14_list_pools(name):
    _check(getattr(pools, name), name)


def test_FR_14_health_pools_per_band():
    for band, pool in pools.HEALTH_PHRASES.items():
        _check(pool, f"HEALTH_PHRASES[{band}]")


def test_FR_14_tier_anchor_pools():
    for tier, pool in pools.TIER_ANCHORS.items():
        _check(pool, f"TIER_ANCHORS[{tier}]")


def test_FR_14_sense_prose_pools():
    for sense, (large, small) in pools.SENSE_PROSE.items():
        _check(large, f"SENSE_PROSE[{sense}].large")
        _check(small, f"SENSE_PROSE[{sense}].small")

