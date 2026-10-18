import pytest

from gguf_zoo import pools

# Pools whose entries are complete sentences (exempt from punctuation rule).
SENTENCE_POOLS = {"BEHAVIOR"}


def _iter_pools():
    """Auto-discover every selection pool in pools.py (FR-14.5): module-level
    lists of strings, dict values that are lists of strings, and tuple-of-lists
    values (SENSE_PROSE)."""
    for name, val in vars(pools).items():
        if name.startswith("_"):
            continue  # private word banks are checked explicitly below
        if isinstance(val, list) and val and all(isinstance(e, str) for e in val):
            yield name, val
        elif isinstance(val, dict):
            for key, sub in val.items():
                if isinstance(sub, list) and sub and all(isinstance(e, str)
                                                         for e in sub):
                    yield f"{name}[{key}]", sub
                elif isinstance(sub, tuple):
                    for i, lst in enumerate(sub):
                        if isinstance(lst, list) and lst and all(
                                isinstance(e, str) for e in lst):
                            yield f"{name}[{key}][{i}]", lst


ALL_POOLS = dict(_iter_pools())


def _base(name):
    return name.split("[")[0]


def _check(pool, name):
    assert len(pool) >= 20, f"{name} has {len(pool)} (<20)"
    assert len(set(pool)) == len(pool), f"{name} has duplicates"
    for e in pool:
        assert isinstance(e, str) and e.strip(), f"{name} has empty entry"
        if _base(name) not in SENTENCE_POOLS:
            assert e[-1] not in ".!?", f"{name}: {e!r} ends in punctuation"


def test_FR_14_5_auto_discovery_finds_known_pools():
    for known in ("INTEGUMENT", "PALETTE", "BEHAVIOR", "NEURAL_RANGE",
                  "NEURAL_ORGAN_ROLES", "QUANT_MODIFIER_POOLS",
                  "HEALTH_PHRASES", "TIER_ANCHORS", "HABITAT_PHRASES"):
        assert any(k == known or k.startswith(known + "[") for k in ALL_POOLS), known


@pytest.mark.parametrize("name", sorted(ALL_POOLS))
def test_FR_14_list_pools(name):
    _check(ALL_POOLS[name], name)


def test_FR_14_word_banks():
    _check(pools._ADJ, "_ADJ")
    _check(pools._CLAUSE, "_CLAUSE")
    _check(pools._HEALTH_NOUN, "_HEALTH_NOUN")
    for band, adj in pools._HEALTH_ADJ.items():
        _check(adj, f"_HEALTH_ADJ[{band}]")
    _check(pools._TIER_MOD, "_TIER_MOD")
    for sense, nouns in pools._SENSE_NOUNS.items():
        _check(nouns, f"_SENSE_NOUNS[{sense}]")
    for hab, phrases in pools.HABITAT_PHRASES.items():
        _check(phrases, f"HABITAT_PHRASES[{hab}]")
