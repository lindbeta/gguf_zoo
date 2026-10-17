from collections import Counter

from gguf_zoo import pools
from gguf_zoo.render import build_description
from gguf_zoo.rules import HABITAT_NAMES, habitat_class


def test_FR_15_1_equal_thirds_assignment():
    counts = Counter(habitat_class(i) for i in range(3000))
    assert counts[0] == counts[1] == counts[2] == 1000


def test_FR_15_2_tier_anchors_equal_habitats():
    for tier, (terr, aer, aqu) in pools.TIER_HAB_COUNTS.items():
        assert terr == aer == aqu, f"tier {tier} unbalanced: {(terr, aer, aqu)}"


def test_FR_15_3_habitat_feature_pools_ge_20():
    for name in ["TERRRESTRIAL_FEATURES", "AERIAL_FEATURES", "AQUATIC_FEATURES"]:
        pool = getattr(pools, name)
        assert len(pool) >= 20 and len(set(pool)) == len(pool), name


def test_FR_15_4_plant_feature_pool_ge_20():
    assert len(pools.PLANT_FEATURES) >= 20
    assert len(set(pools.PLANT_FEATURES)) == len(pools.PLANT_FEATURES)


def test_FR_15_6_entry_names_a_habitat(demo_kv):
    out = build_description(demo_kv, device="cpu")
    assert any(h in out for h in HABITAT_NAMES)
