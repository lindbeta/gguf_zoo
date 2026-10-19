from gguf_zoo import pools
from gguf_zoo.render import body_of, build_description

NON_MAMMAL = [c for c in pools.CLADES if c not in ("mammal", "mythical")]
NON_BIRD = [c for c in pools.CLADES if c not in ("bird", "mythical")]
POOLS = ["INTEGUMENT", "EARS", "HEADGEAR", "EYES", "GUT",
         "TERRRESTRIAL_FEATURES", "AERIAL_FEATURES", "AQUATIC_FEATURES"]


def test_FR_18_1_every_anchor_tagged():
    for tier, (terr, aer, aqu) in pools._TIER_HAB.items():
        for c in terr + aer + aqu:
            assert c in pools.ANCHOR_CLADE, (tier, c)
            assert c in pools.ANCHOR_HAB, (tier, c)
    assert set(pools.ANCHOR_CLADE.values()) <= set(pools.CLADES)


def test_FR_18_2_mammal_traits_absent_for_other_clades():
    for clade in NON_MAMMAL:
        assert "a thick layered fur pelt" not in pools.pool_for(
            "INTEGUMENT", pools.INTEGUMENT, clade)
        assert "enormous tufted ears" not in pools.pool_for(
            "EARS", pools.EARS, clade)
        assert "a branching antler crown" not in pools.pool_for(
            "HEADGEAR", pools.HEADGEAR, clade)
        assert "cloven hooves" not in pools.pool_for(
            "TERRRESTRIAL_FEATURES", pools.TERRRESTRIAL_FEATURES, clade)
        assert "a blowhole at the crown" not in pools.pool_for(
            "AQUATIC_FEATURES", pools.AQUATIC_FEATURES, clade)
        assert "a normal mammalian gut" not in pools.pool_for(
            "GUT", pools.GUT, clade)


def test_FR_18_2_bird_traits_absent_for_other_clades():
    for clade in NON_BIRD:
        assert "a pelt of interlocking feathers" not in pools.pool_for(
            "INTEGUMENT", pools.INTEGUMENT, clade)
        assert "feathers layered for lift" not in pools.pool_for(
            "AERIAL_FEATURES", pools.AERIAL_FEATURES, clade)


def test_FR_18_3_mythical_accepts_all():
    for name in POOLS:
        pool = getattr(pools, name)
        assert pools.pool_for(name, pool, "mythical") == pool


def test_FR_18_4_filtered_pools_never_empty():
    for name in POOLS:
        pool = getattr(pools, name)
        for clade in pools.CLADES:
            assert pools.pool_for(name, pool, clade), (name, clade)


def test_FR_18_anchor_habitat_consistent(demo_kv):
    out = build_description(demo_kv, device="cpu")
    body = body_of(out)
    anchor_line = next(ln for ln in body.splitlines() if ln.strip())
    creature = pools.anchor_creature(
        anchor_line.split(" — ")[0][0].lower() + anchor_line.split(" — ")[0][1:])
    hab = pools.ANCHOR_HAB[creature]
    hab_name = ["terrestrial", "aerial", "aquatic"][hab]
    assert f"It is {hab_name}" in body
