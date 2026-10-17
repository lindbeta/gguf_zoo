from gguf_zoo.rules import (body_dimensions, estimate_params, length_phrase,
                            size_tier)


def test_FR_4_1_all_boundaries():
    cases = [
        (50e6, "Submicroscopic"), (200e6, "Microscopic"), (800e6, "Minute"),
        (2e9, "Small"), (10e9, "Medium-small"), (50e9, "Medium"),
        (300e9, "Medium-large"), (800e9, "Large"), (2e12, "Megafauna"),
        (10e12, "Gigantic"), (50e12, "Titanic"), (300e12, "Kaiju"),
        (600e12, "Beyond Kaiju"),
    ]
    for n, name in cases:
        assert size_tier(int(n))[0] == name, name


def test_FR_4_1_boundary_strict_less():
    assert size_tier(int(100e6))[0] == "Microscopic"   # 100e6 not < 100e6
    assert size_tier(int(100e6) - 1)[0] == "Submicroscopic"


def test_FR_4_2_none_unclassifiable():
    assert size_tier(None) == ("Unclassifiable", "a size that resists measurement")
    assert estimate_params.__name__ == "estimate_params"


def test_FR_4_5_format_uses_B_and_T():
    from gguf_zoo.render import _fmt_params
    assert _fmt_params(7_000_000_000) == "~7.0B"
    assert _fmt_params(2_000_000_000_000) == "~2.0T"


def test_FR_4_6_body_dimensions():
    assert body_dimensions(None) is None
    assert body_dimensions(0) is None
    microbe = body_dimensions(10**6)
    kaiju = body_dimensions(10**15)
    assert microbe and kaiju
    assert microbe[0].endswith("µm") and microbe[1].endswith("µg")
    assert kaiju[0].endswith("m") and kaiju[1].endswith("megatonnes")
    mid = body_dimensions(10**12)
    assert mid[0] == "2.5 m" and mid[1] == "500.0 kg"


def test_FR_4_6_monotonic():
    len_u = {"µm": 1e-6, "cm": 1e-2, "m": 1.0}
    mass_u = {"µg": 1e-9, "mg": 1e-6, "g": 1e-3, "kg": 1.0,
              "tonnes": 1e3, "megatonnes": 1e6}
    prev_l = prev_m = -1.0
    for e in range(6, 16):
        l, m = body_dimensions(10**e)
        lv = float(l.split()[0]) * len_u[l.split()[1]]
        mv = float(m.split()[0]) * mass_u[m.split()[1]]
        assert lv >= prev_l and mv >= prev_m
        prev_l, prev_m = lv, mv


def test_FR_4_6_length_phrase_matches_anatomy():
    assert length_phrase("a full-grown grey wolf") == "from nose to tail"
    assert length_phrase("a full-grown eagle") == "from beak to tail"
    assert length_phrase("a young goldfish") == "from snout to tail"
    assert length_phrase("a full-grown ladybird") == "from head to tail"
    assert length_phrase("a young tree frog") == "from head to rump"
    assert length_phrase("a full-grown earthworm") == "from end to end"
    assert length_phrase("a full-grown paramecium") == "across"
    assert length_phrase("a young soil mite") == "from end to end"
    assert length_phrase("a full-grown silverfish") == "from head to tail"
    assert length_phrase("a full-grown ocean-draining serpent") == "from snout to tail"
