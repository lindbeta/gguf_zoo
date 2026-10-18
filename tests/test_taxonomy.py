from gguf_zoo.parser import build_meta
from gguf_zoo.rules import build_profile
from gguf_zoo.taxonomy import binomial, epithet_candidates, genus_for, gloss


def _prof(name, **extra):
    kv = {"general.name": name, "general.architecture": "llama",
          "general.file_type": 15}
    kv.update(extra)
    m = build_meta(kv)
    return build_profile(m), m


def test_FR_7_1_genus_from_name():
    p, m = _prof("MysteryBeast")
    genus, gender, band = genus_for(p, m)
    assert genus.startswith("Myster")
    assert gender in ("m", "f", "n")


def test_FR_7_1_noise_and_missing_name_fallback():
    p, m = _prof("gguf base 7b")
    genus, _, _ = genus_for(p, m)
    assert genus.startswith("Ignot")


def test_FR_7_2_epithets_from_traits_gender_agreement():
    p, m = _prof("X", **{"general.architecture.llama.vocab_length": 500000})
    cands = epithet_candidates(p)
    assert cands  # at least one epithet fires
    name = binomial(p, m)
    assert name.split()[0] == genus_for(p, m)[0]


def test_FR_7_4_authority_from_author():
    p, m = _prof("X")
    m.author = "Microsoft"
    assert "Microsoftia" in binomial(p, m)


def test_FR_7_5_gloss_english():
    from gguf_zoo import pools
    p, m = _prof("X")
    g = gloss(p, m)
    assert any(ph in g for ph in pools.HABITAT_PHRASES["terrestrial"])
    assert not any(l.isupper() for l in g.replace(" ", ""))


def test_FR_7_6_deterministic_binomial():
    p1, m1 = _prof("MysteryBeast")
    p2, m2 = _prof("MysteryBeast")
    assert binomial(p1, m1) == binomial(p2, m2)
