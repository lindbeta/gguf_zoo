from gguf_zoo.attributes import (ATTR_LABELS, FICTIONAL_ATTRIBUTES,
                                 REAL_ATTRIBUTES, classify_attribute,
                                 eval_attribute_classifier)
from gguf_zoo.render import RULE, build_description

BASE = {"general.architecture": "llama", "general.name": "AttrTest",
        "general.file_type": 15}


def test_FR_16_1_every_key_gets_a_label():
    for key in ["general.mystery_field", "llama.rope.dim", "zzz", "",
                "tokenizer.ggml.add_bos", "n_layers_x", "spiracle_count"]:
        assert classify_attribute(key, device="cpu") in ATTR_LABELS


def test_FR_16_2_label_space():
    assert len(FICTIONAL_ATTRIBUTES) >= 8
    assert not set(FICTIONAL_ATTRIBUTES) & set(REAL_ATTRIBUTES)
    assert set(REAL_ATTRIBUTES) | set(FICTIONAL_ATTRIBUTES) <= set(ATTR_LABELS)
    for display, kind in FICTIONAL_ATTRIBUTES.values():
        assert display and kind in ("organ", "structure")


def test_FR_16_3_fictional_attribute_renders_named_anatomy():
    kv = dict(BASE, **{"general.spiracle_count": 7})
    out = build_description(kv, device="cpu")
    assert "UNIQUE TRAITS" in out
    assert "Spiracle Array (structure)" in out
    assert "spiracle_count" not in out.split(RULE, 2)[2]


def test_FR_16_3_real_attribute_adds_no_line():
    # 'general.vocab_size' is unmapped but classifies to the real 'diet'
    # attribute, which is already rendered from its canonical source.
    kv = dict(BASE, **{"general.vocab_size": 32000})
    out = build_description(kv, device="cpu")
    assert "UNIQUE TRAITS" not in out


def test_FR_16_3_abstain_renders_named_anatomy_without_citation():
    kv = dict(BASE, **{"zzzq": 1})
    out = build_description(kv, device="cpu")
    assert "UNIQUE TRAITS" in out
    assert "zzzq" not in out.split(RULE, 2)[2]


def test_FR_16_4_deterministic_inference():
    for key in ["general.mystery_field", "spiracle_count", "quant_bits"]:
        assert (classify_attribute(key, device="cpu")
                == classify_attribute(key, device="cpu"))


def test_FR_16_4_three_run_consensus_reruns_on_disagreement(monkeypatch):
    from gguf_zoo import attributes as A
    calls = []

    def fake(key, seed, device="auto"):
        calls.append(seed)
        if len(calls) <= 3:
            return (["size", "gut", "eyes"][len(calls) % 3], 0.9)
        return ("size", 0.9)

    monkeypatch.setattr(A, "_predict_attr", fake)
    assert A.classify_attribute("whatever", seed=1, device="cpu") == "size"
    assert len(calls) == 6
    assert len(set(calls)) == 6


def test_FR_16_5_alias_grouped_accuracy():
    res = eval_attribute_classifier(device="cpu")
    assert res["accuracy"] >= 0.95
