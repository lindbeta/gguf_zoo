import gguf_zoo.classifier as clf
from gguf_zoo.classifier import (classify_input_type,
                                 classify_input_type_consensus, eval_classifier)
from gguf_zoo.senses import VISIBLE_SENSES, SenseResult


def test_FR_6_required_mappings():
    assert classify_input_type("image", 0).sense == "sight"
    assert classify_input_type("video", 0).sense == "sight"
    assert classify_input_type("audio", 0).sense == "hearing"
    assert classify_input_type("sound", 0).sense == "hearing"
    assert classify_input_type("text", 0).sense == "touch"
    assert classify_input_type("image", 0).source == "override"


def test_FR_6_7_alias_grouped_accuracy():
    res = eval_classifier(device="cpu")
    assert res["accuracy"] >= 0.95


def test_FR_6_10_consensus_requires_three_agreeing(monkeypatch):
    calls = {"n": 0}

    def fake(name, draw, device="auto"):
        calls["n"] += 1
        # first round (3 calls) disagree, second round agree
        if calls["n"] <= 3:
            senses = ["sight", "hearing", "touch"]
            return SenseResult(senses[(calls["n"] - 1) % 3], 0.9, "model")
        return SenseResult("sight", 0.9, "model")

    monkeypatch.setattr(clf, "classify_input_type", fake)
    res = classify_input_type_consensus("whatever", seed=1)
    assert res.sense == "sight"
    assert calls["n"] >= 6  # at least two rounds


def test_FR_6_10_consensus_reproducible():
    a = classify_input_type_consensus("audio", seed=42, device="cpu")
    b = classify_input_type_consensus("audio", seed=42, device="cpu")
    assert a.sense == b.sense


def test_FR_6_10_no_nonvisible_emitted():
    for name in ["image", "audio", "text", "thermal", "voltage", "unknownxyz"]:
        res = classify_input_type_consensus(name, seed=3, device="cpu")
        assert res.sense in VISIBLE_SENSES or res.sense == "sense_unknown"
