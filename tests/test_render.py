from gguf_zoo.pools import BEHAVIOR
from gguf_zoo.render import RULE, build_description


def test_FR_9_1_rule_lines_width_72(demo_kv):
    out = build_description(demo_kv, device="cpu")
    assert RULE == "=" * 72
    assert out.count(RULE) >= 2


def test_FR_9_sections_present(demo_kv):
    out = build_description(demo_kv, device="cpu")
    assert "FIELD GUIDE ENTRY —" in out
    assert "Parameters   :" in out
    assert "Quantization :" in out
    assert "SENSES:" in out


def test_FR_4_6_intro_states_size_not_class(demo_kv):
    out = build_description(demo_kv, device="cpu")
    body = out.split(RULE, 2)[2]
    assert "-class creature" not in body
    assert "It measures about " in body and "weighs about " in body


def test_FR_9_2_behavior_sentence_from_pool(demo_kv):
    out = build_description(demo_kv, device="cpu")
    assert any(b in out for b in BEHAVIOR)


def test_FR_3_4_body_contains_no_raw_gguf_strings(demo_kv):
    out = build_description(demo_kv, device="cpu")
    body = out.split(RULE, 2)[2]  # everything after the header block
    for raw in ["Q4_K_M", "mystery_field", "MysteryBeast", "qwen2",
                "Image", "Text", "Embeddings", "42"]:
        assert raw not in body, raw


def test_FR_3_4_body_has_no_mechanism_annotations(demo_kv):
    out = build_description(demo_kv, device="cpu")
    body = out.split(RULE, 2)[2]
    for meta in ["modalities", "unmapped", "metadata", "classifier",
                 "declared", "(from "]:
        assert meta not in body, meta


def test_FR_3_3_identical_trait_lines_collapse(monkeypatch, demo_kv):
    from gguf_zoo import render
    monkeypatch.setattr(
        render, "anomaly_organs",
        lambda *a, **k: [("Nuchal Sail (structure) — same strut.", "k1"),
                         ("Nuchal Sail (structure) — same strut.", "k2"),
                         ("Villus (organ) — other.", "k3")])
    out = render.build_description(demo_kv, device="cpu")
    assert out.count("Nuchal Sail (structure) — same strut.") == 1
    assert "Villus (organ) — other." in out


def test_FR_6_9_senses_render_senses_only(demo_kv):
    out = build_description(demo_kv, device="cpu")
    assert "SENSES:" in out
    body = out.split(RULE, 2)[2]
    assert any(ln.startswith("  • Sight") for ln in body.splitlines())
    assert "->" not in body  # no input -> sense arrows anymore
