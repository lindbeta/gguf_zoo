from gguf_zoo.render import RULE, build_description, render_prompt


def test_FR_17_1_prompt_section_present(demo_kv):
    out = build_description(demo_kv, device="cpu")
    assert "IMAGE PROMPT:" in out
    prompt = out.split("IMAGE PROMPT:\n", 1)[1]
    assert "Wildlife field-guide illustration" in prompt


def test_FR_17_1_prompt_covers_description_details(demo_kv):
    out = build_description(demo_kv, device="cpu")
    prompt = out.split("IMAGE PROMPT:\n", 1)[1]
    body = out.split(RULE, 2)[2].split("IMAGE PROMPT:")[0]
    # every bullet/section of the description reappears in the prompt
    for probe in ["weighing about", "on the head:", "foraging as",
                  "condition:", "behaving thus:"]:
        assert probe in prompt, probe
    assert "It wears" not in prompt  # prose rephrased, not copy-pasted whole
    assert "CONDITION —" not in prompt
    assert "UNIQUE TRAITS" not in prompt
    assert body  # sanity


def test_FR_17_2_style_tags_close_prompt(demo_kv):
    prompt = render_prompt(demo_kv, device="cpu")
    assert prompt.rstrip().endswith("scientifically meticulous")
    assert "watercolor and ink" in prompt


def test_FR_17_3_prompt_deterministic_and_matches_section(demo_kv):
    p1 = render_prompt(demo_kv, device="cpu")
    p2 = render_prompt(demo_kv, device="cpu")
    assert p1 == p2
    out = build_description(demo_kv, device="cpu")
    section = out.split("IMAGE PROMPT:\n", 1)[1].strip()
    assert p1 == section


def test_FR_17_3_prompt_has_no_raw_metadata(demo_kv):
    prompt = render_prompt(demo_kv, device="cpu")
    for raw in ["Q4_K_M", "mystery_field", "MysteryBeast", "qwen2",
                "Image", "Text", "Embeddings", "42", "modalities",
                "metadata", "classifier"]:
        assert raw not in prompt, raw
