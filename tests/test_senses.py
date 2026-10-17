from gguf_zoo.parser import build_meta
from gguf_zoo.pools import SENSE_PROSE
from gguf_zoo.senses import (FICTIONAL_SENSES, INTERNAL_SENSES, VISIBLE_SENSES,
                             Sense, detect_input_types)


def test_FR_6_1_visible_vs_nonvisible_disjoint():
    assert set(VISIBLE_SENSES).isdisjoint(INTERNAL_SENSES)
    assert set(VISIBLE_SENSES).isdisjoint(FICTIONAL_SENSES)
    assert "sight" in VISIBLE_SENSES and "telepathy" not in VISIBLE_SENSES


def test_FR_6_9_every_visible_sense_has_prose():
    for s in VISIBLE_SENSES:
        assert s in SENSE_PROSE and all(SENSE_PROSE[s])


def test_FR_6_8_detect_image_audio_text():
    m = build_meta({"tokenizer.ggml.pre": "clip-vision",
                    "tokenizer.ggml.tokens": (["<image>", "<audio>"] if False
                                              else ["<image>", "<audio>"]),
                    "tokenizer.ggml.eos_token_id": 1})
    types = detect_input_types(m)
    assert "image" in types and "audio" in types and "text" in types


def test_FR_6_8_default_text():
    assert detect_input_types(build_meta({})) == ["text"]


def test_FR_6_8_tools_and_embeddings():
    m = build_meta({"general.architecture.llama.attention.key_length": 128,
                    "tokenizer.ggml.tools": 1})
    types = detect_input_types(m)
    assert "embeddings" in types and "tools" in types


def test_FR_6_1_sense_enum_complete():
    assert Sense.SENSE_UNKNOWN.value == "sense_unknown"
    assert set(FICTIONAL_SENSES) <= {s.value for s in Sense}
