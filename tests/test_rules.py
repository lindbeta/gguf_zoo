from gguf_zoo import pools
from gguf_zoo.parser import build_meta
from gguf_zoo.rules import (VOCAB_DIET_BOUNDARIES, build_profile, diet_class,
                            estimate_params, extract_features, palette_id)


def test_FR_2_1_arch_regex_matches_any_arch():
    m = build_meta({"general.architecture": "qwen2",
                    "general.architecture.qwen2.embedding_length": 1536,
                    "general.architecture.qwen2.block_length": 28,
                    "general.architecture.qwen2.vocab_length": 151936})
    f = extract_features(m)
    assert f.n_embd == 1536 and f.n_layers == 28 and f.n_vocab == 151936


def test_FR_2_2_multiple_arch_keys_last_wins():
    m = build_meta({"general.architecture.gemma.block_length": 10,
                    "general.architecture.gemma2.block_length": 40})
    assert extract_features(m).n_layers == 40


def test_FR_2_3_sparse_file_no_raise():
    f = extract_features(build_meta({}))
    assert f.n_layers is None and f.n_embd is None
    assert estimate_params(f) is None
    build_profile(build_meta({}))  # must not raise


def test_FR_2_1_diet_boundaries_shared():
    assert VOCAB_DIET_BOUNDARIES == [16000, 48000, 100000, 200000, 400000]
    assert diet_class(16384) == 1  # 16384 falls in second class
    assert diet_class(10000) == 0 and diet_class(500000) == 5


def test_FR_2_estimate_params_moe_scaling():
    m = build_meta({"general.architecture.llama.embedding_length": 4096,
                    "general.architecture.llama.block_length": 32,
                    "general.architecture.llama.vocab_length": 32000,
                    "general.architecture.llama.expert_count": 8,
                    "general.architecture.llama.expert_used_count": 2})
    f = extract_features(m)
    assert estimate_params(f) < estimate_params(
        type(f)(f.arch, f.name, f.file_type, f.n_layers, f.n_embd, f.n_vocab,
                None, None, [], {}))


def test_FR_2_real_gguf_layout():
    m = build_meta({"general.architecture": "gemma4",
                    "gemma4.embedding_length": 3840,
                    "gemma4.block_count": 48,
                    "gemma4.vocab_size": 262144,
                    "gemma4.feed_forward_length": 15360})
    f = extract_features(m)
    assert f.n_embd == 3840 and f.n_layers == 48 and f.n_vocab == 262144
    assert f.unknown == {}  # <arch>.* keys are recognized, not anomalies


def test_FR_2_vocab_from_tokens_count():
    m = build_meta({"general.architecture": "gemma4",
                    "gemma4.embedding_length": 3840, "gemma4.block_count": 48,
                    "tokenizer.ggml.tokens": ["a", "b", "c"]})
    assert extract_features(m).n_vocab == 3
    m2 = build_meta({"general.architecture": "gemma4",
                     "gemma4.embedding_length": 3840, "gemma4.block_count": 48,
                     "tokenizer.ggml.tokens": "<array:8:262144>"})
    assert extract_features(m2).n_vocab == 262144


def test_FR_4_2_size_label_wins():
    m = build_meta({"general.architecture": "qwen35moe",
                    "general.size_label": "35B",
                    "qwen35moe.embedding_length": 2048,
                    "qwen35moe.block_count": 48,
                    "qwen35moe.vocab_size": 151936})
    assert estimate_params(extract_features(m)) == 35_000_000_000
    m2 = build_meta({"general.size_label": "300m"})
    assert estimate_params(extract_features(m2)) == 300_000_000


def test_FR_2_standard_general_keys_recognized():
    m = build_meta({"general.basename": "x", "general.finetune": "it",
                    "general.license": "gemma", "general.quantized_by": "u",
                    "general.size_label": "12B", "general.type": "model",
                    "general.quantization_version": 2,
                    "general.sampling.temp": 1.0,
                    "tokenizer.chat_template": "t",
                    "general.mystery_field": 42})
    f = extract_features(m)
    assert set(f.unknown) == {"general.mystery_field"}


def test_palette_from_tokenizer_lineage_across_arches():
    a = build_meta({"general.architecture": "llama",
                    "tokenizer.ggml.pre": "llama-bpe"})
    b = build_meta({"general.architecture": "deepseek2",
                    "tokenizer.ggml.pre": "llama-bpe"})
    assert palette_id(a) == palette_id(b) == 0
    c = build_meta({"general.architecture": "gemma2",
                    "tokenizer.ggml.pre": "qwen2"})
    assert palette_id(c) == 1


def test_palette_vocab_fallback_when_no_tokenizer():
    a = build_meta({"general.architecture": "llama",
                    "general.architecture.llama.vocab_length": 151936})
    b = build_meta({"general.architecture": "gemma",
                    "general.architecture.gemma.vocab_length": 151936})
    assert palette_id(a) == palette_id(b) == 3  # 100000 <= v < 200000
    assert palette_id(build_meta({})) == 7


def test_palette_lineage_beats_vocab_fallback():
    m = build_meta({"general.architecture": "phi3",
                    "tokenizer.ggml.pre": "nova-bpe",
                    "general.architecture.phi3.vocab_length": 32064})
    assert palette_id(m) == 2


def test_profile_palette_draws_within_family():
    m = build_meta({"general.architecture": "qwen2",
                    "tokenizer.ggml.pre": "qwen2"})
    p = build_profile(m)
    assert p.palette == 1
    assert len(pools.PALETTE_FAMILIES[p.palette]) >= 20
