"""Unit tests for gguf-zoo. Run: python -m unittest discover -s tests -v"""
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gguf_zoo import rules
from gguf_zoo.parser import ModelMeta, parse_gguf
from gguf_zoo.render import render
from gguf_zoo.rules import SIZE_ANCHORS, teacher


def _enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv_str(k, v):
    return _enc_str(k) + struct.pack("<I", 8) + _enc_str(v)


def _kv_u32(k, v):
    return _enc_str(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def _kv_u64(k, v):
    return _enc_str(k) + struct.pack("<I", 10) + struct.pack("<Q", v)


def _kv_bool(k, v):
    return _enc_str(k) + struct.pack("<I", 7) + struct.pack("<B", int(v))


def _kv_str_array(k, vals):
    out = _enc_str(k) + struct.pack("<I", 9) + struct.pack("<I", 8)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += _enc_str(v)
    return out


def _kv_i32_array(k, vals):
    out = _enc_str(k) + struct.pack("<I", 9) + struct.pack("<I", 5)
    out += struct.pack("<Q", len(vals))
    for v in vals:
        out += struct.pack("<i", v)
    return out


def _tensor(name, dims, dtype=8, offset=0):
    out = _enc_str(name) + struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    return out + struct.pack("<I", dtype) + struct.pack("<Q", offset)


def build_tiny_gguf(path):
    kvs = [
        _kv_str("general.architecture", "llama"),
        _kv_str("general.name", "Tiny Test Model"),
        _kv_u32("general.file_type", 15),  # Q4_K_M
        _kv_u32("general.quantization_version", 3),
        _kv_u32("llama.context_length", 4096),
        _kv_u32("llama.block_count", 16),
        _kv_u32("llama.embedding_length", 2048),
        _kv_u32("llama.feed_forward_length", 5632),
        _kv_u32("llama.attention.head_count", 16),
        _kv_i32_array("llama.attention.head_count_kv", [4]),
        _kv_str_array("tokenizer.ggml.tokens", ["a", "b", "c", "d", "e"]),
        _enc_str("tokenizer.ggml.token_type") + struct.pack("<I", 9)
        + struct.pack("<I", 4) + struct.pack("<Q", 5)
        + b"".join(struct.pack("<i", t) for t in (1, 1, 3, 3, 1)),
        _kv_bool("tokenizer.ggml.add_bos_token", True),
        _kv_bool("tokenizer.ggml.add_eos_token", False),
        _kv_u64("general.parameters.total", 1_234_567_890),
    ]
    tensors = [
        _tensor("token_embd.weight", [2048, 5], 12),
        _tensor("blk.0.attn_q.weight", [2048, 2048], 12),
        _tensor("blk.0.ffn_up.weight", [5632, 2048], 8),  # mixed dtype
        _tensor("output.weight", [2048, 5], 12),
    ]
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(kvs)))
        for kv in kvs:
            f.write(kv)
        for t in tensors:
            f.write(t)
        f.write(b"\x00" * 128)  # fake tensor data, must never be read


class TestGGUFRoundTrip(unittest.TestCase):
    def test_header_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tiny.gguf")
            build_tiny_gguf(path)
            m = parse_gguf(path)
            self.assertEqual(m.architecture, "llama")
            self.assertEqual(m.name, "Tiny Test Model")
            self.assertEqual(m.quant_type, "Q4_K_M")
            self.assertEqual(m.quant_block_style, "K-quant")
            self.assertEqual(m.context_length, 4096)
            self.assertEqual(m.n_layers, 16)
            self.assertEqual(m.d_model, 2048)
            self.assertEqual(m.d_ff, 5632)
            self.assertEqual(m.n_heads, 16)
            self.assertEqual(m.n_kv_heads, 4)
            self.assertEqual(m.head_dim, 128)
            self.assertEqual(m.vocab_size, 5)
            self.assertEqual(m.n_special_tokens, 2)
            self.assertTrue(m.add_bos)
            self.assertFalse(m.add_eos)
            self.assertEqual(m.n_params, 1_234_567_890)
            self.assertEqual(m.n_tensors, 4)
            self.assertTrue(m.mixed_dtypes)


class TestLayer1Teacher(unittest.TestCase):
    def test_body_size_ladder(self):
        self.assertEqual(rules.body_size_class(5e7), 0)
        self.assertEqual(rules.body_size_class(2e8), 1)
        self.assertEqual(rules.body_size_class(8e8), 2)
        self.assertEqual(rules.body_size_class(3e9), 3)
        self.assertEqual(rules.body_size_class(15e9), 4)
        self.assertEqual(rules.body_size_class(50e9), 5)
        self.assertEqual(rules.body_size_class(300e9), 6)
        self.assertEqual(rules.body_size_class(700e9), 7)
        self.assertEqual(rules.body_size_class(2e12), 8)
        self.assertEqual(rules.body_size_class(1e13), 9)
        self.assertEqual(rules.body_size_class(5e13), 10)  # titan band
        self.assertEqual(rules.body_size_class(1e14), 11)  # >=100T is kaiju
        self.assertEqual(rules.body_size_class(5e14), 11)

    def test_condition_gut_build_eye_length_diet(self):
        self.assertEqual(rules.condition_class("F16"), 0)
        self.assertEqual(rules.condition_class("Q8_0"), 1)
        self.assertEqual(rules.condition_class("Q6_K"), 2)
        self.assertEqual(rules.condition_class("Q4_K_M"), 3)
        self.assertEqual(rules.condition_class("Q3_K_M"), 4)
        self.assertEqual(rules.condition_class("Q2_K"), 5)
        self.assertEqual(rules.gut_class(4096), 0)
        self.assertEqual(rules.gut_class(32768), 1)
        self.assertEqual(rules.gut_class(131072), 2)
        self.assertEqual(rules.gut_class(262144), 3)
        self.assertEqual(rules.build_class(8192, 2048), 2)   # ratio 4
        self.assertEqual(rules.build_class(6144, 2048), 1)   # ratio 3
        self.assertEqual(rules.build_class(4096, 2048), 0)   # ratio 2
        self.assertEqual(rules.eye_class(64, 64), 0)         # compound
        self.assertEqual(rules.eye_class(32, 32), 1)         # fewer large
        self.assertEqual(rules.eye_class(64, 8), 2)          # GQA
        self.assertEqual(rules.eye_class(64, 1), 3)          # cyclops
        self.assertEqual(rules.body_length_class(10), 0)
        self.assertEqual(rules.body_length_class(32), 1)
        self.assertEqual(rules.body_length_class(60), 2)
        self.assertEqual(rules.body_length_class(80), 3)
        self.assertEqual(rules.diet_class(32000), 1)
        self.assertEqual(rules.diet_class(8000), 0)
        self.assertEqual(rules.diet_class(200000), 4)  # v5: bulk browser


class TestLayer2Teacher(unittest.TestCase):
    def _meta(self, **kw):
        m = ModelMeta()
        for k, v in kw.items():
            setattr(m, k, v)
        return m

    def test_integument_priority(self):
        m = self._meta(has_ssm=True, is_moe=True, quant_block_style="K-quant",
                       quant_type="Q4_K_M")
        self.assertEqual(rules.integument_class(m), 0)  # SSM wins
        m = self._meta(has_ssm=False, is_moe=True, quant_block_style="K-quant",
                       quant_type="Q4_K_M")
        self.assertEqual(rules.integument_class(m), 1)  # MoE second
        m = self._meta(quant_block_style="K-quant", quant_type="Q4_K_M")
        self.assertEqual(rules.integument_class(m), 2)  # fur
        m = self._meta(quant_block_style="legacy", quant_type="Q4_0")
        self.assertEqual(rules.integument_class(m), 3)  # scales
        m = self._meta(quant_block_style="lossless", quant_type="F16")
        self.assertEqual(rules.integument_class(m), 4)  # bare

    def test_palette_lookup(self):
        self.assertEqual(rules.palette_id("llama"), 0)
        self.assertEqual(rules.palette_id("qwen2"), 1)
        self.assertEqual(rules.palette_id("mistral"), 2)
        self.assertEqual(rules.palette_id("gemma2"), 3)
        self.assertEqual(rules.palette_id("phi3"), 4)
        self.assertEqual(rules.palette_id("deepseek2"), 5)
        self.assertEqual(rules.palette_id("falcon"), 6)
        self.assertEqual(rules.palette_id("command-r"), 6)
        self.assertEqual(rules.palette_id("gpt2"), 7)
        self.assertEqual(rules.palette_id(""), 7)

    def test_headgear_partial_rope_override(self):
        self.assertEqual(rules.headgear_id("yarn", False), 2)
        self.assertEqual(rules.headgear_id("dynamic", False), 3)
        self.assertEqual(rules.headgear_id("linear", False), 1)
        self.assertEqual(rules.headgear_id("none", False), 0)
        # partial_rope overrides everything
        self.assertEqual(rules.headgear_id("yarn", True), 4)
        self.assertEqual(rules.headgear_id("none", True), 4)

    def test_diet_breadth_boundaries(self):
        # v5 6-class ladder, lower bound inclusive / upper exclusive
        self.assertEqual(rules.diet_class(15999), 0)    # obligate specialist
        self.assertEqual(rules.diet_class(16384), 1)    # picky browser
        self.assertEqual(rules.diet_class(100352), 3)   # broad-spectrum omnivore
        self.assertEqual(rules.diet_class(399999), 4)   # bulk browser
        self.assertEqual(rules.diet_class(400000), 5)   # all-eater
        # marking_complexity resolves from the SAME boundaries
        self.assertEqual(rules.marking_complexity_id(16384), 1)   # faint
        self.assertEqual(rules.marking_complexity_id(400000), 5)  # extreme ornate
        self.assertEqual(rules.marking_complexity_id(15999), 0)   # plain coat


class TestSharedDietMarkingBoundaries(unittest.TestCase):
    def test_diet_and_marking_share_one_boundary_constant(self):
        sweep = [1, 8192, 15999, 16000, 16384, 47999, 48000, 65536, 99999,
                 100000, 100352, 199999, 200000, 262144, 399999, 400000,
                 512000, 1000000]
        for v in sweep:
            self.assertEqual(rules.diet_class(v),
                             rules.marking_complexity_id(v),
                             f"drift at vocab={v}")


class TestCheckpointIncompatibility(unittest.TestCase):
    def test_v4_checkpoint_rejected(self):
        import torch
        from gguf_zoo.model import CreatureNet
        stub = CreatureNet().state_dict()
        # simulate a v4 checkpoint: 3-class marking head + legacy diet key
        stub["heads.marking_complexity.2.weight"] = torch.zeros(3, 128)
        stub["heads.marking_complexity.2.bias"] = torch.zeros(3)
        stub["heads.diet.2.weight"] = torch.zeros(3, 128)
        stub["heads.diet.2.bias"] = torch.zeros(3)
        with self.assertRaises(RuntimeError) as cm:
            CreatureNet().load_state_dict(stub)
        self.assertIn("v4", str(cm.exception))


class TestRendererDeterminism(unittest.TestCase):
    def test_same_meta_same_output(self):
        m = ModelMeta(architecture="qwen2", name="x", n_params=1e9,
                      quant_type="Q4_K_M", quant_block_style="K-quant",
                      context_length=4096, n_layers=28, d_model=1536,
                      d_ff=8960, n_heads=12, n_kv_heads=2, head_dim=128,
                      vocab_size=151936, has_vision=True)
        f1, p1 = render(m, teacher(m))
        f2, p2 = render(m, teacher(m))
        self.assertEqual(f1, f2)
        self.assertEqual(p1, p2)
        self.assertGreaterEqual(len(f1.split(". ")), 4)

    def test_different_meta_varies(self):
        m1 = ModelMeta(architecture="llama", name="a", n_params=70e9,
                       quant_type="Q8_0", quant_block_style="legacy")
        m2 = ModelMeta(architecture="llama", name="b", n_params=70e9,
                       quant_type="Q8_0", quant_block_style="legacy")
        f1, _ = render(m1, teacher(m1))
        f2, _ = render(m2, teacher(m2))
        self.assertNotEqual(f1, f2)


class TestDeviceFallbackCPU(unittest.TestCase):
    def test_train_one_epoch_cpu(self):
        from gguf_zoo.data import generate_data
        from gguf_zoo.train import train
        with tempfile.TemporaryDirectory() as d:
            data_dir = os.path.join(d, "data")
            generate_data(2000, data_dir, seed=1)
            ckpt = os.path.join(d, "tiny.pt")
            train(data_dir, epochs=1, batch_size=256, device="cpu",
                  ckpt=ckpt, seed=1, verbose=False)
            self.assertTrue(os.path.exists(ckpt))
            import torch
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.assertIn("model", ck)


class TestIntegumentPrioritySSMMoEQ2(unittest.TestCase):
    def test_ssm_moe_q2_resolves_to_exoskeleton_patchy(self):
        m = ModelMeta(has_ssm=True, is_moe=True, quant_type="Q2_K",
                      quant_block_style="K-quant")
        prof = teacher(m)
        self.assertEqual(prof["integument"], 0)  # exoskeleton
        self.assertTrue(prof["patchy_modifier"])


# real-animal keywords that must NOT appear in the fiction-only kaiju class
REAL_ANIMAL_KEYWORDS = [
    "dog", "cat", "horse", "elephant", "whale", "bear", "bird", "fish",
    "spider", "tiger", "lion", "wolf", "mouse", "rat", "cow", "pig",
    "monkey", "shark", "saur", "mammoth", "rhino", "giraffe", "crocodile",
    "alligator", "serpent", "python", "eagle", "falcon", "condor", "insect",
    "beetle", "bee", "ant ", "worm", "slug", "snail", "crab", "frog",
]


class TestAnchorIntegrity(unittest.TestCase):
    def test_min_anchors(self):
        for cls, entry in SIZE_ANCHORS.items():
            self.assertGreaterEqual(len(entry["anchors"]), 10,
                                    f"class {cls} has <10 anchors")

    def test_classes_0_to_9_real_only(self):
        for cls in range(10):
            self.assertEqual(SIZE_ANCHORS[cls]["register"], "real")
            for name, fictional in SIZE_ANCHORS[cls]["anchors"]:
                self.assertFalse(fictional, f"{name} marked fictional")

    def test_kaiju_fiction_only(self):
        entry = SIZE_ANCHORS[11]
        self.assertEqual(entry["register"], "fiction")
        for name, fictional in entry["anchors"]:
            self.assertTrue(fictional, f"{name} must be fictional")
            low = " " + name.lower() + " "
            for kw in REAL_ANIMAL_KEYWORDS:
                self.assertNotIn(kw, low,
                                 f"kaiju anchor '{name}' contains real-animal keyword '{kw}'")

    def test_titan_mixed(self):
        entry = SIZE_ANCHORS[10]
        self.assertEqual(entry["register"], "mixed")
        real = sum(1 for _, f in entry["anchors"] if not f)
        fiction = sum(1 for _, f in entry["anchors"] if f)
        self.assertGreaterEqual(real, 3)
        self.assertGreaterEqual(fiction, 3)


if __name__ == "__main__":
    unittest.main()
