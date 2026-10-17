"""Unit tests for gguf-zoo. Run: python -m unittest discover -s tests -v"""
import os
import re
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gguf_zoo import rules
from gguf_zoo import pools
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

    def test_pools_live_same_class_varies(self):
        # same class, different files -> different phrasings (pools are live)
        metas = [ModelMeta(architecture="llama", name=f"v{i}", n_params=70e9,
                           quant_type="Q8_0", quant_block_style="legacy",
                           context_length=131072, n_layers=80, d_model=8192,
                           d_ff=28672, n_heads=64, n_kv_heads=8,
                           head_dim=128, vocab_size=32000, language="en")
                 for i in range(20)]
        descs = {render(m, teacher(m))[0] for m in metas}
        self.assertGreater(len(descs), 1)


class TestPoolHygiene(unittest.TestCase):
    def _all_pools(self):
        items = []
        for trait, classes in pools.TRAIT_POOLS.items():
            for cls, pool in classes.items():
                items.append((f"{trait}[{cls}]", pool))
        for name, pool in pools.SLOT_POOLS.items():
            items.append((f"slot:{name}", pool))
        return items

    def test_min_entries_and_grammar(self):
        for name, pool in self._all_pools():
            self.assertGreaterEqual(len(pool), 3, name)
            for entry in pool:
                self.assertEqual(entry, entry.strip(), name)
                self.assertEqual(entry, entry.lower(), f"{name}: {entry}")
                self.assertNotIn(entry[-1], ".,;!", f"{name}: {entry}")

    def test_no_preservation_vocabulary(self):
        for name, pool in self._all_pools():
            for entry in pool:
                for bad in pools.PRESERVATION_BANLIST:
                    self.assertNotIn(bad, entry,
                                     f"{name}: preservation word '{bad}' in "
                                     f"'{entry}'")

    def test_no_cross_trait_leakage(self):
        for trait, classes in pools.TRAIT_POOLS.items():
            self.assertEqual(sorted(classes), list(range(rules.HEAD_SIZES[trait])))
            for cls, pool in classes.items():
                for entry in pool:
                    for other, kws in pools.CROSS_TRAIT_KEYWORDS.items():
                        if other == trait:
                            continue
                        for kw in kws:
                            pat = r"\b" + re.escape(kw) + r"\b"
                            self.assertIsNone(
                                re.search(pat, entry),
                                f"{trait}[{cls}] '{entry}' leaks {other} "
                                f"keyword '{kw}'")


class TestPoolStreams(unittest.TestCase):
    def test_pick_deterministic(self):
        self.assertEqual(pools.pick("abc123", "condition", 3),
                         pools.pick("abc123", "condition", 3))
        self.assertEqual(pools.pick_slot("abc123", "pose"),
                         pools.pick_slot("abc123", "pose"))

    def test_streams_decorrelated(self):
        # without the trait salt, condition and build would always land on the
        # same pool index for a given file; independent streams break that
        cond_pool = pools.TRAIT_POOLS["condition"][3]
        build_pool = pools.TRAIT_POOLS["build"][2]
        same = sum(
            cond_pool.index(pools.pick(f"file{i}", "condition", 3))
            == build_pool.index(pools.pick(f"file{i}", "build", 2))
            for i in range(300))
        self.assertLess(same, 300)


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


class TestAnchorIntegrity(unittest.TestCase):
    def test_min_anchors(self):
        for cls, entry in SIZE_ANCHORS.items():
            self.assertGreaterEqual(len(entry["anchors"]), 10,
                                    f"class {cls} has <10 anchors")


class TestAnchorTaxonomy(unittest.TestCase):
    def test_every_anchor_has_taxonomy(self):
        for cls, entry in SIZE_ANCHORS.items():
            for name in entry["anchors"]:
                taxon = rules.anchor_taxonomy(name)
                self.assertTrue(taxon, f"{name} has empty taxonomy")
                self.assertGreaterEqual(len(taxon.split()), 2,
                                        f"{name}: {taxon} not binomial")

    def test_prompt_includes_taxonomy(self):
        m = ModelMeta(architecture="qwen2", name="x", n_params=1e9,
                      quant_type="Q4_K_M", quant_block_style="K-quant",
                      context_length=4096, n_layers=28, d_model=1536,
                      d_ff=8960, n_heads=12, n_kv_heads=2, head_dim=128,
                      vocab_size=151936, has_vision=True)
        _, prompt = render(m, teacher(m))
        anchors = SIZE_ANCHORS[int(teacher(m)["body_size"])]["anchors"]
        taxons = {rules.anchor_taxonomy(a) for a in anchors}
        self.assertTrue(any(t in prompt for t in taxons),
                        f"no taxonomy binomial in prompt: {prompt}")


if __name__ == "__main__":
    unittest.main()


def _taxo_meta(arch="llama", name="x.gguf", **kw):
    m = ModelMeta(architecture=arch, name=name)
    for k, v in kw.items():
        setattr(m, k, v)
    return m


STEM_ARCH = {
    "Llama": "llama", "Qwen": "qwen", "Mistral": "mistral",
    "Gemma": "gemma", "Phi": "phi", "Profund": "deepseek",
    "Fav": "mixtral", "Lorica": "mamba", "Ignot": "starcoder",
}

BAND_CLS = {"micro": 0, "insect": 2, "small": 4,
            "mammal": 6, "mega": 8, "colossal": 10}

EPITHET_PROFILE = {
    "fur": {"integument": 2}, "scales": {"integument": 3},
    "quills": {"integument": 1}, "exoskeleton": {"integument": 0},
    "bare": {"integument": 4}, "patchy": {"integument": 5},
    "immaculate": {"condition": 0}, "faded": {"condition": 2},
    "corrupt": {"condition": 4},
    "horned": {"headgear": 1}, "unicorn": {"headgear": 4},
    "hornless": {"headgear": 0}, "cyclops": {"eye": 3},
    "small_gut": {"gut": 0}, "ruminant": {"gut": 2},
    "great_gut": {"gut": 3},
    "specialist": {"diet_breadth": 0}, "alleater": {"diet_breadth": 5},
    "tan": {"palette": 0}, "jade": {"palette": 1}, "violet": {"palette": 4},
    "gold": {"palette": 5}, "dark": {"palette": 7},
    "ignotus": {},
}


class TaxonomyTests(unittest.TestCase):
    def test_determinism(self):
        from gguf_zoo import taxonomy
        meta = _taxo_meta("qwen2", "a.gguf", n_params=7e9)
        prof = {"body_size": 6, "integument": 2, "condition": 0,
                "headgear": 2, "gut": 1, "diet_breadth": 3, "palette": 1}
        self.assertEqual(taxonomy.binomial(prof, meta),
                         taxonomy.binomial(prof, meta))

    def test_gender_agreement(self):
        from gguf_zoo import taxonomy
        for stem, arch in STEM_ARCH.items():
            for band, cls in BAND_CLS.items():
                genus, gender, _ = taxonomy.genus_for(
                    _taxo_meta(arch), {"body_size": cls})
                self.assertIn(genus, taxonomy.GENUS_FUSIONS[(stem, band)])
                for key, forms in taxonomy.EPITHETS.items():
                    prof = dict(EPITHET_PROFILE[key], body_size=cls)
                    name = taxonomy.binomial(prof, _taxo_meta(arch))
                    self.assertIn(f" {forms[gender]} ", f" {name} ",
                                  f"{genus} ({gender}) + {key}: {name}")

    def test_table_integrity(self):
        import re
        from gguf_zoo import taxonomy
        for stem in STEM_ARCH:
            for band in taxonomy.BANDS:
                self.assertTrue(taxonomy.GENUS_FUSIONS[(stem, band)],
                                f"empty cell {stem}/{band}")
        used = set()
        used.update(taxonomy.INTEG_MAP.values(), taxonomy.COND_MAP.values(),
                    taxonomy.HEADGEAR_MAP.values(), taxonomy.GUT_MAP.values(),
                    taxonomy.DIET_MAP.values(), taxonomy.PALETTE_MAP.values())
        used.update(["faded", "hornless", "cyclops", "ignotus"])
        for key in used:
            self.assertIn(key, taxonomy.EPITHETS)
        for cell in taxonomy.GENUS_FUSIONS.values():
            for genus in cell:
                self.assertTrue(genus.isascii())
                for run in re.findall(r"[bcdfghjklmnpqrstvwxyz]{3,}",
                                      genus.lower()):
                    self.assertEqual(run, "str",
                                     f"{genus} has consonant cluster {run!r}")

    def test_priority_and_skips(self):
        from gguf_zoo import taxonomy
        self.assertEqual(taxonomy.epithet_candidates({"body_size": 6}),
                         ["ignotus"])
        self.assertEqual(
            taxonomy.epithet_candidates(
                {"integument": 2, "headgear": 0, "condition": 2}),
            ["fur"])
        cands = taxonomy.epithet_candidates({"integument": 2, "headgear": 2})
        self.assertEqual(cands[:2], ["fur", "horned"])
        name = taxonomy.binomial({"body_size": 2, "integument": 2,
                                  "headgear": 2}, _taxo_meta("llama"))
        self.assertRegex(name, r"^Llama\w+ pilosa cornuta")

    def test_strict_mode(self):
        import json
        from gguf_zoo import taxonomy
        with tempfile.TemporaryDirectory() as d:
            reg = os.path.join(d, "names.json")
            prof = {"body_size": 8, "integument": 2, "headgear": 2,
                    "palette": 1}
            m1 = _taxo_meta("phi3", "one.gguf")
            m2 = _taxo_meta("phi3", "two.gguf")
            loose = taxonomy.binomial(prof, m1)
            self.assertEqual(loose, taxonomy.binomial(prof, m2))
            n1 = taxonomy.binomial(prof, m1, strict=True, registry_path=reg)
            n2 = taxonomy.binomial(prof, m2, strict=True, registry_path=reg)
            self.assertNotEqual(n1, n2)
            with open(reg) as f:
                self.assertTrue(json.load(f))
        with tempfile.TemporaryDirectory() as d:
            reg = os.path.join(d, "names.json")
            a1 = taxonomy.binomial(prof, m1, strict=True, registry_path=reg)
            a2 = taxonomy.binomial(prof, m2, strict=True, registry_path=reg)
            self.assertEqual((a1, a2), (n1, n2))

    def test_authority(self):
        from gguf_zoo import taxonomy
        m = _taxo_meta("qwen2", "thing.gguf")
        self.assertIn(" qwen2 ", f" {taxonomy.binomial({'body_size': 6}, m)} ")
        mf = _taxo_meta("llama", "mymodel-v2", base_model="Llama-3.1-8B")
        self.assertIn(" & Llama", taxonomy.binomial({"body_size": 6}, mf))
        self.assertEqual(taxonomy._latinize_author("Microsoft"), "Microsoftia")
        self.assertEqual(taxonomy._latinize_author("bartowski"), "bartowski")

    def test_year_omitted_when_unknown(self):
        from gguf_zoo import taxonomy
        self.assertNotIn(",", taxonomy.binomial({"body_size": 6},
                                                 _taxo_meta()))
        m = _taxo_meta()
        m.year = 2025
        self.assertIn(", 2025", taxonomy.binomial({"body_size": 6}, m))

    def test_speed(self):
        import time
        from gguf_zoo import taxonomy
        meta = _taxo_meta("qwen2", "s.gguf")
        prof = {"body_size": 6, "integument": 2, "condition": 0}
        taxonomy.binomial(prof, meta)
        t0 = time.perf_counter()
        for _ in range(100):
            taxonomy.binomial(prof, meta)
        self.assertLess((time.perf_counter() - t0) / 100, 0.001)

    def test_render_includes_name_and_plaque(self):
        from gguf_zoo import taxonomy
        meta = _taxo_meta("llama", "r.gguf", n_params=7e9,
                          quant_type="F16", quant_block_style="lossless")
        prof = teacher(meta)
        field, prompt = render(meta, prof)
        name = taxonomy.binomial(prof, meta)
        self.assertTrue(field.startswith(f"*{name}*"))
        self.assertIn("museum diorama", prompt)
        self.assertIn("specimen plaque", prompt)
        macro = _taxo_meta("llama", "m.gguf", n_params=1e6,
                           quant_type="Q4_K_M")
        _, p2 = render(macro, teacher(macro))
        self.assertNotIn("specimen plaque", p2)


