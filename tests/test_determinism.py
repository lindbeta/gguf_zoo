import os
import subprocess
import sys

from gguf_zoo.demo import build_gguf, demo_pairs
from gguf_zoo.parser import parse_gguf_bytes
from gguf_zoo.render import build_description
from gguf_zoo.rules import Die


def test_FR_8_2_identical_twice(demo_kv):
    assert build_description(demo_kv, device="cpu") == build_description(demo_kv, device="cpu")


def test_FR_8_1_die_seeded():
    a = Die("seed").choice([1, 2, 3, 4])
    b = Die("seed").choice([1, 2, 3, 4])
    assert a == b


def test_FR_8_3_independent_of_pythonhashseed():
    env = dict(os.environ)
    outs = []
    for hs in ("0", "12345"):
        env["PYTHONHASHSEED"] = hs
        r = subprocess.run([sys.executable, "-m", "gguf_zoo", "infer", "--demo",
                            "--device", "cpu"], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout)
    assert outs[0] == outs[1]


def test_FR_8_4_unknown_sorted_not_order_dependent():
    kv1 = {"general.architecture": "llama", "general.name": "x",
           "general.file_type": 15, "z.field": 1, "a.field": 2}
    kv2 = {"a.field": 2, "z.field": 1, "general.architecture": "llama",
           "general.name": "x", "general.file_type": 15}
    assert build_description(kv1, device="cpu") == build_description(kv2, device="cpu")
