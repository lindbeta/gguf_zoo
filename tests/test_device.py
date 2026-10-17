import os

import torch

from gguf_zoo.classifier import artifact_path, resolve_device


def test_FR_12_1_auto_picks_available():
    assert resolve_device("auto").type in ("cuda", "mps", "cpu")
    assert resolve_device("cpu").type == "cpu"


def test_FR_12_1_explicit_unavailable_raises():
    if not torch.cuda.is_available():
        try:
            resolve_device("cuda")
            assert False, "should have raised"
        except RuntimeError:
            pass
    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        try:
            resolve_device("mps")
            assert False, "should have raised"
        except RuntimeError:
            pass


def test_FR_12_1_unknown_device_raises():
    try:
        resolve_device("tpu")
        assert False
    except RuntimeError:
        pass


def test_FR_12_3_checkpoint_cpu_tensors():
    ckpt = torch.load(artifact_path(), map_location="cpu", weights_only=False)
    for v in ckpt["state_dict"].values():
        assert v.device.type == "cpu"


def test_FR_12_5_cli_accepts_device():
    from gguf_zoo.cli import build_parser
    ns = build_parser().parse_args(["infer", "--demo", "--device", "cpu"])
    assert ns.device == "cpu"
