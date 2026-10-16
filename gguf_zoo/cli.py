"""argparse CLI for gguf-zoo."""
import argparse
import json
import os
import sys

import numpy as np
import torch

from . import rules
from .model import CreatureNet, encode_features, resolve_device
from .parser import ModelMeta, parse_gguf, classify_block_style
from .render import render
from .rules import HEADS, teacher


def _meta_from_synthetic(spec: dict) -> ModelMeta:
    m = ModelMeta()
    m.architecture = str(spec.get("architecture", spec.get("arch", ""))).lower()
    m.name = str(spec.get("name", "synthetic"))
    m.language = str(spec.get("language", "")).lower()
    m.quant_type = str(spec.get("quant", "unknown"))
    m.quant_block_style = classify_block_style(m.quant_type)
    m.quant_version = int(spec.get("quant_version", 3))
    m.n_params = int(float(spec.get("n_params", 0)))
    m.context_length = int(spec.get("context", spec.get("context_length", 0)))
    m.n_layers = int(spec.get("layers", spec.get("n_layers", 0)))
    m.d_model = int(spec.get("d_model", 0))
    m.d_ff = int(spec.get("d_ff", 0))
    m.n_heads = int(spec.get("n_heads", 0))
    m.n_kv_heads = int(spec.get("n_kv_heads", 0))
    m.head_dim = int(spec.get("head_dim", 0)) or (
        (m.d_model // m.n_heads) if m.n_heads and m.d_model else 0)
    m.vocab_size = int(spec.get("vocab", spec.get("vocab_size", 0)))
    m.is_moe = bool(spec.get("moe", False))
    m.has_vision = bool(spec.get("vision", False))
    m.has_audio = bool(spec.get("audio", False))
    m.is_multimodal = m.has_vision or m.has_audio
    m.sliding_window = bool(spec.get("sliding_window", False))
    m.has_ssm = bool(spec.get("ssm", False))
    m.mixed_dtypes = bool(spec.get("mixed_dtypes", False))
    m.rope_freq_base = float(spec.get("rope_freq_base", 10000))
    m.rope_scaling_type = str(spec.get("rope_scaling", "none"))
    m.partial_rope = bool(spec.get("partial_rope", False))
    m.mtp_tensors = bool(spec.get("mtp", False))
    m.n_special_tokens = int(spec.get("n_special_tokens", 0))
    m.add_bos = bool(spec.get("add_bos", False))
    m.add_eos = bool(spec.get("add_eos", False))
    m.byte_fallback = bool(spec.get("byte_fallback", False))
    m.pooling_type = str(spec.get("pooling", "") or "")
    m.is_encoder_decoder = bool(spec.get("enc_dec", False))
    m.n_split_files = int(spec.get("n_split_files", 1))
    m.rms_epsilon = float(spec.get("rms_epsilon", 1e-5))
    m.n_tensors = int(spec.get("n_tensors", 0))
    return m


def _predict_profile(meta: ModelMeta, ckpt: str, device: str) -> dict:
    dev = resolve_device(device)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model = CreatureNet().to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    x = torch.from_numpy(encode_features(meta)).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = model(x)
    profile = {h: int(logits[h].argmax(dim=1).item()) for h in HEADS}
    profile["patchy_modifier"] = profile["condition"] in (4, 5)
    return profile


def _do_infer(args):
    if bool(args.gguf) == bool(args.synthetic):
        print("error: provide exactly one of --gguf or --synthetic", file=sys.stderr)
        sys.exit(2)
    if args.gguf:
        meta = parse_gguf(args.gguf)
        src = args.gguf
    else:
        meta = _meta_from_synthetic(json.loads(args.synthetic))
        src = "synthetic record"

    if args.ckpt and os.path.exists(args.ckpt):
        profile = _predict_profile(meta, args.ckpt, args.device)
        source = f"student model ({args.ckpt})"
    else:
        print(f"warning: checkpoint '{args.ckpt}' not found - falling back to "
              f"teacher rules. Run 'python gguf_zoo.py train' first to use the "
              f"neural student.", file=sys.stderr)
        profile = teacher(meta)
        source = "teacher rules (fallback)"

    field, prompt = render(meta, profile)
    print(f"=== gguf-zoo field guide ===")
    print(f"source: {src}")
    print(f"profile from: {source}")
    print(f"architecture: {meta.architecture or '?'}  quant: {meta.quant_type}  "
          f"params: {meta.n_params:,}  ctx: {meta.context_length:,}")
    print()
    print("FIELD DESCRIPTION")
    print(field)
    print()
    print("IMAGE PROMPT")
    print(prompt)


def main(argv=None):
    p = argparse.ArgumentParser(prog="gguf-zoo",
                                description="GGUF metadata -> creature field guide")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-data", help="generate synthetic training data")
    g.add_argument("--n", type=int, default=200000)
    g.add_argument("--out", default="data")
    g.add_argument("--seed", type=int, default=42)

    t = sub.add_parser("train", help="train the student network")
    t.add_argument("--data", default="data")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    t.add_argument("--ckpt", default="model.pt")
    t.add_argument("--seed", type=int, default=42)

    i = sub.add_parser("infer", help="describe a GGUF file or synthetic record")
    i.add_argument("--gguf")
    i.add_argument("--synthetic", help="JSON record, e.g. '{\"n_params\": 1e9, ...}'")
    i.add_argument("--ckpt", default="model.pt")
    i.add_argument("--device", default="cpu")

    e = sub.add_parser("evaluate", help="per-head accuracy on the test split")
    e.add_argument("--data", default="data")
    e.add_argument("--ckpt", default="model.pt")
    e.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])

    args = p.parse_args(argv)
    if args.cmd == "gen-data":
        from .data import generate_data
        generate_data(args.n, args.out, args.seed)
    elif args.cmd == "train":
        from .train import train
        train(args.data, args.epochs, args.batch_size, args.device,
              args.ckpt, args.seed)
    elif args.cmd == "infer":
        _do_infer(args)
    elif args.cmd == "evaluate":
        from .train import evaluate
        evaluate(args.data, args.ckpt, args.device)


if __name__ == "__main__":
    main()
