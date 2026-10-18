"""argparse CLI for gguf-zoo (FR-10)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .attributes import (eval_attribute_classifier, train_attribute_classifier)
from .classifier import classify_input_type_consensus, eval_classifier, train_classifier
from .demo import (DEMO_MODELS, demo_bytes, demo_quant_names,
                   normalize_demo_params, resolve_demo_quant)
from .parser import GGUFError, build_meta, read_metadata
from .render import build_description
from .senses import detect_input_types


def cmd_infer(args) -> int:
    chosen = [bool(args.file), bool(args.meta), bool(args.demo)]
    if sum(chosen) != 1:
        print("error: exactly one of <file>, --meta, or --demo is required",
              file=sys.stderr)
        return 2
    if args.in_depth and args.meta:
        print("error: --in-depth needs tensor data; not available with --meta",
              file=sys.stderr)
        return 2
    demo_kwargs = {}
    if args.demo:
        try:
            if args.demo_params:
                demo_kwargs["params"] = normalize_demo_params(args.demo_params)
            if args.demo_quant:
                demo_kwargs["quant"] = resolve_demo_quant(args.demo_quant)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.demo_name:
            demo_kwargs["name"] = args.demo_name
    if args.demo and args.save:
        Path(args.save).write_bytes(demo_bytes(args.demo, **demo_kwargs))
    try:
        if args.demo:
            data = demo_bytes(args.demo, with_tensors=args.in_depth,
                              **demo_kwargs)
            from .parser import parse_gguf_bytes
            kv, header = parse_gguf_bytes(data)
            source = data if args.in_depth else None
        elif args.meta:
            kv, header, source = json.loads(
                Path(args.meta).read_text(encoding="utf-8")), {}, None
        else:
            kv, header = read_metadata(args.file)
            source = args.file if args.in_depth else None
    except (GGUFError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(build_description(kv, header=header, device=args.device,
                            params_override=args.params,
                            in_depth=args.in_depth, source=source),
          end="")
    return 0


def cmd_train(args) -> int:
    path = train_classifier(seed=args.seed, device=args.device, out_path=args.out)
    print(f"trained -> {path}", file=sys.stderr)
    return 0


def cmd_eval(args) -> int:
    res = eval_classifier(device=args.device)
    print(f"alias-grouped accuracy: {res['accuracy'] * 100:.2f}%")
    labels = res["labels"]
    print("confusion matrix (rows=true, cols=pred):")
    print("      " + "".join(f"{l[:5]:>6}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:5]:>5} " + "".join(f"{v:6d}" for v in res["confusion"][i]))
    ok = res["accuracy"] >= 0.95
    print("all targets met" if ok else "BELOW TARGET")
    return 0 if ok else 1


def cmd_train_attrs(args) -> int:
    path = train_attribute_classifier(seed=args.seed, device=args.device,
                                      out_path=args.out)
    print(f"trained -> {path}", file=sys.stderr)
    return 0


def cmd_eval_attrs(args) -> int:
    res = eval_attribute_classifier(device=args.device)
    print(f"alias-grouped accuracy: {res['accuracy'] * 100:.2f}%")
    labels = res["labels"]
    print("confusion matrix (rows=true, cols=pred):")
    print("      " + "".join(f"{l[:5]:>6}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:5]:>5} " + "".join(f"{v:6d}" for v in res["confusion"][i]))
    ok = res["accuracy"] >= 0.95
    print("all targets met" if ok else "BELOW TARGET")
    return 0 if ok else 1


def cmd_senses(args) -> int:
    res = classify_input_type_consensus(args.name, seed=args.seed, device=args.device)
    print(f"{args.name!r} -> {res.sense} (conf={res.confidence:.2f}, {res.source})")
    return 0


def cmd_detect(args) -> int:
    try:
        kv, _ = read_metadata(args.file)
    except (GGUFError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    meta = build_meta(kv)
    for name in detect_input_types(meta):
        res = classify_input_type_consensus(name, device=args.device)
        print(f"{name:<12} -> {res.sense}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gguf-zoo", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("infer", help="generate a field-guide entry")
    infer.add_argument("file", nargs="?", help="path to a .gguf file")
    infer.add_argument("--meta", help="flat JSON of key->value metadata")
    infer.add_argument("--demo", nargs="?", const="qwen2", metavar="NAME",
                       choices=sorted(DEMO_MODELS),
                       help="use a built-in demo model (qwen2 [default] or mixtral)")
    infer.add_argument("--demo-name", dest="demo_name", default=None,
                       metavar="NAME",
                       help="with --demo: override the demo's model name")
    infer.add_argument("--demo-params", dest="demo_params", default=None,
                       metavar="N",
                       help="with --demo: parameter count — a full number "
                            "(7155000000) or a case-insensitive k/m/b/t "
                            "suffix (300m, 7B, 1.5t)")
    infer.add_argument("--demo-quant", dest="demo_quant", default=None,
                       metavar="QUANT",
                       help="with --demo: quantization level (case-insensitive); "
                            "accepted: " + ", ".join(demo_quant_names()))
    infer.add_argument("--save", help="write the demo GGUF bytes to PATH")
    infer.add_argument("--params", type=int, help="override the parameter estimate")
    infer.add_argument("--in-depth", action="store_true", dest="in_depth",
                       help="read sampled tensor weights for extra detail (FR-19)")
    infer.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    infer.set_defaults(func=cmd_infer)

    tr = sub.add_parser("train-senses", help="train the sense classifier")
    tr.add_argument("--out", default=None)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval-senses", help="evaluate the sense classifier")
    ev.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ev.set_defaults(func=cmd_eval)

    ta = sub.add_parser("train-attributes", help="train the unmapped-field -> attribute classifier")
    ta.add_argument("--out", default=None)
    ta.add_argument("--seed", type=int, default=0)
    ta.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ta.set_defaults(func=cmd_train_attrs)

    ea = sub.add_parser("eval-attributes", help="evaluate the attribute classifier")
    ea.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ea.set_defaults(func=cmd_eval_attrs)

    se = sub.add_parser("senses", help="classify a single input-type name")
    se.add_argument("--name", required=True)
    se.add_argument("--seed", type=int, default=0)
    se.add_argument("--draws", type=int, default=3)
    se.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    se.set_defaults(func=cmd_senses)

    de = sub.add_parser("detect-senses", help="classify a file's declared inputs")
    de.add_argument("file")
    de.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    de.set_defaults(func=cmd_detect)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
