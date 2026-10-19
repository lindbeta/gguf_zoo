"""argparse CLI for gguf-zoo (FR-10)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .attributes import (eval_attribute_classifier, train_attribute_classifier)
from .classifier import classify_input_type_consensus, eval_classifier, train_classifier
from .demo import (DEMO_MODELS, demo_bytes, demo_quant_names,
                   normalize_demo_params, resolve_demo_quant)
from .parser import GGUFError, build_meta, read_metadata
from .render import build_description, render_prompt
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
    img_requested = any(getattr(args, a) is not None for a in
                        ("image_url", "image_key", "image_model",
                         "save_image", "save_description"))
    if img_requested:
        if not args.image_url or not args.image_url.startswith(
                ("http://", "https://")):
            print("error: --image-url must be an http(s) URL when image "
                  "generation is requested", file=sys.stderr)
            return 2
        if not args.image_model:
            print("error: --image-model is required for image generation",
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
    description = build_description(
        kv, header=header, device=args.device, params_override=args.params,
        in_depth=args.in_depth, source=source)
    print(description, end="")
    if img_requested:
        from .imagegen import ImageGenError, default_paths, generate_image
        try:
            prompt = render_prompt(kv, device=args.device,
                                   in_depth=args.in_depth, source=source,
                                   header=header)
            key = args.image_key or os.environ.get("OPENAI_API_KEY", "")
            img, ext = generate_image(prompt, args.image_url,
                                      args.image_model, key,
                                      args.image_size)
        except (ImageGenError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        def_img, def_desc = default_paths(build_meta(kv).name, ext)
        img_path = args.save_image or def_img
        desc_path = args.save_description or def_desc
        try:
            Path(img_path).write_bytes(img)
            Path(desc_path).write_text(description, encoding="utf-8")
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"image -> {img_path}", file=sys.stderr)
        print(f"description -> {desc_path}", file=sys.stderr)
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
    infer.add_argument("--image-url", dest="image_url", default=None,
                       metavar="URL",
                       help="OpenAI-compatible image generation API URL "
                            "(http:// or https://; /v1/images/generations is "
                            "appended as needed)")
    infer.add_argument("--image-key", dest="image_key", default=None,
                       metavar="KEY",
                       help="API key for --image-url (defaults to "
                            "$OPENAI_API_KEY; omit for keyless local servers)")
    infer.add_argument("--image-model", dest="image_model", default=None,
                       metavar="MODEL",
                       help="image model name on the target system "
                            "(required when image generation is requested)")
    infer.add_argument("--image-size", dest="image_size",
                       default="1024x1024", metavar="WxH",
                       help="requested image size (default 1024x1024)")
    infer.add_argument("--save-image", dest="save_image", default=None,
                       metavar="PATH",
                       help="write the generated image to PATH (default "
                            "<model_name>_species_image.<ext>)")
    infer.add_argument("--save-description", dest="save_description",
                       default=None, metavar="PATH",
                       help="write the full description to PATH (default "
                            "<model_name>_species_description.md)")
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
