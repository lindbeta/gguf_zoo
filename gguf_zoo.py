"""Thin entry shim. Prefer the `gguf-zoo` console script or `python -m gguf_zoo`."""
from gguf_zoo.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
