# gguf-zoo

`gguf-zoo` reads the **metadata header** of a `.gguf` model file (never the
weights) and procedurally generates a deterministic fictional **creature field
guide**: body size from the parameter estimate, physical condition from the
quantization, senses from the declared input modalities (via a small trained
classifier), and a New-Latin binomial from the model name and traits.

## Features

- GGUF v2/v3 header-only parser (no tensor reads, no third-party GGUF lib).
- Unknown metadata fields become deterministic "cryptid anomaly" organs.
- Parameter-count size ladder (virus → kaiju) and quant → health ladder with
  additional condition descriptions.
- A small **trained** input-type → sense classifier (GPU when available) with a
  three-run consensus; only externally visible senses are rendered, the rest
  become anomalies (with a focus on fictional senses).
- A second **trained** classifier attributes every unmapped metadata field to a
  best-match physical attribute (real or fictional), so no field is left
  unmapped and no raw key, value, quant name, or input name ever leaks into
  the guide body — only the header block quotes the file.
- Balanced terrestrial / aerial / aquatic habitats with matching locomotion
  features (feathers, wings, fins, gas bladders, …) and plant-like variants.
- Fully deterministic: same input ⇒ byte-identical output.

## Requirements

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/)
- `numpy`, `torch` (installed by `uv sync`). A CUDA/MPS GPU is used for training
  and inference when present; otherwise CPU.

## Setup

```bash
uv sync
```

## Quickstart

```bash
uv run gguf-zoo infer --demo
```

## Usage

| Command | Description |
|---|---|
| `uv run gguf-zoo infer <file.gguf>` | generate a field guide from a GGUF file |
| `uv run gguf-zoo infer --meta metadata.json` | generate from a flat JSON of `key -> value` |
| `uv run gguf-zoo infer --demo [--save out.gguf]` | use a built-in synthetic model |
| `uv run gguf-zoo infer <file> --params N` | override the parameter estimate |
| `uv run gguf-zoo train-senses [--seed N] [--device D]` | (re)train the sense classifier |
| `uv run gguf-zoo eval-senses` | report alias-grouped accuracy + confusion matrix |
| `uv run gguf-zoo train-attributes [--seed N] [--device D]` | (re)train the field→attribute classifier |
| `uv run gguf-zoo eval-attributes` | attribute-model accuracy + confusion matrix |
| `uv run gguf-zoo senses --name "mel spectrogram"` | classify one input-type name |
| `uv run gguf-zoo detect-senses <file.gguf>` | classify a file's declared inputs |

Flags: `--device auto|cuda|mps|cpu` (default `auto`), `--seed`, `--draws`.

## Training the classifiers

```bash
uv run gguf-zoo train-senses --device auto
uv run gguf-zoo eval-senses
uv run gguf-zoo train-attributes --device auto
uv run gguf-zoo eval-attributes
```

The sense checkpoint is written to `gguf_zoo/artifacts/sense_clf.pt` (override
with the `GGUF_ZOO_ARTIFACT` env var); the attribute checkpoint to
`gguf_zoo/artifacts/field_attr_clf.pt` (override with `GGUF_ZOO_ATTR_ARTIFACT`).
Both are trained on built-in synthetic datasets labeled by a deterministic
teacher. The attribute classifier maps unmapped metadata field keys to
best-match physical attributes — the real FR-2 trait families plus eight
fictional attributes (Spiracle Array, Fermentation Crop, …) that absorb unknown
fields as invented anatomy instead of colliding with real traits.

## Example input → output

Input (built-in demo metadata: a qwen2-like multimodal model, Q4_K_M):

```json
{"general.architecture": "qwen2", "general.name": "MysteryBeast Qwen2 Demo",
 "general.file_type": 15, "tokenizer.ggml.pre": "qwen2-vision"}
```

Output of `uv run gguf-zoo infer --demo`:

```
========================================================================
FIELD GUIDE ENTRY — Mysteryapis pilosa viridis Zookeeperia (the Worn)
Source model : MysteryBeast Qwen2 Demo (qwen2)
Parameters   : ~1.3B  ->  size class: Small (housefly to honeybee)
Quantization : Q4_K_M (Q4_K_M K-quant)  ->  condition: weathered
========================================================================

A young ground beetle — a hairy, green insect of the written word. It measures about 0.7 cm from nose to tail and weighs about 8 mg. It wears a glaze of polished shell, colored milky opal under a highland heather bloom; colors that gleam wet then dull, a mottled blotch pattern, a plain back with a marked belly. On the head: a spined ridge from brow to nape with ears that fold flat; eyes that blink sideways. Short and thickset, round and barrel-girthed, it carries a liver-heavy digesting mass and forages as a browser of high leaves. It is terrestrial, bearing burrowing foreclaws, a low crawling gait.

SENSES:
  • Sight          humming eyes like wet garnets that glow when fed
  • Touch          blood-hot velvety manipulator pads that never fully close
  • Vibration      frosted long sensory hairs that split to see around corners

CONDITION — WEATHERED: the specimen is stiff in every membrane.
  • mottled and asymmetric — the two sides never quite match
  • patchily mangled in blocky clumps, whole swaths of tissue replaced by coarse scar

UNIQUE TRAITS:
  • Nuchal Sail (structure) — a jointed spur that rotates freely in its socket.

It keeps a territory it never leaves and never defends.
```

## How it works

Metadata is mapped to traits (size ladder, quant health, gut/build/eyes/etc.),
the trained sense classifier assigns visible senses to declared inputs, the
trained attribute classifier gives every unmapped field a best-match
attribute, and a seeded `Die` selects prose from fixed pools. Only the header
block quotes the file; the body is pure fiction. See `SPEC.md` for the full
specification.

## Testing

```bash
uv run pytest -q
```

Every functional requirement in `SPEC.md` is covered by a test (see
`tests/TRACEABILITY.md`).

## Project layout

```
gguf_zoo/            parser, rules, senses, classifier, attributes, taxonomy,
                     anomalies, render, cli, artifacts/
tests/               pytest suite + TRACEABILITY.md
pyproject.toml       uv project (deps + console script gguf-zoo)
.agents/SPEC.md      full specification
```

## Determinism & license

Same input always yields byte-identical output (no wall-clock, no salted
`hash()`). MIT licensed.
