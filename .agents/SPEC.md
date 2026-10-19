# SPEC: `gguf-zoo` — GGUF Metadata → Fictional Creature Field Guide

**Version:** 2.0
**Program name:** `gguf-zoo` (hyphenated). The importable Python package is
`gguf_zoo` (underscores — hyphens are illegal in module names), and a console
script `gguf-zoo` is installed so users type `gguf-zoo …`. See §0.5 and FR-10.
**Status:** Ready for implementation
**Language:** Python 3.11+
**Tooling:** [`uv`](https://docs.astral.sh/uv/) (environment, dependencies, lockfile, runner)
**Target:** Windows/macOS/Linux; GPU (CUDA/MPS) used for training **and**
inference when available, CPU fallback otherwise (FR-12)
**Runtime deps:** `numpy`, `torch` (training + inference of the sense classifier)
**Test deps:** `pytest`
**Deliverable:** a CLI tool + importable library + **two trained** classifiers
(input-type→sense, unmapped-field→attribute) + a pytest suite

---

## 0. Context for an implementer with none

You are building a tool that opens a `.gguf` file, reads **only its metadata header**
(never the tensors/weights), and emits a **deterministic** fictional creature
description — a naturalist's field-guide entry — derived procedurally from that
metadata. Exactly two machine-learned components exist: a **small text classifier
that maps the *name* of an input type (e.g. `"image"`, `"mel spectrogram"`,
`"imu"`) to a sense** (sight, hearing, touch, …), and a **second small text
classifier that maps an *unmapped metadata field key* to its best-match physical
attribute** (FR-16), so that no field is ever left unmapped and no raw metadata
string is ever echoed in the guide body. Everything else is deterministic,
rule-based procedural generation ("the biology").

**Hard rule (FR-3.4):** everything after the header block (the `=`-ruled block
naming the source model, parameters, and quantization) is pure fiction. No
string copied from the GGUF file — field keys, values, quant names, declared
input names — may appear in the body.

### 0.1 What a GGUF file is

GGUF ("GGML Universal Format", llama.cpp) stores a quantized LLM. Layout:

```
offset  size  field
0       4     magic bytes  "GGUF"
4       4     uint32       version (2 or 3; v1 has no n_kv field — reject)
8       8     uint64       n_tensors
16      8     uint64       n_kv
24      ...   n_kv key-value pairs
        ...   tensor info records   <-- read ONLY in --in-depth mode (FR-19)
        ...   tensor data           <-- sampled ONLY in --in-depth mode (FR-19)
```

All integers are **little-endian**. A KV pair is:

```
uint64 len, len bytes UTF-8   key
uint32 type
       value
```

Type codes:

| code | type | wire size |
|---|---|---|
| 0 | UINT8 | 1 |
| 1 | INT8 | 1 |
| 2 | UINT16 | 2 |
| 3 | INT16 | 2 |
| 4 | UINT32 | 4 |
| 5 | INT32 | 4 |
| 6 | FLOAT32 | 4 |
| 7 | BOOL | 1 |
| 8 | STRING | uint64 len + bytes |
| 9 | ARRAY | elem_type (uint32) + uint64 count + elements |
| 10 | UINT64 | 8 (v3 only) |
| 11 | INT64 | 8 (v3 only) |
| 12 | FLOAT64 | 8 (v3 only) |

Arrays may nest. Arrays of length > 64 are stored opaquely as
`"<array:{elem_type}:{count}>"` so a 128k-token vocabulary neither bloats memory
nor the output; the parser must still **skip the element bytes** on the wire so
subsequent KVs stay aligned.

### 0.2 What "quant" means here

`general.file_type` is a small integer naming the dominant quantization of the
weights. Fewer bits = more lossy compression. **Bit depth is the creature's
physical health**: an F32 model is a pristine animal; a 2-bit model is a walking
corpse. Sub-classes (`_K_*`, `_S/_M/_L`, `_0/_1`, `IQ*`) further shape the body.

### 0.3 What "parameters" means

Parameter count is **not stored** in GGUF. Estimate it from architecture fields:

```
n_params ≈ n_vocab * n_embd * 2
         + n_layers * (4 * n_embd**2 + 8 * n_embd**2)
```

(`4·d²` = attention QKVO projections; `8·d²` ≈ SwiGLU MLP with hidden ≈ 8/3·d
and 3 matrices.) For Mixture-of-Experts, scale the MLP term by
`expert_used_count / expert_count`. The estimate is approximate and the output
must label it as such (`~7.0B`).

### 0.4 Non-goals

- Do **not** read tensor info records or tensor data, **except** in
  `--in-depth` mode (FR-19), which samples a bounded, deterministic subset.
  The default path never touches them.
- Do **not** call the network or any external LLM at inference time; all prose is
  from built-in pools. Both classifiers are trained **locally** from built-in
  synthetic datasets.
- Do **not** produce non-deterministic output: same input ⇒ byte-identical
  output, forever (see FR-8).

### 0.5 Toolchain: Python + `uv`

The project is written in **Python (≥ 3.11)** and managed exclusively with
[`uv`](https://docs.astral.sh/uv/). No `pip`/`venv`/`conda` instructions; no
requirements.txt. `uv` owns the interpreter pin, the virtualenv, dependency
resolution, and the lockfile.

Project files:

- `pyproject.toml` — declares `[project]` (name `gguf-zoo`, `requires-python = ">=3.11"`,
  deps `numpy`, `torch`), a console script `[project.scripts] gguf-zoo = "gguf_zoo.cli:main"`,
  and an optional `[dependency-groups]`/dev group for `pytest`.
- `.python-version` — pins the interpreter (e.g. `3.11` or newer) so `uv` fetches
  it automatically.
- `uv.lock` — committed lockfile; reproducible installs.
- `.venv/` — the project-local virtual environment created by `uv sync`
  (git-ignored; never committed).
- `.gitignore` — must ignore `.venv/`, `__pycache__/`, `*.pyc`, and `.pytest_cache/`.

All commands go through `uv`. The installed program is invoked as `gguf-zoo`
(the console script); `uv run python -m gguf_zoo …` is the module-equivalent
form used interchangeably below:

```
uv sync                       # create .venv + install deps from the lockfile
uv run gguf-zoo infer --demo
uv run pytest -q
uv run gguf-zoo train-senses
```

Rules:

- **Proper uv virtual environment.** All work happens in a project-local `.venv`
  created and managed by `uv` (`uv sync` creates it; `uv run` uses/creates it
  automatically). The system/global interpreter must **not** be used to install
  or run the project. `uv sync --frozen` must reproduce the exact environment
  from `uv.lock`.
- The `.venv` is disposable and git-ignored; a fresh `uv sync` on a clean clone
  must yield a working environment with no manual `pip install`.
- `uv.lock` **must** be committed and updated (`uv lock`) whenever dependencies
  change; the spec assumes it exists.
- Bare `python`/`pytest` invocations are acceptable only *inside* the activated
  `uv` environment (`.venv`); documented commands use `uv run`.
- On machines without a CUDA GPU, drop the optional PyTorch CUDA index from
  `pyproject.toml` and re-`uv sync`; training **and** inference fall back to CPU
  automatically (FR-12). When a CUDA GPU is present, the CUDA torch build must be
  used so the GPU is actually available to `resolve_device`.

---

## 1. Deliverables

```
gguf_zoo/
  __init__.py          # re-exports public API
  parser.py            # GGUF header parser  -> (kv dict, header info) + ModelMeta
  rules.py             # deterministic metadata -> trait Profile ("the biology")
  anomalies.py         # unmapped fields + uncovered senses -> fictional anatomy
  senses.py            # sense taxonomy, input-type seed lexicon, dataset builder
  classifier.py        # small char/word n-gram network + rule fallback + artifacts
  attributes.py        # second trained model: unmapped field key -> attribute (FR-16)
  taxonomy.py          # Latin binomial from name + traits
  weights.py           # FR-19: in-depth tensor info + sampled weight statistics
  pools.py             # all prose pools (module constants)
  render.py            # deterministic field-guide renderer
  cli.py               # argparse entry point
  artifacts/
    sense_clf.pt       # trained checkpoint (produced by `train-senses`)
    field_attr_clf.pt  # trained checkpoint (produced by `train-attributes`)
gguf_zoo.py            # thin shim; primary entry is the `gguf-zoo` console script
tests/
  conftest.py          # synthetic GGUF byte-builder fixtures
  test_parser.py
  test_rules.py
  test_size.py
  test_quant.py
  test_senses.py
  test_classifier.py
  test_anomalies.py
  test_attributes.py   # FR-16: unmapped-field -> attribute classifier
  test_weights.py      # FR-19: in-depth weight sampling, dequant, quant-robustness
  test_taxonomy.py
  test_determinism.py
  test_render.py
  test_cli.py
  test_device.py         # FR-12: GPU-when-available device selection
  test_readme.py         # FR-13: README sections + example output is real
  test_pools.py          # FR-14: every prose pool has >= 20 varied entries
  test_traceability.py   # FR-11: every FR id maps to a passing test
  TRACEABILITY.md        # FR id -> test(s) map (FR-11.3)
pyproject.toml         # uv project definition (deps + pytest dev group)
.python-version        # interpreter pin for uv
uv.lock                # committed lockfile
.gitignore             # ignores .venv/, __pycache__/, *.pyc, .pytest_cache/
.venv/                 # uv-managed virtualenv (created by `uv sync`, not committed)
README.md              # user-facing docs (FR-13)
SPEC.md                # this file
```

### 1.1 Public API (`from gguf_zoo import ...`)

```python
read_metadata(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]
    # (kv_dict, {"version": int, "n_tensors": int, "n_kv": int, "kv_end": int})
build_meta(kv: dict) -> ModelMeta                    # normalized ModelMeta dataclass
extract_features(md: ModelMeta) -> "Features"         # raw + recognized fields
estimate_params(f: Features) -> int | None
size_tier(n_params: int | None) -> tuple[str, str]    # (tier name, range prose)
quant_info(f: Features) -> QuantInfo
classify_input_type(name: str, draw: int) -> SenseResult   # <- TRAINED classifier; stochastic
classify_input_type_consensus(name: str, seed: int) -> SenseResult
    # runs classify_input_type 3x with distinct draws until all 3 agree (FR-6.10)
detect_input_types(md: ModelMeta) -> list[str]        # declared modalities
binomial(profile: Profile, meta: ModelMeta) -> str
classify_attribute(key: str, seed: int = 0) -> str
    # <- SECOND TRAINED classifier: unmapped field key -> best-match attribute
    # (FR-16); runs 3x with distinct seeds until all 3 agree (FR-6.10 protocol)
train_attribute_classifier(seed: int = 0, device: str = "auto") -> Path
eval_attribute_classifier(device: str = "auto") -> dict
anomaly_organs(unknown: dict, extra_senses: list[tuple[str, str]] = ...) -> list[tuple[str, str]]
    # (anatomy prose, internal source id — never printed); also covers senses NOT
    # handled by the sense model (FR-3.7)
build_profile(md: ModelMeta) -> Profile
build_description(kv: dict, header: str = "", ..., in_depth: bool = False,
                  source: str | Path | bytes | None = None) -> str
    # in_depth=True adds the FR-19 NEUROLOGY section; `source` is the .gguf
    # path or bytes the kv came from (needed to reach the tensor section)
neural_profile(source, kv, header) -> "NeuralProfile | None"   # FR-19
render_prompt(kv: dict) -> str   # detailed image-gen prompt (FR-17)
```

### 1.2 Dataclasses

```python
@dataclass
class Features:
    arch: str | None
    name: str | None
    file_type: int | str | None
    n_layers: int | None
    n_embd: int | None
    n_vocab: int | None
    n_expert: int | None
    n_expert_used: int | None
    input_types: list[str]        # declared input type NAMES, ordered, de-duped
    unknown: dict[str, Any]       # every key not recognized

@dataclass
class QuantInfo:
    name: str            # "Q4_K_M"
    label: str           # "4-bit K-quant"
    band: int            # 0..10 health band (10 = pristine)
    modifiers: list[str] # deterministic additional-description prose, fixed order
    recognized: bool     # False if file_type missing/unparseable

@dataclass
class Profile:           # the full "biology" of one creature
    body_size: int       # 0..12 tier index
    condition: int       # 0..10 band
    gut: int
    build: int
    eye: int
    body_length: int
    diet_breadth: int
    integument: int
    palette: int
    habitat_tint: int
    fastness: int
    gradient_pattern: int
    marking_complexity: int
    headgear: int
    ear_size: int
    senses: list[tuple[str, str]]   # (input type name, visible sense id), in detection order
    # ... plus the boolean modifiers from rules.py
```

---

## 2. Functional requirements

Each requirement carries an ID used by the test suite. **All must pass.**

### FR-1 — GGUF binary parsing (`parser.py`)

1. **1.1** Accept a file whose first 4 bytes are `GGUF`; otherwise raise
   `GGUFError("Not a GGUF file (bad magic)")`.
2. **1.2** Support version 2 and 3. Raise `GGUFError` for version < 2.
3. **1.3** Decode every type code in §0.1, including nested arrays.
4. **1.4** Read **only** the KV section. Never seek past `n_kv` pairs. A file
   with 0 tensors and 0 KV pairs parses to `({}, {...})`.
5. **1.5** Truncated/short files raise `GGUFError` (wrap `struct.error`), never
   leak a raw `struct.error`.
6. **1.6** Undecodable UTF-8 uses `errors="replace"`, never raises.
7. **1.7** `header_info` == `{"version": int, "n_tensors": int, "n_kv": int}`.
8. **1.8** `read_metadata` returns the raw KV dict *and* a `ModelMeta` can be
   built from any flat `key -> value` mapping (used by `--meta` and `--demo`).

### FR-2 — Metadata → physical attributes

Basic physical attributes come from GGUF metadata via these mappings. Real GGUF
names architecture fields `<arch>.<field>` (e.g. `gemma4.embedding_length`,
`gemma4.block_count`); the legacy form `general.architecture.<arch>.<field>` is
also accepted. `block_count`/`block_length` and `vocab_size`/`vocab_length` are
aliases. All arch keys are matched by regex; the **last** match wins; missing
fields stay `None` and never raise. A key is **recognized** (never an anomaly)
iff it is a standard `general.*` key (`architecture`, `name`, `language`,
`author`, `base_model`, `basename`, `finetune`, `license`, `license_link`,
`quantized_by`, `size_label`, `type`, `quantization_version`, `sampling.*`), a
`tokenizer.*` key, an `<arch>.*` key, or the legacy arch form; anything else is
an unknown field (FR-3).

| attribute | source key(s) | rule |
|---|---|---|
| `body_size` | estimated `n_params` | FR-4 size ladder |
| `condition` | `general.file_type` | FR-5 quant health ladder |
| `gut` | `<arch>.context_length` | ≤4K tiny crop; ≤32K normal mammalian gut; ≤131072 ruminant chamber; >131072 whale gut |
| `build` | `<arch>.feed_forward_length` / `embedding_length` | ratio >3.2 thick bulk; 2.5–3.2 balanced; <2.5 wiry |
| `eye` | `<arch>.attention.head_count`, `head_count_kv` | ≥48 compound eyes; 16–47 fewer large eyes; `kv*4 ≤ heads` → GQA big eyes/cheap nerve; `kv==1` → cyclops |
| `body_length` | `<arch>.block_count` | <20 compact; 20–40; 40–70; >70 centipede-long neural path |
| `diet_breadth` | `<arch>.vocab_length` | 6-class ladder, §2.1 boundaries |
| `marking_complexity` | `<arch>.vocab_length` | same 6-class boundaries (kept identical) |
| `integument` | `has_ssm` / MoE / quant block style | priority: SSM→exoskeleton; MoE→quills; K-quant→fur; legacy→scales; lossless→bare |
| `palette` | `tokenizer.ggml.pre` + `tokenizer.ggml.model`; fallback `<arch>.vocab_length` bucket | tokenizer **lineage** drives color, never architecture: llama-bpe→desert tan; qwen→jade+vermilion; nova/mistral→mistral blue; gemma→pearl/blue; phi→violet; deepseek→ink+gold; falcon/rwkv/cohere→rust; unknown lineage→vocab bucket (6 classes over `VOCAB_DIET_BOUNDARIES`); no vocab→slate. Models sharing a tokenizer lineage share a color family across architectures |
| `habitat_tint` | `general.language`/`tokenizer` langs | arctic / multilingual / temperate / jade-ink / desert-ochre |
| `fastness` | `general.file_type` | Q8+ true color; Q4–Q6 washed; Q2/Q3 pale |
| `gradient_pattern` | `<arch>.rope.freq_base` | ≥100000 slow gradient; ≥10000 even; <10000 banded |
| `headgear` | `<arch>.rope.scaling.type`, `partial_rope` | none→hornless; linear→spikes; yarn→antler crown; dynamic→asymmetric rack; partial→single horn (override) |
| `ear_size` | `<arch>.attention.key_length` | ≥128 big tufted; 64–127 moderate; <64 sleek |
| `second_jaw` | any key containing `mtp` | additive |
| `has_baleen` | any key containing `pooling` | additive |
| `two_bodied` | `<arch>.encoder_decoder` | additive |
| `lipless` | `add_bos`/`add_eos` | both→lips; neither→lipless |

**2.1** Shared diet/marking boundaries
`VOCAB_DIET_BOUNDARIES = [16000, 48000, 100000, 200000, 400000]` (lower bound
inclusive, upper exclusive): `0` plain/obligate specialist … `5` extreme
ornate/all-eater. One constant so the two ladders cannot drift.

**2.2** MoE internal anatomy: `n_expert` present ⇒ hundreds of tiny hexagonal
digestive chambers, only a few active per meal ("termite-mound beast").

**2.3** Every attribute has a deterministic default when its source is missing;
nothing raises on a sparse file.

### FR-3 — Unknown metadata fields

1. **3.1** Any key not recognized becomes an entry in `Features.unknown`.
   *Recognized* = the §2 keys plus any key matching `^general\.file_type$`,
   `^tokenizer\.ggml\..+`, `^general\.architecture\..+`.
2. **3.2** Each unknown field renders as a **cryptid anomaly**: a prose organ
   of unknown function chosen deterministically from
   `sha256(f"{key}::{value!r}")` seeded RNG. Same key+value ⇒ same organ,
   independent of dict ordering and `PYTHONHASHSEED`.
   **Never use builtin `hash()`** (salted per process).
3. **3.3** Anomalies print under a `UNIQUE TRAITS` heading, sorted by key.
   Identical anatomy lines are printed **once** (two fields that classify to the
   same fictional attribute with the same description collapse to one line).
   The raw key and value are **never printed** — they are only the deterministic
   seed for the anatomy (FR-3.2) and the input to the attribute classifier
   (FR-16).
4. **3.4 — No raw metadata in the body.** Everything after the header block (the
   `=`-ruled block with source model / parameters / quantization) must contain
   **no string copied from the GGUF metadata**: no field keys, no values, no
   quant name (e.g. `Q4_K_M`), no declared input-type names (e.g. `Image`).
    There is **no `[AUDIT]` footer** — every field is accounted for by the
    deterministic mapper (FR-2) or the attribute classifier (FR-16), so nothing
    needs auditing in the output. Section headings are plain too: **no
    parenthetical mechanism annotations** (e.g. "SENSES (from declared input
    modalities)" or "UNIQUE TRAITS (fictional anatomy from unmapped
    metadata…)") — the body never explains how it was generated.
5. **3.5** Zero unknown fields **and** zero uncovered senses ⇒ the section does
   not print at all.
6. **3.6** Unknown values of any GGUF type (arrays incl. nested, bools, floats,
   huge 64-bit ints) render without error.
7. **3.7** The `UNIQUE TRAITS` section **also covers senses that the sense
   model cannot place** — i.e. senses with no externally visible organ (FR-6.1).
   These are the internal senses (`proprioception, interoception, nociception,
   chronoception, baroception, graviception, balance, magnetoreception`) and,
   above all, the **fictional senses** (`telepathy, empathy, precognition,
   retrocognition, clairvoyance, psychometry, aurapathy, manasense, dangersense,
   soulsight, forcesense, voidsense, deathsense, bloodsense, quantum_sense,
   probability_sense`). Each such sense is emitted as an anomaly line, deterministically
   (FR-3.2), tagged as a suspected organ, and biased toward the fictional pool:
   for every non-visible sense the organ description is drawn from the *fictional*
   anomaly pool. The source input name is **not** printed (FR-3.4); the line is
    pure invented anatomy, e.g.
    `• a knotted ganglion that flickers with a sight that has not happened yet.`
8. **3.8** When both unmapped metadata fields and unmapped senses exist, the
   metadata anomalies print first (sorted by key), then the sense anomalies
   (sorted by sense id).
9. **3.9 — Unknown anatomy IS fictional anatomy.** Every unknown-field anomaly and
   every uncovered-sense anomaly is rendered as a named piece of **fictional
   anatomy** — a fictional *organ* or a fictional *bodily structure* — rather
   than a generic "organ of unknown function". The unknown metadata field (or
   uncovered sense) is the *evidence* for that anatomy; the anatomy is the
    creature's invented explanation for it. The heading is the plain
    `UNIQUE TRAITS:` — no parenthetical explaining the mechanism (FR-3.4).
10. **3.10 — Naming.** Each fictional anatomy gets a deterministic name (seeded
    by `sha256(f"{key}::{value!r}")`, FR-3.2) of one of two forms:
    - **Pseudo-organ word** — any word that *sounds* like an organ or bodily
      structure (Latin/Greek-flavoured), e.g. `Flossum`, `Ventriculus`,
      `Ganglion`, `Plexus`, `Sinus`, `Bulbus`, `Hilum`, `Sulcus`, `Alveolus`,
      `Villus`, `Septum`, `Ligamentum`, `Aponeurosis`, `Chorda`, `Raphé`.
    - **Possessive person-name prefix** — a person's first or last name made
      possessive and prepended, e.g. `Felix's Flossum`, `Martha's Meningia`,
      `Gustav's Ganglion`, `Dr. Okonkwo's Sinus`. Names come from a first/last
      name pool; the possessive is formed with `'s`.
    The name is paired with a kind (FR-3.11).
11. **3.11 — Two kinds, two descriptions.** Each fictional anatomy is either an
    **organ** or a **bodily structure**, and the prose differs by kind:
    - **organ** → a description of *organ-like behaviour/function*: what it
      secretes, filters, pumps, pulses, ripens, molts, hums, times, etc.
      (e.g. "a pale gland that weeps on a rhythm no one has timed").
    - **bodily structure** → a description of its *structure and its movement*:
      shape, material, and how it moves or is moved (e.g. "a coiled cartilage
      strut that rotates a quarter-turn on each exhale").
     The kind is chosen deterministically from the same seed. Every description
     is prefixed with **where on the creature the anatomy sits** — a generic
     anatomical location phrase from the `ANATOMY_LOCATIONS` pool (e.g. "at the
     nape", "in the chest cavity"), drawn from the same seed.
 12. **3.12 — Line format.** Each anomaly line is
     `• <Anatomy Name> (<organ|structure>) — <location>, <behaviour | structure+movement>.`
     and **nothing else**: no key, no value, no input name (FR-3.4), and **no
     "Function unknown." suffix** — the invented description already speaks for
     itself, and a line must never both describe what an anatomy does and call
     its function unknown.
     The anatomy name is either a fictional attribute name attributed
     by the attribute classifier (FR-16.3) or, when the classifier abstains, a
     name from the FR-3.10 naming pools. Examples:
     - `• Fermentation Crop (organ) — beneath the jaw, a pale gland that weeps on
       a rhythm no one has timed.`
     - `• the Helix of Marrow (structure) — at the base of the tail, a coiled
       cartilage strut that rotates a quarter-turn on each exhale.`

### FR-4 — Size tier from parameter count

**4.1** Bands, evaluated in order, strict `<`:

| # | upper bound | tier name | range prose |
|---|---|---|---|
| 1 | 100e6 | Submicroscopic | virus to bacterium |
| 2 | 500e6 | Microscopic | microbe to midge |
| 3 | 1e9 | Minute | roundworm to gnat |
| 4 | 4e9 | Small | housefly to honeybee |
| 5 | 20e9 | Medium-small | scorpion to songbird |
| 6 | 100e9 | Medium | rat to hawk |
| 7 | 500e9 | Medium-large | cat to pig |
| 8 | 1e12 | Large | ape to horse |
| 9 | 5e12 | Megafauna | bison to elephant |
| 10 | 20e12 | Gigantic | Tyrannosaurus to sauropod |
| 11 | 100e12 | Titanic | titan |
| 12 | 500e12 | Kaiju | kaiju |
| — | ≥ 500e12 | Beyond Kaiju | a kaiju that dwarfs mountains |

**4.2** Parameter count resolution order: (a) if `general.size_label` is present
and parses as `<number>[K|M|B|T]` (case-insensitive, e.g. `12B`, `300m`, `1.5t`)
or a bare full number (`7155000000`), that declared total wins —
it is authoritative for MoE models, whose *active* param count the formula
under-reports; (b) otherwise the formula from `n_layers`, `n_embd`, `n_vocab`,
where `n_vocab` may also come from the element count of
`tokenizer.ggml.tokens` (list length, or the count in an opaque
`<array:t:n>` marker); (c) if neither is available `estimate_params` returns
`None`. `size_tier(None)` → `("Unclassifiable", "a size that resists measurement")`.

**4.3** MoE scaling: when `n_expert` and `n_expert_used` are both present and
`n_expert_used < n_expert`, scale the MLP term by `n_expert_used / n_expert`.

**4.4** Each tier owns a body-plan anchor list (reuse the curated lists: virion,
bacillus/tardigrade, rotifer/nematode, gnat/dipteran, bee/scorpion,
passerine/rat, raptor/cat, pig/ape, horse/bison, elephant/theropod,
sauropod/titan, leviathan/kaiju). The chosen anchor is drawn from the seeded RNG
(FR-8). Every anchor carries a Linnaean binomial used in the prompt
(real scientific names where they exist; mock-Latin otherwise).

**4.5** Output line: `Parameters   : ~7.0B  ->  size class: Small (housefly to honeybee)`.
Use `B` below 1e12, `T` at or above.

**4.6** The body text never names the size class (FR-3.4 spirit: the header
block carries the class). Instead the intro states the creature's **actual
size and weight**, log-log interpolated from the estimated parameter count
against an anchor ladder (bacterium ≈ 1 µm / 1 pg … kaiju ≈ 300 m / 50 Mt),
formatted adaptively (µm/cm/m; µg/mg/g/kg/tonnes/megatonnes):
`It measures about 2.5 m from nose to tail and weighs about 500.0 kg`.
The measure phrase is anatomy-appropriate for the tier anchor (FR-4.6):
`from nose to tail` (default), `from beak to tail` (birds), `from snout to
tail` (fish/whales/serpents), `from head to tail` (insects/arachnids),
`from head to rump` (frogs/newts), `from end to end` (worms/jellyfish/mites),
`across` (viruses/microbes).
When the param count is unclassifiable: `Its size and weight resist measurement`.

### FR-5 — Quant → physical condition

**5.1** Map `general.file_type` to `(name, band)` using the enum below (matches
llama.cpp `llm_ftype`; note there is no value 4). Bands: 10 pristine … 0 dust.

| value | name | band | | value | name | band |
|---|---|---|---|---|---|---|
| 0 | ALL_F32 | 10 | | 11 | Q3_K_S | 3 |
| 1 | F16 | 9 | | 12 | Q3_K_M | 3 |
| 2 | Q4_0 | 4 | | 13 | Q3_K_L | 4 |
| 3 | Q4_1 | 5 | | 14 | Q4_K_S | 5 |
| 5 | Q5_0 | 6 | | 15 | Q4_K_M | 6 |
| 6 | Q5_1 | 7 | | 16 | Q5_K_S | 7 |
| 7 | Q8_0 | 9 | | 17 | Q5_K_M | 8 |
| 8 | IQ2_XXS | 1 | | 18 | Q6_K | 8 |
| 9 | IQ2_XS | 1 | | 19 | Q8_1 | 9 |
| 10 | Q2_K | 2 | | 20–39 | IQ2_S … IQ4_XL | see monotonic rule |

Band → health word: `10 pristine, 9 excellent, 8 good, 7 fair, 6 weathered,
5 frail, 4 emaciated, 3 diseased, 2 near-corpse, 1 skeletal, 0 dust`.

**5.2** **Monotonicity invariant:** if quant A has strictly more bits than quant
B (same family), `band(A) >= band(B)`. Tests assert this over the whole table.

**5.3** Sub-class modifiers — applied to the description, in this fixed order,
for **every** pattern matching the quant name (multiple may apply). The prose
lives in `pools.QUANT_MODIFIER_POOLS` (FR-14: **≥ 20 phrasings per pool**);
entry 0 is the canonical phrasing, and the renderer draws one variant per
matching pattern with the seeded die:

| regex | pool key | canonical phrasing (entry 0) |
|---|---|---|
| `XXS` | `XXS` | bones like capillary tubes; it looks ready to snap |
| `XS` | `XS` | fine-boned and compact, every part reduced to its minimum |
| `_S$` | `S` | sleek and close-coated, with nothing to spare |
| `_M$` | `M` | mottled and asymmetric — the two sides never quite match |
| `_L$` | `L` | thick-set and heavy-coated, carrying extra everything |
| `K` | `K` | patchily mangled in blocky clumps, whole swaths of tissue replaced by coarse scar |
| `_0$` | `Q0` | blunt and undetailed; its features are all broad strokes |
| `_1$` | `Q1` | finely etched, every ridge and whisker crisply drawn |

Rendered as additional `• <prose>` description lines under the condition
section — a condition has no subclasses, only additional descriptions, and the
quant name never appears there (FR-3.4).

**5.4** Unrecognized / missing `file_type`:
- string value → derive band via `re.match(r"(?:IQ|Q)(\d+)", name)`:
  `band = clamp(bits*2 - 2, 0, 8)`, `recognized=False`.
- missing/unparseable → band `5`, `recognized=False`.
- When `recognized` is False the quant line ends with
  `  [UNKNOWN QUANT — condition estimated]`.

**5.5** Condition section format:

```
CONDITION — WEATHERED: the specimen is <phrase from health pool>.
  • patchily mangled in blocky clumps, ...
```

### FR-6 — Input types → senses  (TRAINED CLASSIFIER)

This is the only machine-learned component. It accepts the **name** of an input
type **plus a random number** and maps it to an **externally visible sense**,
resolved by running three times with three different random numbers until all
three agree (FR-6.10). The three mappings below are hard requirements and must
hold for 100% of inputs, model or not:

| input type | required sense |
|---|---|
| `image` (and synonyms: img, picture, photo, frame, …) | **sight** |
| `video` (and synonyms: clip, motion, footage, …) | **sight** |
| `sound` / `audio` (waveform, speech, mel spectrogram, …) | **hearing** |
| `text` (string, token, document, prompt, code, …) | **touch (feel)** |

All other input type names map to the nearest **externally visible** sense in
`VISIBLE_SENSES` (§6.1); anything the classifier cannot place as a visible sense
(including every internal and fictional sense) routes to the anomalous-anatomy
path (FR-3.7).

**6.1 — Sense taxonomy & visibility.** The full sense enum lives in `senses.py`,
but **the sense model determines externally visible senses only** — those a
field guide can point at (eyes, ears, whiskers, antennae, feelers, …). The
classifier's label space is therefore exactly `VISIBLE_SENSES`:

| visible sense | external organ(s) the model implies |
|---|---|
| `sight` | eyes / ocelli / eyespots |
| `hearing` | ears / tympana / frilled pinnae |
| `touch` | whiskers / feelers / manipulator pads |
| `taste` | tongue / mouthparts |
| `smell` | nostrils / olfactory antennae |
| `thermoception` | heat-pit organs / infrared pits |
| `electroreception` | ampullae / electroreceptor pores |
| `echolocation` | melon / nasal sonar structure |
| `vibration` | sensory hairs / feelers |
| `hygroception` | humidity-sensing hairs |
| `chemoreception` | chemoreceptive antennae |
| `polarization_vision` | compound eyes |
| `sonar` | melon |

Senses **without** an externally visible organ are **not** classifier outputs and
are handled by FR-3.7 instead:

- internal (real): `proprioception, interoception, nociception, chronoception,
  baroception (pressure), graviception, balance, magnetoreception`;
- speculative (fictional): `telepathy, empathy, precognition, retrocognition,
  clairvoyance, psychometry, aurapathy (life-force), manasense (magical energy),
  dangersense, soulsight, forcesense, voidsense, deathsense, bloodsense,
  quantum_sense, probability_sense`.

`sense_unknown` is the reserved fallback label; it never renders in the SENSES
section (it too falls through to FR-3.7 as "an organ of unknown function, warm
and faintly humming, that answers to no known sense"). Adding or removing a
`VISIBLE_SENSES` row changes `|senses|` and therefore requires retraining.

**6.2 — Dataset (built-in, deterministic).** `senses.py` defines a seed lexicon
mapping input-type aliases → **visible** sense, e.g.: `image/img/picture/photo/
frame/rgb/pixels/jpeg/png/screenshot/camera feed` → `sight`; `video/clip/footage/
motion/optical flow/24fps` → `sight`; `audio/sound/wav/waveform/speech/voice/
music/mel spectrogram/mfcc` → `hearing`; `text/string/token/document/transcript/
caption/prompt/code/markdown` → `touch`; `thermometer/thermal camera/infrared/
heat map/pit organ` → `thermoception`; `gas sensor/electronic nose/voc` →
`smell`; `taste sensor/electronic tongue/ph` → `taste`; `sonar/ultrasound
rangefinder/tof/depth sensor` → `echolocation`; `voltage/eeg/ecg/ampullae` →
`electroreception`; `geophone/seismometer/accelerometer` → `vibration`; and so on
across `VISIBLE_SENSES` only. Aliases whose target sense is internal or fictional
(e.g. `imu/gyroscope` → balance, `magnetometer/compass` → magnetoreception,
`brainwave/thought receiver` → telepathy) are **excluded from the label space**
and routed to the anomaly path (FR-3.7); they are never training labels. The
generator then augments deterministically (seeded RNG):

- modifiers prefixed/suffixed/appended: `raw, pre, post, encoded, tensor,
  stream, batch, 1d, 2d, multi, embedded, tokenized, serialized, input, data`;
- separators: `""`, `_`, `-`, ` `;
- vowel-stripped abbreviations;
- optional case jitter.

Target ≈ 8–20k samples. **Split by seed alias (grouped split)**: no alias'  variants
appear in two splits, so the test set measures generalization to *unseen*
input-type names. Class-balance the splits.

**6.3 — Model** (`classifier.py`). Small, CPU-trainable in seconds:
character-level. Vocabulary = printable ASCII + `_` + `-` + `<pad>` + `<unk>`.
Encode each name to ≤32 chars. `Embedding(V=96, d=8)` → concat of mean-pool and
max-pool over positions → `Linear(→128) → GELU → Dropout(0.1) → Linear(→|senses|)`.
< 100k parameters. Alternative acceptable: hashed char-2/3-gram bag → 2-layer MLP
(same size). Loss = cross-entropy; optimizer AdamW; ≤20 epochs; deterministic
seed (`_seed_all`); early-stopping on val accuracy. Training runs on the device
from `resolve_device("auto")` (FR-12): GPU (CUDA/MPS) when present, else CPU.

**Inference is stochastic and draw-driven.** `forward` takes a `draw: int`; it
seeds a local `random.Random(draw)` that (a) keeps dropout active at eval time
and (b) applies a small embedding jitter (≈0.02 std). The same input name with
two different draws may therefore yield different logits — this is what the
3-run consensus in FR-6.10 exploits. The *sequence* of draws comes from the
seeded `Die` (FR-8.1), so the finally accepted sense is still fully
deterministic (FR-8.5).

**6.4 — Required-mapping guarantee.** Before consulting the model, normalize the
name (lowercase; strip; collapse separators). If it is in the hard override
table (`image, image input, img, picture, photo, video, video input, clip,
sound, sound input, audio, audio input, text, text input, string, …`) return the
required sense with confidence `1.0`. This makes the invariant hold even if the
trained model regresses.

**6.5 — Inference.** `classify_input_type(name, draw) -> SenseResult(sense,
confidence, source)` where `source ∈ {"override","model","fallback"}` and `draw`
is the random number (FR-6.3). If the model's max softmax < `CONF_THRESHOLD`
(default 0.55) or the name is empty, fall back to: the hard override table, then
a keyword-rule matcher, then `sense_unknown`. A result of `sense_unknown`, or a
sense outside `VISIBLE_SENSES`, is **not** printed in the SENSES section; it is
passed to FR-3.7 as an explained-as-unexplained organ. Inference runs on the
device from `resolve_device("auto")` (FR-12): GPU when present, else CPU.

**6.6 — Artifact & training CLI.** `model.pt`-style checkpoint at
`gguf_zoo/artifacts/sense_clf.pt` containing `state_dict`, `label_list`,
`config`, and a `dataset_hash` (sha256 of the generated dataset). If the
checkpoint is missing at runtime, `classify_input_type` auto-trains on first use
(seeded) and caches it; `gguf-zoo train-senses` forces a (re)train;
`gguf-zoo eval-senses` reports test accuracy and a confusion matrix. Loading a
checkpoint whose `dataset_hash`/`label_list` disagrees with the code raises a
clear error rather than mis-loading.

**6.7 — Acceptance for the classifier:** held-out **alias-grouped** test accuracy
≥ 0.95 over `VISIBLE_SENSES`; the four required mappings (`image`, `video`,
`sound`/`audio`, `text`) at 100%; the 3-run consensus converges within
`MAX_CONSENSUS_ROUNDS` for every test name; and **no** internal or fictional
sense is ever emitted by the model (such inputs must resolve to a visible sense
or fall through to FR-3.7).

**6.8 — Detection of declared inputs.** `detect_input_types(meta)` scans the KV
dict for declared modalities and returns input-type *names*: `tokenizer.ggml.pre`
containing `vision`/`clip`; tokens containing `<image>`/`<image_start>`; a
`chat_template` containing `image`; tokens/template containing `<video>`/`video`
or `<audio>`/`audio`; presence of `tokenizer.ggml.eos_token_id` or
`vocab_length` ⇒ `text`; keys ending `.attention.key_length` ⇒ `embeddings`;
keys containing `tool`/`function` ⇒ `tools`. Dedup, preserve order, default
`["text"]`. Each name is then run through `classify_input_type_consensus`
(FR-6.10); non-visible/unknown results fall through to FR-3.7.

**6.9 — Sense prose.** Render under the plain heading `SENSES:` (no
parenthetical about the mechanism, FR-3.4).
The declared input names are **not** printed (FR-3.4); only the senses appear,
one line per distinct sense in first-detection order (duplicate senses are
collapsed):

```
  • {SenseName:<14} {prose}
```

Prose pools are tier-aware. Required prose (large tiers / small tiers
Submicroscopic·Microscopic·Minute):

| sense | large-tier prose | small-tier prose |
|---|---|---|
| sight | a cluster of multifaceted eyes like wet garnets | a light-sensitive eyespot, its only window on the world |
| hearing | vast frilled ears, rimmed with blood-hot cartilage | cilia that taste vibration before sound has a name |
| touch | velvety manipulator pads, whorled like fingerprints, that read the world by touch | whisker-analogues that braille every surface it passes |
| any other visible sense | an organ of unknown function, warm and faintly humming, that answers to no known sense | *(same)* |

(Other senses get their own templated organ phrases from `pools.py`; every sense
in `VISIBLE_SENSES` must have non-empty prose.)

**6.10 — Three-run consensus (required, system-wide).** This is the classification
protocol for **every** model invocation in the system — the sense classifier here
and the attribute classifier (FR-16.4). For **every** declared input, the
model is run **three times with three different random seeds**; the input only
resolves to a sense once all three runs return the **same** sense. If the three
disagree, all three are run again with three fresh random seeds, and this repeats
until they agree (cap at
`MAX_CONSENSUS_ROUNDS = 50`, after which the majority/first visible sense is
taken and a `[UNSETTLED SENSE]` note is emitted). Each declared input gets its
own fresh set of random numbers.

```python
def classify_input_type_consensus(name, seed):
    rng = random.Random(seed)          # seed comes from the Die (FR-8.1)
    for _ in range(MAX_CONSENSUS_ROUNDS):
        draws = [rng.getrandbits(64) for _ in range(3)]   # 3 different numbers
        results = [classify_input_type(name, d) for d in draws]
        if results[0].sense == results[1].sense == results[2].sense:
            return results[0]          # all three agree -> accept
    return _majority(results)          # capped fallback + [UNSETTLED SENSE]
```

All draws derive from the single seeded `Die`, so consensus output is
reproducible byte-for-byte (FR-8.5). Tests must assert that a mock model which
disagrees on the first round still converges on the second, and that a
model fed three start-up draws returns the unanimously agreed sense.

### FR-7 — Latin binomial from the model name + traits

**7.1** **Genus** derives from the model's `general.name`, Latinized and fused
with a size-band animal root:

- split `general.name` on non-alphanumerics; transliterate to ASCII
  (Cyrillic/Greek/punctuation table + NFKD fold); lowercase; drop noise tokens
  `{gguf, base, instruct, chat, it, ml, model, beta, lite, turbo, pro, plus,
  finetune, merged, abliterated, safetensors}` and pure numbers (`7b`, `3.1`).
- stem = first surviving token, capitalized, fused with a band suffix:
  `micro→ -(i)um, insect→ -apis, small→ -iola, mammal→ -ferus, mega→ -odon,
  colossal→ -titan` (gender m/f/n accordingly).
- if the name is missing / all-noise, use the stem `Ignot` and still succeed.

**7.2** **Species epithet(s)** come from the trait `Profile` (appearance, size,
features). At most two, chosen by fixed priority
`integument > condition > headgear > eye > gut > diet_breadth > palette`,
inflected to the genus gender. Examples: fur→`pilosus/a/um`,
scales→`squamatus/a/um`, quills→`setosus/a/um`, exoskeleton→`loricatus/a/um`,
great-gut→`magaventer`, cyclops→`cyclops`, horned→`cornutus/a/um`,
all-eater→`omnivorus/a/um`, plus palette adjectives (tan→`fuscus`,
jade→`viridis`, blue→`caeruleus`, pearl→`perlaceus`, violet→`violaceus`,
gold→`aureus`, rust→`rubigineus`), and
size adjectives for the tier. Skip rules: a "worn/weathered" condition epithet is
skipped when integument already fired; `hornless` fires only if nothing else did;
the slate (unknown-lineage) palette default never fires; if nothing fires, use
`ignotus/a/um`.

**7.3** **Authority** = Latinized `general.author` (capitalized → `-ia`;
lowercase stays), else the architecture name lowercased; finetunes
(`base_model != name`) append `& <basestem>`. **Year** = file mtime year when
known; never fabricated.

**7.4** Full name anatomy: `Genus epithet [epithet2] Authority[, Year]`.
Worked examples: 1B qwen fur → `Qwenapis pilosa qwen2`; 70B llama scales →
`Llamaferus squamatus llama`; 8x7B MoE → `Favus setosus mixtral`;
120B Q2 titan → `Llamatitan corruptus llama`.

**7.5** A one-clause glossary follows the binomial in the intro, e.g.
`Qwenapis pilosa — a hairy insect of the forest floor`. The closing phrase is
drawn from a per-habitat pool (`HABITAT_PHRASES`, ≥ 20 each, FR-14) that
matches the creature's habitat (FR-18.1), so the gloss names where the
creature lives rather than a fixed tagline.

**7.6** Header line: `FIELD GUIDE ENTRY — Qwenapis pilosa qwen2 (the Sturdy)`,
where the parenthetical common epithet derives from the health band
(`the Unblemished`, … `the Plagued`).

### FR-8 — Determinism

1. **8.1** One `Die` class wraps `random.Random(seed)`; seed =
   `f"{name}|{arch}|{quant_name}"`. No unseeded `random.*` anywhere.
2. **8.2** `build_description(kv)` twice on the same input ⇒ identical strings.
3. **8.3** Output must not depend on `PYTHONHASHSEED` (test runs a subprocess
   with a different seed and diffs stdout).
4. **8.4** No timestamps/`id()`/dict-iteration-order dependence. Unknown fields
   are sorted before rendering. Prose picks use a salted hash stream
   (`sha256(f"{SALT}|{meta_hash}|{trait}:{class}")`) so pools rotate per trait
   without co-occurring.
5. **8.5** The classifier is fully seeded: same dataset + seed ⇒ identical
   checkpoint. Individual draws are stochastic (FR-6.3), but the draw *sequence*
   comes from the seeded `Die`, so `classify_input_type_consensus(name, seed)`
   is deterministic — same `name` + `seed` ⇒ same accepted sense, forever.

### FR-9 — Output document

Fixed section order. The header block (between the `=` rules) is the **only**
place GGUF-derived strings may appear; everything below it is pure fiction
(FR-3.4):

```markdown
---
source_model: "<name> (<arch>)"
parameters: "~7.0B  ->  size class: Small (housefly to honeybee)"
quantization: "Q4_K_M (4-bit K-quant)  ->  condition: weathered"
---

# FIELD GUIDE ENTRY — <binomial> (<epithet>)

<intro paragraph: tier anchor, actual size+weight (FR-4.6), body plan,
          integument, color, posture, habitat+features, appended anatomy>

## SENSES

- **Sight** — a cluster of multifaceted eyes like wet garnets

## CONDITION — WEATHERED

The specimen is <phrase>.

- patchily mangled in blocky clumps, ...

## UNIQUE TRAITS

- Fermentation Crop (organ) — a pale gland that weeps on a rhythm no one has
  timed.
- the Helix of Marrow (structure) — a coiled cartilage strut that rotates a
  quarter-turn on each exhale.

<one behavior sentence>

## IMAGE PROMPT

<single-line image prompt>
```

**9.1** The output is **Markdown**: YAML front matter (double-quoted values)
holding `source_model`, `parameters`, and `quantization`, followed by the H1
`# FIELD GUIDE ENTRY — <binomial> (<epithet>)`. Sections are H2 headings and
items are `- ` list bullets. The front matter is the only place GGUF-derived
strings may appear; everything after the H1 is pure fiction (FR-3.4).
**9.2** Exactly one behavior sentence from a fixed pool, chosen by the seeded RNG.

### FR-10 — CLI

The program is `gguf-zoo`, run through `uv` (see §0.5); `python -m gguf_zoo …`
is the module-equivalent form:

```
uv run gguf-zoo infer <file.gguf>
uv run gguf-zoo infer --meta metadata.json
uv run gguf-zoo infer --demo [NAME] [--save out.gguf]
uv run gguf-zoo infer --demo [--demo-name NAME] [--demo-params N] [--demo-quant Q]
uv run gguf-zoo infer <file.gguf> --params 7155000000   # override estimate
uv run gguf-zoo infer <file.gguf> --in-depth            # read sampled weights (FR-19)
uv run gguf-zoo infer <file.gguf> --image-url URL --image-model M \
    [--image-key KEY] [--image-size WxH] [--save-image PATH] [--save-description PATH]
uv run gguf-zoo train-senses [--out artifacts/sense_clf.pt] [--seed 0] [--device auto]
uv run gguf-zoo eval-senses [--device auto]
uv run gguf-zoo train-attributes [--out artifacts/field_attr_clf.pt] [--seed 0] [--device auto]
uv run gguf-zoo eval-attributes [--device auto]
uv run gguf-zoo senses --name "mel spectrogram" [--seed 0] [--draws 3] [--device auto]
uv run gguf-zoo detect-senses <file.gguf>     # consensus per declared input
```

`--device auto|cuda|mps|cpu` (default `auto`) selects the accelerator per FR-12.

1. **10.1** `infer` requires exactly one of `<file>` / `--meta` / `--demo`; else
   exit 2.
2. **10.2** `--meta` accepts a flat JSON object of `key -> value`.
3. **10.3** `--demo [NAME]` synthesizes a valid GGUF **in memory** and parses it
   through the real reader (not a shortcut). Built-in demos: `qwen2` (default,
   a multimodal Q4_K_M model with unknown fields) and `mixtral` (a pristine
   F32 MoE model); bare `--demo` equals `--demo qwen2`.
4. **10.4** `--save PATH` writes the demo bytes to `PATH`.
5. **10.5** Missing file → stderr message + exit code 1, never a traceback.
6. **10.6** Description goes to stdout; diagnostics to stderr.
7. **10.7** `--in-depth` (FR-19) is accepted by `infer`; with `--meta` (no
   tensor data available) it is an error → exit 2.
8. **10.8** Demo override flags (only meaningful with `--demo`):
   `--demo-name NAME` replaces the demo's `general.name`;
   `--demo-params N` sets the parameter count — a full number (`7155000000`)
   or a **case-insensitive** `k`/`m`/`b`/`t` suffix (`300m`, `7B`, `1.5t`),
   applied via `general.size_label` (FR-4.2);
   `--demo-quant Q` sets `general.file_type` from a **case-insensitive**
   quant name; the accepted values are the canonical `QUANT_TABLE` names
   (plus `F32`) and are listed in `--help`. An invalid `--demo-params` or
   `--demo-quant` value → stderr message + exit 2.
9. **10.9** `infer` **ignores** `--demo-name` / `--demo-params` /
   `--demo-quant` entirely when `--demo` is not present.

### FR-11 — Per-requirement tests (mandatory for completion)

**No functional requirement is considered complete until it has at least one
passing test that exercises it.** This is a hard, structural requirement of the
spec, not a best-effort guideline.

1. **11.1** Every FR and every sub-requirement ID (FR-1.1 … FR-20.6) must be
   covered by at least one named test. A requirement without a test is **not
   done**, even if the code appears to work.
2. **11.2** Tests are named after the requirement they verify, e.g.
   `test_FR_6_10_three_run_consensus`, `test_FR_5_2_quant_monotonicity`,
   `test_FR_3_7_fictional_sense_anomaly`. The FR id must appear in the test
   name or in a `@pytest.mark.req("FR-6.10")` marker so it is greppable.
3. **11.3** A **traceability table** (`tests/TRACEABILITY.md`, generated or
   hand-maintained) maps every FR id → the test(s) that cover it. CI fails if
   any FR id in this SPEC lacks a mapped, passing test.
4. **11.4** Each test must assert the *behavior* the requirement states (its
   numbers, ordering, formats, invariants), not merely that a function ran
   without raising. Happy-path-only coverage of a requirement that specifies
   edge cases (e.g. FR-3.6 "any GGUF type", FR-4.2 `None`) is insufficient.
5. **11.5** Negative/edge requirements get negative tests: bad magic (FR-1.1),
   truncated file (FR-1.5), unknown quant (FR-5.4), unknown metadata of every
   type (FR-3.6), non-visible/fictional senses (FR-3.7), consensus disagreement
   (FR-6.10), and exit codes 0/1/2 (FR-10).
6. **11.6** `uv run pytest -q` must exit 0 with **zero** skipped/xfail tests
   standing in for an unimplemented requirement. A requirement may only be
   marked `xfail` with an inline `# FR-x.y: <reason>` and must be resolved
   before the spec is signed off.

**Traceability map (FR id → test file; each id needs ≥1 passing test):**

| FR | covered by |
|---|---|
| FR-1.1–1.8 | `test_parser.py` |
| FR-2.1–2.3 | `test_rules.py` |
| FR-3.1–3.8 | `test_anomalies.py` |
| FR-4.1–4.5 | `test_size.py` |
| FR-5.1–5.5 | `test_quant.py` |
| FR-6.1–6.10 | `test_senses.py`, `test_classifier.py` |
| FR-7.1–7.6 | `test_taxonomy.py` |
| FR-8.1–8.5 | `test_determinism.py` |
| FR-9.1–9.2 | `test_render.py` |
| FR-10.1–10.9 | `test_cli.py` |
| FR-11.1–11.6 | `test_traceability.py` (asserts the map is complete & green) |
| FR-12.1–12.5 | `test_device.py` |
| FR-13.1–13.13 | `test_readme.py` |
| FR-14.1–14.5 | `test_pools.py` |
| FR-16.1–16.5 | `test_attributes.py` |
| FR-17.1–17.3 | `test_prompt.py` |
| FR-18.1–18.4 | `test_consistency.py` |
| FR-19.1–19.6 | `test_weights.py` |
| FR-20.1–20.6 | `test_imagegen.py` |

### FR-12 — GPU acceleration (training and inference)

The sense classifier **must use a GPU whenever one is available**, for both
training and inference, and fall back to CPU transparently when none is present.

1. **12.1** A single `resolve_device(request: str) -> torch.device` is the only
   place device selection happens. `request="auto"` (the default) picks the first
   available of `cuda` → `mps` → `cpu`. An explicit `cuda`/`mps` that is
   unavailable raises a clear `RuntimeError` (never a silent CPU fallback);
   `cpu` forces CPU.
2. **12.2** Training (`train-senses`) runs on the resolved device: model and the
   train/val tensors are moved to the device once (full-batch residency for the
   small dataset); mixed precision (`torch.autocast` + `GradScaler`) is enabled
   on CUDA and skipped on MPS/CPU. The chosen device and (for CUDA) the device
   name are printed to stderr at start.
3. **12.3** Inference (`classify_input_type`, and therefore the consensus loop)
   runs on the resolved device by default. The checkpoint is saved with CPU
   tensors (`map_location="cpu"` on load) so a GPU-trained model loads on any
   device; tensors are moved to the active device at predict time.
4. **12.4** Determinism is preserved on GPU: `torch.manual_seed` +
   `torch.cuda.manual_seed_all` are set from the same seed (FR-8.5), and the
   draw-driven jitter (FR-6.3) is generated on CPU before moving to the device,
   so the consensus result is reproducible on a given machine. Cross-machine
   byte-identical output is guaranteed only on CPU (documented caveat).
5. **12.5** Every CLI command that touches the model accepts
   `--device auto|cuda|mps|cpu` (default `auto`); `--device` is validated through
   `resolve_device`.

### FR-13 — README.md

The deliverable **must** include a top-level `README.md` that documents the
program for a first-time user. It is a required artifact: the spec is not
complete without it. Required sections, in this order:

1. **13.1 Title + description** — program name and a one-paragraph plain-language
   summary: reads a GGUF file's metadata and procedurally generates a fictional
   creature field-guide entry (size from parameter count, condition from quant,
   senses from input modalities via a trained classifier, Latin binomial).
2. **13.2 Features** — bullet list of what it does (GGUF header-only parsing,
   unknown-field anomalies, size ladder, quant health, trained sense classifier
   with 3-run consensus, GPU acceleration, deterministic output).
3. **13.3 Requirements** — Python ≥ 3.11, `uv`, `numpy`/`torch`; note that a GPU
   (CUDA/MPS) is used when present and CPU otherwise (FR-12).
4. **13.4 Setup** — the exact `uv` commands (`uv sync`, optional CUDA-index note),
   matching §0.5. No pip/venv/conda.
5. **13.5 Quickstart** — a copy-pasteable `uv run gguf-zoo infer --demo`
   that works on a fresh clone with no model file.
6. **13.6 Usage** — a CLI reference covering every subcommand and flag from FR-10
   (`infer`, `train-senses`, `eval-senses`, `senses`, `detect-senses`, `--meta`,
   `--demo`, `--save`, `--params`, `--seed`, `--draws`, `--device`), each with a
   one-line description.
 7. **13.7 Training the classifiers** — how to (re)train and evaluate both
    trained models (`train-senses`/`eval-senses`, `train-attributes`/
    `eval-attributes`) and where the checkpoints live.
8. **13.8 Example input → output** — at least one concrete example: a sample
   input (the `--demo` metadata or a small `--meta` JSON) and the **exact**
   field-guide output it produces, in a fenced block. Because output is
   deterministic (FR-8), the shown output must be the program's real output, not
   hand-written prose.
9. **13.9 How it works** — short prose pointing at the size ladder, quant health
   table, sense classifier, and naming rules (may link to this SPEC).
10. **13.10 Testing** — `uv run pytest -q` and a note that every FR is covered
    (FR-11).
11. **13.11 Project layout** — the directory tree from §1.
12. **13.12 Determinism & license** — the reproducibility guarantee and a license
    line.

**13.13** `test_readme.py` asserts the README exists, contains every required
section heading (13.1–13.12), and that the fenced example output in §13.8 is
**byte-identical** to the program's actual output for the documented input
(`infer --demo` or the pinned `--meta` JSON). A stale example fails the suite.

### FR-14 — Prose pools (word lists)

All prose comes from fixed pools in `pools.py` (no LLM, no network). To make
output varied and rich rather than repetitive:

1. **14.1** Every list of terms or phrases in `pools.py` — every attribute pool
   a `Die`/hash-stream draws from, **and every shared word bank** (adjective and
   clause banks, health nouns and per-band health adjectives, tier life-stage
   modifiers, per-sense organ nouns) — contains **no fewer than 20 entries**.
   More is better; there is no upper limit. This applies to (at least): health
   phrases (per band), tier anchors (per tier), integument, palette, habitat
   tint, fastness, gradient pattern, marking complexity, eyes, gut, build, body
   length, diet breadth, headgear, ear size, behavior sentences, metadata
   anomaly organs, fictional-anatomy names, fictional-anatomy organ-behaviour
   descriptions, fictional-anatomy structure+movement descriptions, person
    names, the per-sense organ nouns and phrase pools, the per-habitat gloss
    phrases (`HABITAT_PHRASES`), the quant sub-class modifier pools
    (`QUANT_MODIFIER_POOLS`), and the in-depth neural section phrase pools
    (`NEURAL_*`). Every entry must be relevant to the pool it
   sits in, and each pool should span **both realistic and fictional** imagery
   (real anatomy and behaviour alongside invented,
   impossible, or marvellous ones).
2. **14.2** A wider range of descriptions is strongly preferred. Some entries
   may be near-synonyms or share a theme, but the pool should span varied
   imagery, register, and phrasing rather than 20 near-copies of one idea.
3. **14.3** Entries are lowercase drop-in fragments (except proper names in the
   person-name pool), with no trailing punctuation, and must not hardcode a
   different attribute's meaning (no cross-attribute keyword leakage). Entries
   must also be internally coherent: no self-contradictory or physiologically
   incoherent claims (e.g. a life-stage that contradicts "full-grown", one side
   of a body being "better fed" than the other, plumage wording in a pool that
   also serves scale- or fur-covered creatures).
4. **14.4** `test_pools.py` asserts every pool has ≥ 20 entries (FR-14.1), that
   entries are unique and non-empty (FR-14.2/14.3), and that no *descriptor*
   entry ends in sentence punctuation. The behavior-sentence pool is exempt from
   the punctuation rule (its entries are complete sentences).
5. **14.5** `pools.py` holds **all selection values**: any set of
   interchangeable phrases or options from which one or more entries is chosen
   for output must be defined in `pools.py` and obey FR-14.1–14.4 — no other
   module may define prose pools (quant sub-class modifiers live in
   `QUANT_MODIFIER_POOLS`, in-depth bucket phrasings in the `NEURAL_*` pools).
   Deterministic lookup tables that map a key to a single canonical value
   (size ladder, health words, habitat names, measure phrases, Latin epithets,
   bucket thresholds) are **not** selection pools and stay with the logic that
   uses them. `test_pools.py` **auto-discovers** every public list-of-strings
   constant (and dict/tuple-of-lists) in `pools.py` and runs the FR-14 checks
   on it, so a newly added pool cannot skip the rules.

### FR-15 — Habitat, locomotion & plant-like features

1. **15.1** Every creature is assigned exactly one of three **habitats** —
   `terrestrial`, `aerial`, `aquatic` — with **equal representation** (each
   chosen with probability 1/3 from the seeded `Die`).
2. **15.2** Each tier's anchor pool contains an **equal number** of terrestrial,
   aerial and aquatic anchors (no tier may be biased toward one habitat).
3. **15.3** Habitat drives locomotion/appendage features, drawn from a pool per
   habitat (each ≥ 20, FR-14):
   - **terrestrial**: legs, hooves, paws, burrowing claws, digitigrade stance,
     padded feet, strong hind legs, spurs, etc.
   - **aerial**: feathers, leathery wings, skin flaps, gliding membranes,
     broad wings, flapping motion, hollow bones, keeled sternum, tail fan, etc.
   - **aquatic**: fins, gas bladders, flukes, gills, streamlined body, webbed
     limbs, lateral line, blowhole, countershading, etc.
4. **15.4** A deterministic subset of creatures (≈ one third) additionally have
   **plant-like features**, drawn from a plant pool (≥ 20): photosynthesis,
   petals, leaves, bark (the outer covering of a tree), chlorophyll-green
   skin, fronds, thorns, sap, growth rings, a flower crown, vine tendrils, etc.
5. **15.5** The field guide renders the habitat and its features (and any
   plant-like features) in the intro paragraph.
 6. **15.6** `test_habitat.py` asserts: the three habitats are equally likely
    (assignment over many seeds is balanced), every tier anchor pool has equal
    terrestrial/aerial/aquatic counts (FR-15.2), the habitat and plant feature
    pools each have ≥ 20 entries (FR-15.3/15.4), and a rendered entry names a
    habitat.

### FR-16 — Unmapped field → attribute classifier (SECOND TRAINED MODEL)

No metadata field is ever left unmapped and no raw field key/value is ever
printed (FR-3.4). Recognized keys are mapped deterministically by FR-2; every
**unrecognized** key is mapped by a second small trained classifier to its
**best-match physical attribute**.

1. **16.1 — Coverage.** For every key in `Features.unknown`, `classify_attribute(key)`
   returns an attribute label. The guide body never echoes the key or its value;
   there is no `[AUDIT]` footer.
2. **16.2 — Label space.** `REAL_ATTRIBUTES` — one label per FR-2 trait family
   (`size, condition, gut, build, eyes, body_length, diet, integument, palette,
   fastness, gradient, headgear, ears, modality, identity`) — plus
   **`FICTIONAL_ATTRIBUTES`**, at least **8 fictional physical attributes**
   (e.g. `spiracle_array`, `fermentation_crop`, `resonance_bulla`, `nuchal_sail`,
   `tide_ganglion`, `marrow_lattice`, `shadow_membrane`, `gizzard_stones`), each
   with a display name and a kind (`organ` or `structure`). Fictional attributes
   exist so unknown fields land on invented anatomy instead of colliding with
   the real attributes already derived from recognized keys. A reserved
   `attr_unknown` label is the abstain fallback (never a training label).
  3. **16.3 — Rendering.** A field classified to a **fictional** attribute renders
     `• <Display Name> (<kind>) — <location>, <prose from that kind's pool>.` per FR-3.12
   (prose seeded by `sha256(f"{key}::{value!r}")`, FR-3.2). A field classified
   to a **real** attribute adds nothing — that attribute is already rendered
   from its canonical source. A field classified `attr_unknown` renders named
   fictional anatomy from the FR-3.10 naming pools, still without any citation.
4. **16.4 — Model & CLI.** Same architecture family as the sense classifier
   (hashed char n-gram bag → 2-layer MLP, FR-6.3), trained on a built-in
   synthetic dataset of key aliases (per-attribute teacher lexicon + morpheme
   synthesis, grouped split, FR-6.2 method). Inference follows the **three-run
   consensus protocol of FR-6.10**: every `classify_attribute` call runs the
   model three times with three different random seeds; the label is accepted
   only when all three runs agree, otherwise all three are re-run with fresh
   seeds (same cap and fallback as FR-6.10). Confidence below `CONF_THRESHOLD`
   abstains to `attr_unknown`. Checkpoint at `gguf_zoo/artifacts/field_attr_clf.pt`
   (override with `GGUF_ZOO_ATTR_ARTIFACT`), auto-trained on first use;
   `gguf-zoo train-attributes` / `gguf-zoo eval-attributes` (FR-12 device rules
   apply).
5. **16.5 — Acceptance.** Alias-grouped held-out accuracy ≥ 0.95; the label
space contains ≥ 8 fictional attributes disjoint from the real ones;
    `classify_attribute` is deterministic for a fixed seed (FR-6.10 seeds derive
    from a seeded RNG); and a rendered guide body contains no
    raw key, value, quant name, or input name from the input metadata.

### FR-17 — Image-generator prompt

1. **17.1** The rendered output ends with an `IMAGE PROMPT:` section: a single
    detailed prompt on **one unwrapped line** that covers **every detail of the
    description** — tier anchor + gloss, actual size and weight with the
    anatomy-appropriate measure phrase (FR-4.6), integument, palette, habitat
    tint, fastness, gradient, markings, headgear, ears, eyes, body length,
    build, gut, diet, habitat and locomotion/plant features, appended anatomy
    (MoE chambers, second jaw, baleen, two bodies, lipless), condition with its
    additional descriptions (FR-5.3), visible senses with their organ prose,
    unique traits, and the behavior sentence.
2. **17.2** The prompt closes with fixed wildlife-illustration style tags
    (field-guide plate, watercolor and ink, diffused daylight, plain
    background, full body in profile) so image generators render a natural
    history illustration rather than a photo or fantasy splash.
3. **17.3** The prompt is drawn from the same seeded draws as the description
    (identical wording, no extra randomness), is deterministic (FR-8), and
    obeys FR-3.4 (no raw keys, values, quant names, input names, or mechanism
    annotations). `render_prompt(kv)` returns the same prompt standalone.

### FR-18 — Anchor-consistent anatomy

1. **18.1** Every tier anchor creature is tagged with a **clade** (microbe,
   worm, bug, amphibian, reptile, bird, mammal, fish, mythical) and its
   habitat is fixed by its position in the tier table (equal thirds preserved,
   FR-15.2). The habitat is **derived from the drawn anchor**, never drawn
   independently.
2. **18.2** Entries in clade-sensitive pools (integument, ears, headgear, eyes,
   gut, habitat features) carry a set of compatible clades (`_RESTRICT`);
   unlisted entries fit every clade. The renderer draws the anchor first and
   then samples every clade-sensitive pool through `pool_for(pool, clade)`,
   so a tadpole never gets antlers, fur, or a patagium, and a bird never gets
   tufted mammal ears.
3. **18.3** `mythical` anchors (kaiju, titans, wyrms) accept every entry —
   invented biology is the point. If filtering would empty a pool, the full
   pool is used as fallback.
4. **18.4** `test_consistency.py` asserts every anchor is tagged, every tag is
   a valid clade, clade-restricted entries are absent from the other clades'
   filtered pools, and filtered pools are never empty.

### FR-19 — In-depth neural reading (`--in-depth`)

1. **19.1** With `--in-depth`, `infer` parses the **tensor info records**
   (name, dims, GGML dtype, offset) and samples actual weight values from the
   tensor data section of the same file (or the demo's synthetic tensors).
   The KV-only default path is unchanged.
2. **19.2** Sampling is bounded and deterministic: for each supported tensor,
   up to 64 quantization blocks chosen at **evenly spaced** positions (no RNG).
   Supported dtypes: F32, F16, BF16, Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K;
   tensors of any other dtype (IQ*, unknown) are skipped.
3. **19.3** Per-tensor statistics from the dequantized samples: RMS, mean,
   kurtosis, and sparsity (fraction of |values| < ¼·RMS). Aggregated across
   weight tensors (norm tensors excluded): mean log₁₀ RMS (**spread**),
   std of log₁₀ RMS (**heterogeneity**), mean kurtosis (**outlier index**),
   mean sparsity, and the least-squares slope of log₁₀ RMS vs. block index
   (**layer trend**).
4. **19.4** Each aggregate is bucketed into coarse fixed bins (low/mid/high)
   and rendered as **separate in-depth sections** in the style of a species
   biology manual: `APPEARANCE VARIATION` (color range, pattern template
   variety, seasonal molt), `TEMPERAMENT` (boldness, activity cycle,
   individual personality), `SPECIES BEHAVIOR` (migration, courtship, calls,
   sociality), `NATURAL HABITAT` (home range, shelter), and `ORGAN VARIATION`
   — how the creature's anomalous/fictional organs and senses vary between
   individuals and shape roles (e.g. larger telepathy organs serve as the
   group's scouts). No raw "nervous system" statistics are printed. Every
   bucketed selection draws from a `NEURAL_*` pool of **21 entries** (7
   phrasings per bucket, FR-14) with a die seeded on model name + architecture
   only — never on quant — so the same model at any quant level picks
   identical phrasings. The IMAGE PROMPT gains matching range/temperament/
   behavior segments.
5. **19.5** **Quant robustness:** RMS, kurtosis and sparsity are approximately
   invariant to quantization noise, and the coarse bucketing absorbs the
   residual differences, so different quant levels of the same model yield
   **very similar** descriptions. `test_weights.py` asserts identical facet
   buckets for identical synthetic weights stored as F32, Q8_0 and Q4_0.
6. **19.6** All in-depth output is deterministic: same file → byte-identical
   description (evenly spaced sampling, fixed thresholds, no RNG).

### FR-20 — Remote image generation (OpenAI-compatible)

1. **20.1** `infer` accepts `--image-url URL`, `--image-key KEY`,
   `--image-model MODEL`, `--image-size WxH` (default `1024x1024`),
   `--save-image PATH`, and `--save-description PATH`.
2. **20.2** Image generation is requested when any of these flags is present.
   Then `--image-url` (must start with `http://` or `https://`) and
   `--image-model` are required; otherwise → stderr message + exit 2.
   `--image-key` falls back to `$OPENAI_API_KEY` and may be empty for keyless
   local servers.
3. **20.3** The IMAGE PROMPT line is POSTed to the OpenAI-compatible
   `…/images/generations` endpoint (the path is appended to the URL as needed)
   as `{model, prompt, n:1, size, response_format:"b64_json"}`. The response
   image is taken from `data[0].b64_json` (saved as PNG) or downloaded from
   `data[0].url`. Any transport/API failure → stderr message + exit 1, never
   a traceback.
4. **20.4** Default output names use the sanitized model name
   (non-filesystem characters → `_`): `<model_name>_species_image.<ext>` and
   `<model_name>_species_description.md`; `--save-image` /
   `--save-description` override the respective path.
5. **20.5** The description file contains the full field-guide entry exactly
   as printed to stdout.
6. **20.6** This is the **only** permitted network access at inference time
   (amends FR-0.4 / acceptance §4) and happens solely when the flags are
   given. `test_imagegen.py` monkeypatches the transport — the test suite
   never touches the network.

---

## 3. Test fixtures: synthetic GGUF builder

Put in `tests/conftest.py`. Tests must not depend on any real model file.

```python
import struct
GGUF_MAGIC = b"GGUF"

def enc_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b

def enc_value(typecode: int, value) -> bytes:
    fmt = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
    if typecode == 8:
        return enc_str(value)
    if typecode == 9:
        elem, items = value
        out = struct.pack("<I", elem) + struct.pack("<Q", len(items))
        return out + b"".join(enc_value(elem, it) for it in items)
    return struct.pack(fmt[typecode], value)

def enc_kv(key: str, typecode: int, value) -> bytes:
    return enc_str(key) + struct.pack("<I", typecode) + enc_value(typecode, value)

def build_gguf(pairs, version=3, n_tensors=0) -> bytes:
    body = b"".join(enc_kv(k, t, v) for k, t, v in pairs)
    return (GGUF_MAGIC + struct.pack("<I", version)
            + struct.pack("<Q", n_tensors) + struct.pack("<Q", len(pairs)) + body)
```

Fixtures should expose at least: a minimal valid file, an empty file, a
truncated file, a file with nested arrays, a file with unknown keys of every
type, and a "qwen2-1.5b-like" file (151936 vocab, image inputs).

---

## 4. Sense taxonomy (label space)

`Sense` is an enum. The sense **model predicts only externally visible senses**
(`VISIBLE_SENSES`, FR-6.1); changing membership changes `|senses|` and requires
retraining. All other senses never reach the SENSES section and are described by
the anomalous-anatomy path (FR-3.7).

**Visible — model output (`VISIBLE_SENSES`):**

| id | sense | implied organ |
|---|---|---|
| sight | vision / photoreception | eyes / ocelli / eyespots |
| hearing | audition | ears / tympana |
| touch | somatosensation / tactition | whiskers / feelers / pads |
| taste | gustation | tongue / mouthparts |
| smell | olfaction | nostrils / olfactory antennae |
| thermoception | temperature / infrared | heat pits |
| electroreception | electric fields | ampullae |
| echolocation | sonar / ultrasound | melon |
| vibration | seismic / vibration | sensory hairs |
| hygroception | humidity | humidity hairs |
| chemoreception | general chemical detection | chemoreceptive antennae |
| polarization_vision | polarized-light sight | compound eyes |
| sonar | active acoustic ranging | melon |

**Non-visible — described as anomalies only (FR-3.7):**

| kind | ids |
|---|---|
| internal (real) | proprioception, interoception, nociception, chronoception, baroception, graviception, balance, magnetoreception |
| fictional | telepathy, empathy, precognition, retrocognition, clairvoyance, psychometry, aurapathy, manasense, dangersense, soulsight, forcesense, voidsense, deathsense, bloodsense, quantum_sense, probability_sense, sense_unknown |

**Required mappings** (FR-6): `image`,`video` → `sight`; `sound`,`audio` →
`hearing`; `text` → `touch`.

---

## 5. Acceptance criteria (definition of done)

1. `uv sync` completes cleanly and `uv run gguf-zoo infer --demo` prints
   a complete field-guide entry and exits 0.
2. `uv run pytest -q` passes in full; in particular:
   - `test_classifier.py` asserts alias-grouped accuracy ≥ 0.95 over
     `VISIBLE_SENSES`, the four required mappings at 100%, that
     `classify_input_type_consensus` requires three agreeing draws (mock model
     that disagrees on round one converges on round two), that the same seed
     reproduces results, and that no internal/fictional sense is ever emitted
     by the model;
   - `test_size.py` asserts every §4.1 boundary (100M…500T) and the
     `Unclassifiable` case;
   - `test_quant.py` asserts the §5.1 table, the §5.2 monotonicity invariant,
     the §5.3 modifier ordering, and the §5.4 unknown-quant fallback;
   - `test_anomalies.py` asserts §FR-3 for every GGUF value type and for
     sortedness; that senses not covered by the sense model (internal and,
     especially, fictional) render under `UNIQUE TRAITS` while visible
     senses do not; and that each anomaly is named fictional anatomy (FR-3.10)
     tagged `organ` or `structure` (FR-3.11) with the matching behaviour vs
     structure+movement description and the possessive person-name form
     (e.g. `Felix's Flossum`) appearing for some entries (FR-3.12).
   - `test_determinism.py` asserts identical output across two calls and across
     a subprocess with a different `PYTHONHASHSEED`;
   - `test_cli.py` asserts exit codes 0/1/2 per FR-10.
3. `uv run gguf-zoo eval-senses` prints per-sense accuracy and a
   confusion matrix and reports "all targets met".
4. Managed entirely by `uv` in a project-local `.venv` (`uv sync` creates it,
   `uv sync --frozen` reproduces it from the committed `uv.lock`; interpreter
   pinned via `.python-version`; `.venv/` git-ignored). No system/global
    interpreter use, no manual `pip install`. No network access at inference
    except the explicit `--image-*` remote generation (FR-20), no
    reading of tensor data except under `--in-depth` (FR-19), and no
    third-party runtime deps beyond `numpy` and `torch`.
5. **FR-11 holds:** every FR id (FR-1.1 … FR-20.6) has at least one passing,
   behavior-asserting test, `tests/TRACEABILITY.md` maps each id to its test(s),
   and `test_traceability.py` fails if any id is unmapped, skipped, or xfailed.
   A requirement without a green test is **not complete**.
6. **FR-12 holds:** `test_device.py` asserts `resolve_device("auto")` returns
   CUDA → MPS → CPU in that order of availability, that an unavailable explicit
   device raises, and that the checkpoint round-trips through CPU tensors. On a
   GPU host, `train-senses` and inference must report using the GPU.
7. **FR-13 holds:** `README.md` exists with all required sections (13.1–13.12),
   and `test_readme.py` confirms its example output is byte-identical to the
   program's real output for the documented input.
 8. **FR-14 holds:** `test_pools.py` asserts every prose pool has ≥ 20 unique,
    non-empty entries with no trailing sentence punctuation (FR-14.1–14.4), and
    auto-discovers every selection pool in `pools.py` (FR-14.5).
 9. **FR-15 holds:** `test_habitat.py` asserts equal-thirds habitat assignment,
    equal terrestrial/aerial/aquatic anchor counts per tier, ≥ 20-entry habitat
    and plant feature pools, and that a rendered entry names a habitat.
10. **FR-16 holds:** `test_attributes.py` asserts the attribute classifier's
    alias-grouped accuracy ≥ 0.95, ≥ 8 fictional attributes disjoint from the
    real ones, deterministic inference, and — end to end — that the rendered
     guide body contains no raw field key, value, quant name, or input name from
     the input metadata (FR-3.4), and that no `[AUDIT]` footer is printed.
11. **FR-19 holds:** `test_weights.py` asserts tensor-info parsing, dequant
    round-trips, deterministic sampling, and that F32/Q8_0/Q4_0 copies of the
    same weights produce identical facet buckets; `test_cli.py` covers
    `--in-depth` (including the `--meta` + `--in-depth` exit-2 case).
