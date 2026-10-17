# gguf-zoo

`gguf-zoo` parses GGUF model files (header + tensor info only — weights are
never read) and uses a small trainable neural network to convert the metadata
into a natural-language **creature field description** plus a ready-to-use
**image-generation prompt**.

The creature profile has two trait layers:

1. **Structural anatomy** — body size, condition, gut, build, senses, body
   length, diet.
2. **Cosmetic traits** — integument/skin type, coloration, headgear, marking
   complexity.

The student network is trained on synthetic data labeled by a deterministic
rule-based **teacher** (the "biology" in `rules.py`). After training,
inference is done by the network; the rules define the biology and the model
learns it. Training automatically uses a GPU (CUDA or Apple MPS) when
available and falls back to CPU transparently.

## Layout

```
gguf_zoo/
  parser.py   GGUF v2/v3 header + tensor-info parser (no external libs)
  rules.py    SIZE_ANCHORS + deterministic teacher rules (both layers)
  data.py     synthetic data generation (pure NumPy)
  model.py    feature encoding + student MLP (get_device lives here)
  train.py    training loop + evaluate (AMP, cosine LR, best-ckpt)
  pools.py    per-class synonym pools + salted hash-stream selection
  render.py   deterministic template renderer (field guide + prompt)
  cli.py      argparse CLI
gguf_zoo.py     entry point
pyproject.toml  uv project (deps + PyTorch cu128 index)
uv.lock         locked dependency versions
.python-version pinned Python (3.14)
tests/          11 unit-test groups (23 test methods)
```

Dependencies: `numpy` and `torch` only.

## Setup (uv)

The project uses [uv](https://docs.astral.sh/uv/) for environment management.
`pyproject.toml` pins the CUDA torch build (cu128) via the PyTorch index;
`uv.lock` and `.python-version` (3.14) make the environment reproducible:

```
uv sync              # creates .venv and installs numpy + torch (cu128)
uv run python gguf_zoo.py --help
```

Or activate the venv and use plain `python`:

```
.venv\Scripts\activate   # Windows (source .venv/bin/activate on Unix)
python gguf_zoo.py --help
```

On machines without a CUDA GPU, edit the torch source in `pyproject.toml`
(remove the `pytorch-cu128` index) and `uv sync` again to get the CPU wheel;
everything else is unchanged (training falls back to CPU automatically).

## Usage

```
python gguf_zoo.py gen-data [--n 200000] [--out data/] [--seed 42]
python gguf_zoo.py train    [--data data/] [--epochs 60] [--batch-size 4096]
                            [--device auto|cuda|mps|cpu] [--ckpt model.pt] [--seed 42]
python gguf_zoo.py infer --gguf path/to/model.gguf [--ckpt model.pt] [--device cpu]
                         [--strict-names]
python gguf_zoo.py infer --synthetic '{"n_params": 1e9, "quant": "Q4_K_M", "context": 4096, "vision": true, "architecture": "qwen2", "vocab": 151936}'
python gguf_zoo.py evaluate [--data data/] [--ckpt model.pt] [--device auto]
```

If `--ckpt` is missing, `infer` warns and falls back to the teacher rules.
`make_demo_gguf.py` writes `examples/qwen2-1.5b-demo.gguf`, a realistic
synthetic GGUF (full tensor table, 151,936-token vocab) for demos.

## GPU / CPU behavior

- `get_device()` picks CUDA → MPS → CPU; `--device` overrides with a clear
  error if the requested device is unavailable. The chosen device and GPU name
  are printed at train start.
- With ≤200K records the entire train/val tensors are moved to the GPU once
  before the epoch loop (full-batch residency).
- Mixed precision (`torch.autocast` + `GradScaler`) on CUDA only; skipped on
  MPS/CPU.
- Defaults: batch 4096 / 60 epochs on GPU, batch 256 / 30 epochs on CPU.
- Optimizer AdamW, cosine LR with 2-epoch warmup, `--seed` seeds numpy and
  torch (incl. `cuda.manual_seed_all`).
- Best-val checkpoint is saved with CPU tensors (`map_location="cpu"` on
  load) so it loads on any device.
- `infer` defaults to CPU (the model is small enough that CPU inference is
  instant); `--device` overrides.

Measured on an RTX 3090: `gen-data` (200K) ≈ 45 s, `train` 60 epochs ≈ 65 s
(target < 3 min), `evaluate` ≈ 2 s.

## The size ladder (Layer 1, `body_size`)

| n_params | class | register |
|---|---|---|
| < 100M | virus-bacteria | real |
| < 500M | microbe-midge | real |
| < 1B | worm-gnat | real |
| < 5B | housefly-honeybee | real |
| < 20B | scorpion-songbird | real |
| < 100B | rat-hawk | real |
| < 500B | cat-pig | real |
| < 1T | ape-horse | real |
| < 5T | bison-elephant | real |
| < 20T | tyrannosaurus-sauropod | real (extinct) |
| < 100T | titan | mixed real + fictional |
| ≥ 100T | kaiju | fiction only |

### Anchor lists

The renderer picks one anchor per output via a hash of the metadata
(deterministic per file, varied across files). Classes 0–9 use only
real/extinct animals; **titan** mixes the largest known animals with
same-scale fictional creatures; **kaiju** uses ONLY fictional kaiju.

| Class | Anchors |
|---|---|
| virus-bacteria | influenza virion, E. coli, amoeba, paramecium, yeast cell, rotifer, tardigrade, nematode, dust mite, springtail, copepod, brine shrimp |
| microbe-midge | fairyfly, thrips, aphid, flea, feather louse, whale louse, seed shrimp, fungus gnat, mosquito, blackfly, small wasp, fruit fly |
| worm-gnat | earthworm, mealworm larva, caterpillar, garden slug, snail, woodlouse, bed bug, silverfish, earwig, small spider, harvestman, small cricket |
| housefly-honeybee | housefly, honeybee, bumblebee, paper wasp, ladybird, ground beetle, cockroach, cricket, grasshopper, dragonfly, monarch butterfly, zebrafish, spring peeper frog, anole lizard |
| scorpion-songbird | scorpion, tarantula, hermit crab, shore crab, goldfish, killifish, tree frog, gecko, wren, finch, sparrow, blackbird, vole, shrew, pygmy rabbit, hatchling turtle |
| rat-hawk | brown rat, black rat, guinea pig, squirrel, chipmunk, ferret, weasel, hedgehog, opossum, pigeon, crow, magpie, kestrel, merlin, marmoset |
| cat-pig | domestic cat, beagle, border collie, fox, raccoon, badger, beaver, capybara, wallaby, bobcat, ocelot, caracal, macaque, baboon, pot-bellied pig, goat |
| ape-horse | human, chimpanzee, bonobo, gorilla, orangutan, grey wolf, cougar, red deer, elk, donkey, zebra, horse, steer, black bear, Labrador retriever |
| bison-elephant | American bison, white rhino, black rhino, Indian rhino, hippo, giraffe, polar bear, walrus, manatee, beluga, camel, moose, musk ox, Asian elephant |
| tyrannosaurus-sauropod | Tyrannosaurus, Spinosaurus, Allosaurus, Triceratops, Stegosaurus, Ankylosaurus, Diplodocus, Camarasaurus, woolly mammoth, woolly rhino, Paraceratherium, giant ground sloth |
| **titan** (mixed) | Argentinosaurus, Patagotitan, Dreadnoughtus, Supersaurus, Alamosaurus, blue whale, fin whale, Perucetus colossus, megalodon, Liopleurodon, **King Kong (MonsterVerse)**, **Titanoboa maximus (fictional giant-serpent register)**, **How to Train Your Dragon sea Leviathan**, **White Worm of Winterfell** |
| **kaiju** (fiction only) | **Godzilla, King Ghidorah, Mothra, Rodan, Mechagodzilla, Gigan, Biollante, SpaceGodzilla, Destoroyah, Shin Godzilla** (Toho); **Knifehead, Raiju, Slattern, Otachi, The Elder** (Pacific Rim); **Sachiel, Zeruel, Armisael** (Evangelion); **Baltan, Zetton** (Ultraman); **Deep Sea King** (One Punch Man); **Colossal Titan** (Attack on Titan) |

## Layer 1 — structural anatomy rules

| Feature | Trait | Rule |
|---|---|---|
| n_params | body_size | the 12-class size ladder above |
| quant | condition | health ladder: F16/F32 ideally muscled; Q8 robust and well-muscled; Q5/Q6 showing signs of malnourishment; Q4 thin, with ribs showing; Q3 gaunt and hollow-flanked; Q2 emaciated |
| context_length | gut_size | ≤4K tiny crop; ≤32K normal mammalian gut; ≤128K ruminant fermentation chamber; >128K whale gut |
| d_ff / d_model | build | >3.2 thick-bodied bulk digestor; 2.5–3.2 balanced; <2.5 wiry/streamlined |
| n_heads, n_kv_heads | eyes | ≥48 compound eye; 16–47 fewer large eyes; n_kv ≪ n_heads big eyes with cheap optic nerve (GQA); n_kv==1 (MQA) cyclops (override) |
| n_layers | body length | <20 tadpole; 20–40 compact; 40–70 deep-bodied; >70 centipede-long neural path |
| modalities | sensory organs | vision = large compound eyes/complex retina; audio = tympanic membranes/antennae; text-only = blind, deaf, vibration-sensing subterranean mammal |
| is_moe | internal anatomy | hundreds of tiny hexagonal digestive chambers, only a few active per meal ("termite-mound beast") |
| vocab_size | diet_breadth | 6-class ladder — see "Diet breadth & marking complexity" below |
| sliding_window/hybrid | gut chambers | mixed fast/slow digestion chambers |

## Layer 2 — cosmetic traits rules

**Integument** — exactly one class, priority top-to-bottom:

| Feature | Skin trait |
|---|---|
| has_ssm_layers | **Chitinous exoskeleton plating** — segmented state-carrying plates (always wins) |
| is_moe | **Spiny/porcupine quill coverage** — hundreds of discrete spines, few raised at once |
| K-quant | **Fur** — layered woven coat (two-level block structure) |
| legacy per-tensor | **Scales** — uniform independent overlapping plates |
| lossless (F16/F32/BF16) | **Smooth bare skin** |
| condition ∈ {Q2,Q3} | overlay: **patchy/mangy version** of the chosen cosmetic skin (structural skins keep their class + patchy modifier) |
| mixed_dtypes | overlay: **variegated integument** — fur on torso, scales on limbs |

**Coloration:**

| Feature | Trait | Rule |
|---|---|---|
| architecture | genus palette | llama = desert tan/sand; qwen = jade green + vermilion; mistral = pale mistral blue; gemma = pearl and blue; phi = deep violet; deepseek = ink black + gold; falcon = rust; command = amber; gpt/other/unknown = slate grey |
| language | habitat tint | cold langs (ru,fi,no,sv,is) = arctic white; >40 langs/multi = tropical vivid; en-only = temperate; zh/ja/ko = jade-and-ink; ar/he = desert ochre; default temperate green |
| quant | color fastness | Q8/F16 colors true to life; Q4/Q5/Q6 colors washed by the sun; Q2/Q3 colors pale and washed-out |
| rope_freq_base | gradient direction | ≥100000 slow gradient dark head → pale tail; ≥10000 even; <10000 banded/striped |
| vocab_size | marking complexity | 6-class ladder — see "Diet breadth & marking complexity" below |

**Headgear & face:**

| Feature | Trait |
|---|---|
| rope_scaling yarn / dynamic / linear / none | full antler crown / asymmetric rack / simple unbranched spikes / hornless |
| rope_freq_base magnitude | horn spiral tightness (high = long lazy curls, low = tight ram's spirals) |
| partial_rope | **single horn / one-sided rack** (overrides headgear class) |
| n_kv_heads == 1 (MQA) | **cyclops** — one great central eye (overrides eye class) |
| head_dim | ≥128 big tufted ears; 64–127 moderate; <64 sleek-headed |
| mtp_tensors | second jaw / double mandible (additive) |
| n_special_tokens | whiskers / sensory bristles, capped at 8 (additive) |
| add_bos/add_eos | defined lips/beak tip if both; lipless mouth if neither (additive) |
| pooling_type present | baleen plates, filter-feeder face (additive) |
| is_encoder_decoder | two-bodied creature, thorax and abdomen distinct (additive) |

**Render-only modifiers:** high n_tensors/n_params ratio = lean and ropy, low
= smooth and fat; `n_split_files > 1` = dotted-line "reconstructed" outline
(specimen shipped in crates); `quant_version == 1` = sepia/aged-paper
taxidermy rendering; `rms_epsilon < 1e-5` = vestigial dewclaws;
`byte_fallback` = visible gizzard stones in the throat.

## Diet breadth & marking complexity (v5)

Both traits share one roughly-doubling boundary constant
(`VOCAB_DIET_BOUNDARIES = [16000, 48000, 100000, 200000, 400000]`; lower bound
inclusive, upper exclusive), so the two ladders can never drift apart. Note
16384 falls in the *second* class, not the first.

**diet_breadth** (Layer 1):

| vocab_size | class | metaphor |
|---|---|---|
| < 16K | 0 obligate specialist | koala/panda: one food plant on one mountain; anything unusual unrecognizable, ground down by byte-fallback gizzard stones |
| 16K–48K | 1 picky browser | nectar specialist (hummingbird/koala grade): a few favored flowers, hovers close to home |
| 48K–100K | 2 generalist forager | raccoon/crow: familiar foods in the usual order, competent scavenger of the known |
| 100K–200K | 3 broad-spectrum omnivore | bear/hog: nearly everything edible is eaten, including things other animals refuse |
| 200K–400K | 4 bulk browser | giraffe/elephant: browses the whole canopy, enormous variety per meal, indiscriminate |
| ≥ 400K | 5 all-eater / baleen strainer | baleen whale or pig of the deep: strains everything from the water — every script, every language, no refusal |

**marking_complexity** (Layer 2), same boundaries:

| vocab_size | class | rendering |
|---|---|---|
| < 16K | 0 plain coat | no markings |
| 16K–48K | 1 faint markings | faint spots/stripes |
| 48K–100K | 2 modest markings | modest patterning |
| 100K–200K | 3 bold patterning | bold patterning |
| 200K–400K | 4 intricate plumage | intricate, every-feather-different plumage |
| ≥ 400K | 5 extreme ornate display | bird-of-paradise overload |

**Design note.** Large vocabularies exist *because* models cover many scripts,
so "broad diet" is a real biological reading of a real engineering decision —
the synthetic generator deliberately correlates vocab size with language
count (multilingual models draw from the high-vocab band). The old 3-tier
ladder couldn't distinguish a 128K model from a 256K model; the new one can.

## Synonym pools (`pools.py`)

Every trait class maps to a pool of phrasings (`TRAIT_POOLS`), and the
renderer fills fixed sentence templates from those pools via **salted hash
streams**:

```python
stream = sha256(f"{POOL_SALT}|{meta_hash}|trait:{trait}:{class_idx}")
idx    = int.from_bytes(stream[:8], "big") % len(pool)
```

Properties:

- **Deterministic** — same GGUF → same `meta_hash` → same phrasing every run;
  the README examples reproduce exactly.
- **Varied** — different files of the same class spread across the pool.
- **Decorrelated** — the trait name salts the stream, so `condition`'s pick
  is independent of `build`'s; no scripted co-occurrence.
- **Versionable** — edit pools freely, then bump `POOL_SALT` ("v1" → "v2") to
  start a fresh determinism epoch.

Sentence *templates* stay fixed (the field-guide voice); only the descriptor
slots rotate, so output stays grammatical: "a sun-washed, bold-patterning
coat" and "colors sharp and true, with ghosted spots" come from the same
class without ever producing ungrammatical prose.

**Pool hygiene** (enforced by unit tests): every pool has ≥3 entries; entries
are lowercase drop-in fragments with no trailing punctuation; no entry may
hardcode a different trait's meaning (cross-trait keyword check); and **no
preservation vocabulary** — these are living creatures, not museum specimens,
so words like *pristine / immaculate / weathered / worn / faded / bleached /
museum / taxidermy* are banned from every pool (the condition ladder is a
health ladder: *ideally muscled → emaciated*).

The pools are pure presentation-layer entropy downstream of the network:
training, evaluation, and teacher/student agreement are untouched.

## Nomenclature (`taxonomy.py`)

Every creature gets a deterministic, grammatically valid New Latin binomial:

```
Genus epithet [epithet2] Authority[, Year]
```

These are playful coinages for a speculative zoo, not real taxonomy.

**Genus** = Latinized architecture stem x size-band animal root, from the
curated `GENUS_FUSIONS` table (9 stems x 6 bands, 1-3 candidates per cell,
picked by the salted hash stream `taxonomy-v1`). Each genus carries a
grammatical gender:

| body_size classes | band | gender | llama stem example |
|---|---|---|---|
| 1-2 | micro | neuter (-um) | Llamium |
| 3-4 | insect | feminine (-apis) | Llamaapis |
| 5-6 | small fauna | feminine (-iola) | Llamaiola |
| 7-8 | mammal | masculine (-ferus) | Llamaferus |
| 9-10 | mega | masculine (-odon) | Llamaferodon |
| 11-12 | colossal | masculine (-titan) | Llamatitan |

Stems: `llama`->Llama, `qwen`->Qwen, `mistral`->Mistral, `gemma`->Gemma,
`phi`->Phi, `deepseek`->Profund, any MoE/`mixtral`->Fav (honeycomb),
`mamba`/SSM->Lorica (segmented armor), unknown->Ignot.

**Epithets** are trait-derived Latin adjectives, at most two, chosen by the
fixed priority `integument > condition > headgear > eye > gut >
diet_breadth > palette`. Skip rules: condition "worn" (Q4-class) is skipped
when integument fired; `hornless` fires only if nothing else did; palette
defaults (mistral blue, pearl, rust) never fire; if nothing fires at all the
fallback is `ignotus/ignota/ignotum` ("the unknown one"). **Every epithet is
inflected to the genus gender** (`pilosus/pilosa/pilosum`), invariant forms
(`parviventris`, `cyclops`) included.

**Authority**: Latinized `general.author` (capitalized names gain -ia:
Microsoft -> Microsoftia; lowercase stays: bartowski), else the architecture
name lowercase. Finetunes (`base_model` != `name`) append `& basegenus`.
**Year**: file mtime year for real files; omitted, never fabricated.

**Collisions**: two models sharing a binomial is legal - real taxonomy
distinguishes homonyms by authority and year. `infer --strict-names` keeps a
`names.json` registry and forces uniqueness: walk down the epithet candidate
list, then append a hash-derived trinomen (`minoris`/`mediocris`/`majoris`).
Registry entries are sorted by meta hash, so strict output is deterministic
for a given input set.

Worked examples:

| creature | name | why |
|---|---|---|
| 1B qwen, fur, jade | `Qwenapis pilosa viridis qwen2` | insect-band feminine genus, -a/-is agreement |
| 7B llama, fur, immaculate | `Llamaferus pilosus immaculatus llama` | mammal-band masculine, -us agreement |
| 8x7B mixtral (MoE) | `Favus setosus mixtral` | honeycomb genus, quill integument |
| 120B titan-class | `Llamatitan corruptus llama` | colossal band, Q2-corrupted condition |

The field description opens with the italic binomial plus a one-clause gloss
(*Llamaferus immaculatus* - an immaculate, hairy mammal of the written
word...), and diorama/plate/engraving image prompts get a specimen-plaque
token (macro photography and kaiju film stills skip it).

## Model

Input: a 28-dim normalized feature vector — the 26 features from the spec
(log10 n_params, log2 context, n_layers, log2 d_model, d_ff/d_model,
log2 vocab, quant ordinal, n_heads, kv-ratio, head_dim, is_moe, vision,
audio, sliding-window, block-style ordinal, mixed_dtypes, has_ssm,
log10 rope base, rope-scaling ordinal, partial_rope, mtp, capped
n_special_tokens, add_bos, add_eos, pooling, enc-dec) **plus** `arch_id` and
`lang_group` ordinals, which the palette and habitat-tint heads require.
Values are clipped to [-6, 6] so out-of-distribution metadata can't drive
wild extrapolation.

Output: 23 heads (11 Layer-1, 12 Layer-2), cross-entropy per head, summed.
`diet_breadth` and `marking_complexity` are 6-class heads (v5); loading a v4
checkpoint (3-class) raises a clear error instead of mis-loading.

Shared trunk: 6 hidden layers 1024→768→640→512→512→384, GELU, LayerNorm,
dropout 0.1 (~3.3M params total). Each head is a small MLP over
`[trunk_output, raw_features]` (skip connection). `body_size` — a sharp 1-D
step function of log10(n_params) — gets a dedicated deep head on that single
feature so shared-trunk smoothing doesn't blur the size-ladder boundaries
(notably titan/kaiju at 1e14).

## Teacher/student agreement

`evaluate` on the 20K-record test split (200K train, seed 42, 60 epochs):

```
body_size 99.2%  condition 100%  gut 100%  build 99.6%  eye 99.8%
body_length 99.2%  diet_breadth 100%  has_vision/has_audio/is_moe/is_subterranean 100%
integument 100%  palette 100%  habitat_tint 100%  fastness 100%
gradient_pattern 100%  marking_complexity 100%  headgear 100%  ear_size 100%
second_jaw/has_baleen/two_bodied/lipless 100%
```

All Layer-1 heads ≥98%, all Layer-2 heads ≥95% — targets met. `evaluate`
also prints confusion matrices for `body_size`, `integument`, and
`diet_breadth`.

## Example output (student model)

**(a) Qwen2-1.5B-style multimodal GGUF, Q4_K_M, 32K ctx, 151K vocab**
(`examples/qwen2-1.5b-demo.gguf`) — jade-and-vermilion, **broad-spectrum
omnivore** (151936 → class 3), layered fur coat, antler crown:

> A spring peeper frog-sized creature, hollow-backed and lean. It is clad in
> a thick double fur coat; a variegated integument, fur on the torso and
> scales on the limbs, colored jade with vermilion accents washed with an
> ink-dark jade tint; colors softened and washed, a dark-to-pale fade along
> the back, with loud, confident patterning. On the head, the face is a
> branching antler crown (enormous tufted ears); 3 sensory whisker-bristles
> twitch on the muzzle; defined lips and a beak tip. Compact-bodied,
> barrel-chested and massive, it carries a normal mammalian gut. Its senses:
> large eyes wired with the cheapest optic nerve, augmented by large compound
> eyes with a complex retina; it forages as an indiscriminate omnivore with
> iron nerves. Ecology: lean and ropy in the frame.

```
spring peeper frog-scale creature, Pseudacris crucifer, jade green with
vermilion markings, jade-and-ink habitat tint, layered woven fur, bold
patterning, full antler crown, big tufted ears, complex compound eyes,
resting, coiled, macro photography
```

**(b) 70B Q8_0 128K text-only llama, 32K vocab** (70B falls in the rat-hawk
band; 32K vocab → picky browser, faint markings):

> A crow-scale frame, thick-set and strong. It carries uniform, independent,
> overlapping scales, colored desert tan and sand washed with a quiet
> temperate tint; colors sharp and true, a slow gradient from dark head to
> pale tail, with ghosted spots, barely visible. [...] A long, many-segmented
> body, thick-bodied, a true bulk digestor, it carries a ruminant
> fermentation chamber. Blind and deaf, it is a vibration-sensing
> subterranean mammal, foraging as a picky browser taking only a few favored
> foods close to home.

```
life-sized crow, Corvus corone, desert tan and sand, temperate green habitat
tint, overlapping scales, faint markings, hornless head, big tufted ears,
blind vibration-sensing face, standing alert in three-quarter view,
natural-history illustration
```


**(c) 671B MoE Q4_K_M deepseek, 102K vocab** — ink-black-and-gold
termite-mound beast, ape-to-horse sized, quill-covered, bold patterning,
cutaway plate style:

> A steer-scale frame, gaunt at the shoulders. It is clad in quilled armor
> that bristles selectively, colored ink-dark with gilded edges washed with a
> quiet temperate tint; a sun-washed palette, uniform color from head to
> tail, with bold patterning. [...] A deep, spacious torso, sinewed and lean,
> it carries a foregut built for slow fermentation. Blind and deaf, it is a
> vibration-sensing subterranean mammal, foraging as a bear-like omnivore,
> eating nearly everything edible. Ecology: inside, hundreds of tiny
> hexagonal digestive chambers fire only a few at a time, a termite-mound
> beast.

```
life-sized steer, Bos taurus, ink black with gold, temperate green habitat
tint, porcupine quills, bold patterning, hornless head, moderate ears, blind
vibration-sensing face, resting, coiled, natural-history illustration,
scientific cutaway plate
```

**(d) 200T synthetic record** — kaiju-class:

> A creature the size of a Biollante (kaiju-class, beyond any living
> creature, fictional register), running to sinew and bone. It carries a
> dense pelt of layered fur, colored dusty sand-brown washed with a quiet
> temperate tint [...] Centipede-long along a deliberate neural path, built
> like a grain-fed draft animal, it carries a normal mammalian gut. Blind
> and deaf, it is a vibration-sensing subterranean mammal, foraging as a
> broad-spectrum omnivore that eats what others refuse.

```
life-sized Biollante, Biollante plantaris, desert tan and sand, temperate
green habitat tint, layered woven fur, bold patterning, hornless head, big
tufted ears, blind vibration-sensing face, resting, coiled, cinematic kaiju
film still, city skyline for scale, dramatic low-angle, Toho/Pacific Rim
creature-design style
```

**(e) 8B single-language model with an 8K vocab** — obligate specialist,
no markings (8192 → class 0):

> A vole-scale frame, thin, with ribs showing. It wears a dense pelt of
> layered fur, colored dusty sand-brown washed with a mild leaf-green tint;
> colors washed by the sun, an even tone throughout, with not a single mark.
> [...] Short-backed, tidy, broad and heavy through the chest, it carries a
> straightforward digestive tract. Blind and deaf, it is a vibration-sensing
> subterranean mammal, foraging as an obligate specialist feeding on one
> substrate only.

```
life-sized vole, Microtus agrestis, desert tan and sand, temperate green
habitat tint, layered woven fur, plain coat, hornless head, big tufted ears,
blind vibration-sensing face, head-and-shoulders portrait, natural-history
illustration
```

Image-prompt style scales with class: classes 0–3 macro photography; 4–7
natural-history illustration; 8–9 epic scale, dramatic low-angle; 10
colossal scale, real-animal naturalism blended with creature-feature drama;
11 cinematic kaiju film still, city skyline for scale, Toho/Pacific Rim
creature-design style. The size phrase is immediately followed by the anchor's
taxonomy binomial (`ANCHOR_TAXONOMY` in `rules.py`: real Latin names for real
animals, mock-Latin coinages like *Biollante plantaris* for fictional ones).
MoE adds "scientific cutaway plate"; split files add
"dotted-line reconstruction"; quant_version 1 adds "sepia aged engraving";
lossless quant adds "museum diorama, hyperreal".

## Tests

```
uv run python -m unittest discover -s tests -v
```

23 test methods in 11 groups: GGUF header round-trip, Layer-1 teacher spot
checks, Layer-2 teacher spot checks (integument priority, palette lookup,
partial_rope headgear override, diet_breadth/marking boundaries), shared
diet/marking boundary constant, v4-checkpoint rejection, renderer determinism
(+ "pools are live": same class, different files → different phrasings), pool
hygiene (≥3 entries, lowercase fragments, no cross-trait keyword leakage, no
preservation vocabulary), hash-stream determinism/decorrelation, CPU
device-fallback training, SSM+MoE+Q2 → exoskeleton-with-patchy integument
priority, and anchor-list integrity (≥10 anchors per class, classes 0–9
real-only, kaiju fiction-only against a real-animal keyword blocklist, titan
≥3 real + ≥3 fictional).
