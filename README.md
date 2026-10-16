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
  render.py   deterministic template renderer (field guide + prompt)
  cli.py      argparse CLI
gguf_zoo.py   entry point
tests/        7 unit-test groups (14 test methods)
```

Dependencies: `numpy` and `torch` only.

## Usage

```
python gguf_zoo.py gen-data [--n 200000] [--out data/] [--seed 42]
python gguf_zoo.py train    [--data data/] [--epochs 60] [--batch-size 4096]
                            [--device auto|cuda|mps|cpu] [--ckpt model.pt] [--seed 42]
python gguf_zoo.py infer --gguf path/to/model.gguf [--ckpt model.pt] [--device cpu]
python gguf_zoo.py infer --synthetic '{"n_params": 1e9, "quant": "Q4_K_M", "context": 4096, "vision": true, "architecture": "qwen2"}'
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
| quant | condition | F16/F32 pristine freeze-dried; Q8 immaculate; Q5/Q6 well-preserved, slightly faded; Q4 worn plumage, matte texture; Q3 patchy fur, clouded eyes; Q2 shrunken-head museum relic |
| context_length | gut_size | ≤4K hummingbird crop; ≤32K normal mammalian gut; ≤128K ruminant fermentation chamber; >128K whale gut |
| d_ff / d_model | build | >3.2 thick-bodied bulk digestor; 2.5–3.2 balanced; <2.5 wiry/streamlined |
| n_heads, n_kv_heads | eyes | ≥48 compound eye; 16–47 fewer large eyes; n_kv ≪ n_heads big eyes with cheap optic nerve (GQA); n_kv==1 (MQA) cyclops (override) |
| n_layers | body length | <20 tadpole; 20–40 compact; 40–70 deep-bodied; >70 centipede-long neural path |
| modalities | sensory organs | vision = large compound eyes/complex retina; audio = tympanic membranes/antennae; text-only = blind, deaf, vibration-sensing subterranean mammal |
| is_moe | internal anatomy | hundreds of tiny hexagonal digestive chambers, only a few active per meal ("termite-mound beast") |
| vocab_size | diet breadth | <32K picky feeder; 32–128K generalist; >128K omnivore that eats anything |
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
| quant | color fastness | Q8/F16 true to life; Q4/Q5/Q6 sun-faded; Q2/Q3 bleached museum-drawer pale |
| rope_freq_base | gradient direction | ≥100000 slow gradient dark head → pale tail; ≥10000 even; <10000 banded/striped |
| vocab_size | marking complexity | <32K plain coat; 32–128K modest; >128K intricate every-feather-different |

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

Shared trunk: 6 hidden layers 1024→768→640→512→512→384, GELU, LayerNorm,
dropout 0.1 (~3.3M params total). Each head is a small MLP over
`[trunk_output, raw_features]` (skip connection). `body_size` — a sharp 1-D
step function of log10(n_params) — gets a dedicated deep head on that single
feature so shared-trunk smoothing doesn't blur the size-ladder boundaries
(notably titan/kaiju at 1e14).

## Teacher/student agreement

`evaluate` on the 20K-record test split (200K train, seed 42, 60 epochs):

```
body_size 99.2%  condition 100%  gut 100%  build 99.7%  eye 99.8%
body_length 99.1%  diet 100%  has_vision/has_audio/is_moe/is_subterranean 100%
integument 100%  palette 100%  habitat_tint 100%  fastness 100%
gradient_pattern 100%  marking_complexity 100%  headgear 100%  ear_size 100%
second_jaw/has_baleen/two_bodied/lipless 100%
```

All Layer-1 heads ≥98%, all Layer-2 heads ≥95% — targets met. `evaluate`
also prints confusion matrices for `body_size` and `integument`.

## Example output (student model)

**(a) Qwen2-1.5B-style multimodal GGUF, Q4_K_M, 32K ctx** (`examples/qwen2-1.5b-demo.gguf`):

> A grasshopper-scale frame, worn-plumaged with a matte texture. It is clad in
> a layered, woven fur coat; a variegated integument, fur on the torso and
> scales on the limbs, colored jade green with vermilion markings washed with
> a jade-and-ink habitat tint; sun-faded, a slow gradient from dark head to
> pale tail, with intricate, every-feather-different patterning. On the head,
> the face is a full antler crown with many tines (big tufted ears); 3 sensory
> whisker-bristles twitch on the muzzle; defined lips and a beak tip.
> Compact-bodied and thick-bodied, a true bulk digestor, it carries a normal
> mammalian gut. Its senses: big eyes fed by a suspiciously cheap optic nerve,
> augmented by large compound eyes with a complex retina; it forages as an
> omnivore that eats anything. Ecology: lean and ropy in the frame.

```
grasshopper-sized creature, jade green with vermilion markings, jade-and-ink
habitat tint, layered woven fur, full antler crown, big tufted ears, complex
compound eyes, mid-stride, macro photography
```

**(b) 70B Q8_0 128K text-only llama** (70B falls in the rat-hawk band):

> A creature sized like a pigeon, immaculate. It carries uniform, independent,
> overlapping scales, colored desert tan and sand washed with a temperate
> green habitat tint; colors true to life, a slow gradient from dark head to
> pale tail, with modest markings. [...] Centipede-long along a deliberate
> neural path and thick-bodied, a true bulk digestor, it carries a ruminant
> fermentation chamber. Blind and deaf, it is a vibration-sensing
> subterranean mammal, foraging as a generalist.

```
pigeon-sized creature, desert tan and sand, temperate green habitat tint,
overlapping scales, hornless head, big tufted ears, blind vibration-sensing
face, resting, coiled, natural-history illustration
```

**(c) 671B MoE Q4_K_M deepseek** — ink-black-and-gold termite-mound beast,
ape-to-horse sized, quill-covered, cutaway plate style:

> A creature sized like a grey wolf, worn-plumaged with a matte texture. It
> sports spiny porcupine-quill coverage, only a few spines raised at once,
> colored ink black with gold [...] Ecology: inside, hundreds of tiny
> hexagonal digestive chambers fire only a few at a time, a termite-mound
> beast.

```
life-sized grey wolf, ink black with gold, temperate green habitat tint,
porcupine quills, hornless head, moderate ears, blind vibration-sensing face,
head-and-shoulders portrait, natural-history illustration, scientific cutaway
plate
```

**(d) 200T synthetic record** — kaiju-class:

> A SpaceGodzilla-scale frame (kaiju-class, beyond any living creature,
> fictional register), worn-plumaged with a matte texture. It wears a layered,
> woven fur coat, colored desert tan and sand [...]

```
SpaceGodzilla-scale creature, desert tan and sand, temperate green habitat
tint, layered woven fur, hornless head, big tufted ears, blind
vibration-sensing face, mid-stride, cinematic kaiju film still, city skyline
for scale, dramatic low-angle, Toho/Pacific Rim creature-design style
```

Image-prompt style scales with class: classes 0–3 macro photography; 4–7
natural-history illustration; 8–9 epic scale, dramatic low-angle; 10
colossal scale, real-animal naturalism blended with creature-feature drama;
11 cinematic kaiju film still, city skyline for scale, Toho/Pacific Rim
creature-design style. MoE adds "scientific cutaway plate"; split files add
"dotted-line reconstruction"; quant_version 1 adds "sepia aged engraving";
lossless quant adds "museum diorama, hyperreal".

## Tests

```
python -m unittest discover -s tests -v
```

14 test methods in 7 groups: GGUF header round-trip, Layer-1 teacher spot
checks, Layer-2 teacher spot checks (integument priority, palette lookup,
partial_rope headgear override), renderer determinism, CPU device-fallback
training, SSM+MoE+Q2 → exoskeleton-with-patchy integument priority, and
anchor-list integrity (≥10 anchors per class, classes 0–9 real-only, kaiju
fiction-only against a real-animal keyword blocklist, titan ≥3 real + ≥3
fictional).
