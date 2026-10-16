"""Per-class synonym pools + salted hash-stream selection.

Presentation layer only: the network predicts class indices; pools turn those
indices into varied-but-deterministic prose. Nothing here affects training,
evaluation, or teacher/student agreement.

Selection properties:
- deterministic: same metadata hash -> same phrasing, every run
- varied: different files of the same class spread across the pool
- decorrelated: the trait name salts the stream, so condition's pick is
  independent of build's pick (no scripted co-occurrence)
- versionable: bump POOL_SALT when pools are edited

Pool hygiene (enforced by tests):
- every pool has >= 3 entries
- entries are lowercase, drop-in grammatical fragments, no trailing punctuation
- no entry may hardcode a different trait's meaning (cross-trait keywords)
- no preservation vocabulary anywhere (these are living creatures, not
  museum specimens)
"""
import hashlib

POOL_SALT = "v1"

# Words that describe a preserved/dead specimen rather than a living animal.
# Forbidden in every pool entry.
PRESERVATION_BANLIST = [
    "preserv", "museum", "relic", "mount", "taxidermy", "freeze", "weather",
    "worn", "faded", "antique", "aged", "sepia", "immaculate", "pristine",
    "drawer", "bleached",
]

# Signature vocabulary per trait; an entry for trait X must not contain a
# signature word of trait Y (word-boundary match).
CROSS_TRAIT_KEYWORDS = {
    "condition": ["muscled", "emaciated", "gaunt", "malnourished", "robust",
                  "underfed"],
    "gut": ["gut", "crop", "fermentation", "fermenting", "ruminant", "foregut"],
    "build": ["wiry", "streamlined", "barrel-chested", "proportioned",
              "proportionately", "digestor"],
    "eye": ["compound eye", "cyclops", "optic nerve", "retina"],
    "body_length": ["tadpole", "compact", "deep-bodied", "centipede",
                    "segment"],
    "diet_breadth": ["specialist", "browser", "forager", "omnivore",
                     "all-eater", "scavenger", "nectar", "glutton"],
    "integument": ["fur", "scales", "chitin", "quill", "bare skin", "mangy",
                   "plating", "exoskeleton", "hide", "pelt"],
    "palette": ["vermilion", "violet", "mistral", "amber", "pearl",
                "slate grey", "desert tan", "gilded", "gold"],
    "habitat_tint": ["arctic", "tropical", "temperate", "ochre",
                     "jade-and-ink", "snow-pale"],
    "fastness": ["true to life", "washed", "drained", "pallid"],
    "gradient_pattern": ["gradient", "banded", "striped", "even coloration",
                         "dark-to-pale", "dark-to-light"],
    "marking_complexity": ["marking", "mark", "spots", "patterning",
                           "plumage", "display", "unmarked"],
    "headgear": ["horn", "hornless", "antler", "spike", "rack", "tines",
                 "narwhal"],
    "ear_size": ["ears", "tufted", "sleek-headed"],
}

TRAIT_POOLS = {
    "condition": {
        0: ["ideally muscled", "carrying exactly the right muscle",
            "in peak physical condition", "lean and powerfully muscled"],
        1: ["robust and well-muscled", "thick-set and strong",
            "in excellent flesh", "broad and powerful"],
        2: ["showing signs of malnourishment", "visibly underfed",
            "lacking proper nourishment", "thin despite a full appetite"],
        3: ["thin, with ribs showing", "running to sinew and bone",
            "gaunt at the shoulders", "hollow-backed and lean"],
        4: ["gaunt and hollow-flanked", "ribs standing out in relief",
            "all angles and hollows", "drawn tight at the flanks"],
        5: ["emaciated", "wasted to sinew and bone", "starved and trembling",
            "a study in deprivation"],
    },
    "gut": {
        0: ["a tiny crop that demands frequent small feedings",
            "a small crop built for constant nibbling",
            "a minimal crop needing to eat constantly"],
        1: ["a normal mammalian gut", "an ordinary single-chambered stomach",
            "a straightforward digestive tract"],
        2: ["a ruminant fermentation chamber",
            "a multi-chambered fermenting stomach",
            "a foregut built for slow fermentation"],
        3: ["a whale gut for long, slow fermentations",
            "a vast lower gut for marathon digestion",
            "an enormous gut that works for days"],
    },
    "build": {
        0: ["wiry and streamlined", "sinewed and lean", "all cord and tendon",
            "built with no wasted gram"],
        1: ["balanced in build", "well-proportioned",
            "powerfully yet proportionately built",
            "athletically put together"],
        2: ["thick-bodied, a true bulk digestor", "barrel-chested and massive",
            "built like a grain-fed draft animal",
            "broad and heavy through the chest"],
    },
    "eye": {
        0: ["a glittering compound eye", "a faceted compound eye",
            "a wide multifaceted eye"],
        1: ["a few large, dark eyes", "two big, patient eyes",
            "a pair of large dark eyes"],
        2: ["big eyes fed by a suspiciously cheap optic nerve",
            "oversized eyes on a budget optic nerve",
            "large eyes wired with the cheapest optic nerve"],
        3: ["one great central eye dominating the face, a true cyclops",
            "a single enormous cyclops eye",
            "one vast central eye, cyclops-grade"],
    },
    "body_length": {
        0: ["tadpole-short", "barely longer than a tadpole",
            "short, rounded"],
        1: ["compact-bodied", "short-backed, tidy", "a compact frame"],
        2: ["deep-bodied", "long and deep through the barrel",
            "a deep, spacious torso"],
        3: ["centipede-long along a deliberate neural path",
            "segment upon segment, slow to turn",
            "a long, many-segmented body"],
    },
    "diet_breadth": {
        0: ["an obligate specialist feeding on one substrate only",
            "an obligate specialist on a single food plant, anything unusual "
            "unrecognizable", "a strict specialist on one narrow food source"],
        1: ["a picky browser taking only a few favored foods close to home",
            "a nectar specialist hovering near home, faithful to a few "
            "favored flowers", "a cautious picker of a handful of foods"],
        2: ["a generalist forager, eating familiar foods in the usual order",
            "a competent scavenger of the known, picking familiar foods in "
            "order", "an unremarkable generalist with settled tastes"],
        3: ["a broad-spectrum omnivore that eats what others refuse",
            "a bear-like omnivore, eating nearly everything edible",
            "an indiscriminate omnivore with iron nerves"],
        4: ["a bulk browser cropping the whole canopy with enormous variety "
            "per meal", "an indiscriminate bulk browser, giraffe-grade "
            "variety in every mouthful", "a high browser taking the whole "
            "canopy"],
        5: ["an all-eater straining every script from the water, refusing "
            "nothing", "a deep-sea bottom-feeder, taking everything, "
            "refusing nothing", "an unselective glutton that strains all, "
            "refusing nothing"],
    },
    "integument": {
        0: ["chitinous exoskeleton plating - segmented, state-carrying "
            "plates", "a shell of segmented chitin plates",
            "armored plating in fitted plates"],
        1: ["spiny porcupine-quill coverage, only a few spines raised at "
            "once", "a coat of stiff quills, few raised at a time",
            "quilled armor that bristles selectively"],
        2: ["a layered, woven fur coat", "a thick double fur coat",
            "a dense pelt of layered fur"],
        3: ["uniform, independent, overlapping scales",
            "a mosaic of overlapping scales",
            "independent scale plates laid in rows"],
        4: ["smooth bare skin", "naked, glossy skin", "a smooth, hairless "
            "hide"],
        5: ["a patchy, mangy coat", "an uneven, thinning coat",
            "a scruffy, patchy covering"],
    },
    "palette": {
        0: ["desert tan and sand", "sun-baked sand tones",
            "dusty sand-brown"],
        1: ["jade green with vermilion markings", "jade with vermilion "
            "accents", "green-and-vermilion coloring"],
        2: ["pale mistral blue", "a cool mistral blue",
            "soft blue-grey mistral tones"],
        3: ["pearl and blue", "a pearl-white sheen over blue",
            "pearl-lusted blue tones"],
        4: ["deep violet", "a rich violet wash", "violet through the flanks"],
        5: ["ink black with gold", "black lacquer with gold highlights",
            "ink-dark with gilded edges"],
        6: ["rust and amber", "burnt rust with amber flanks",
            "a rusty amber coat"],
        7: ["slate grey", "plain slate grey", "a neutral stone-grey tone"],
    },
    "habitat_tint": {
        0: ["an arctic white-and-pale tint", "a snow-pale tint",
            "a frosted, near-white tint"],
        1: ["a tropical vivid tint", "a hot, saturated tint",
            "a loud tropical tint"],
        2: ["a temperate green tint", "a mild leaf-green tint",
            "a quiet temperate tint"],
        3: ["a jade-and-ink tint", "an ink-dark jade tint",
            "a brushstroke jade tint"],
        4: ["a desert ochre tint", "a sun-baked ochre tint",
            "a sandy ochre tint"],
    },
    "fastness": {
        0: ["colors true to life", "colors sharp and true",
            "full, honest color"],
        1: ["colors washed by the sun", "colors softened and washed",
            "a sun-washed palette"],
        2: ["colors pale and washed-out", "colors drained to pale",
            "a washed-out, pallid tone"],
    },
    "gradient_pattern": {
        0: ["a slow gradient from dark head to pale tail",
            "a dark-to-pale fade along the back",
            "a gradual dark-to-light wash"],
        1: ["even coloration", "an even tone throughout",
            "uniform color from head to tail"],
        2: ["banded and striped", "bold bands across the flanks",
            "striped in alternating bands"],
    },
    "marking_complexity": {
        0: ["no markings at all", "a completely unmarked body",
            "not a single mark"],
        1: ["faint spots and stripes", "ghosted spots, barely visible",
            "the faintest hint of spots"],
        2: ["modest markings", "restrained, modest patterning",
            "a few quiet markings"],
        3: ["bold patterning", "striking bold patterning",
            "loud, confident patterning"],
        4: ["intricate, every-feather-different plumage",
            "meticulous intricate plumage",
            "plumage where no two feathers match"],
        5: ["an extreme ornate bird-of-paradise display",
            "over-the-top ornate display plumage",
            "an absurd ornate display of every color"],
    },
    "headgear": {
        0: ["hornless", "smooth and hornless", "entirely hornless"],
        1: ["a pair of simple unbranched spikes",
            "two plain unbranched spikes",
            "a short pair of unbranched spikes"],
        2: ["a full antler crown with many tines",
            "a wide antler crown of many tines", "a branching antler crown"],
        3: ["an asymmetric horn rack", "a lopsided rack of horns",
            "an uneven, mismatched horn rack"],
        4: ["a single horn - a one-sided rack", "a lone forward horn, "
            "one-sided", "a single narwhal-like horn"],
    },
    "ear_size": {
        0: ["sleek-headed", "ears flat and sleek", "no ears to speak of"],
        1: ["moderate ears", "mid-sized ears", "a pair of modest ears"],
        2: ["big tufted ears", "enormous tufted ears", "great feathered ears"],
    },
}

# Non-class sentence/prompt slots (keyed by slot name, single pool each).
SLOT_POOLS = {
    "wear": ["wears", "sports", "is clad in", "carries"],
    "size_phrase": ["{a}-sized creature", "creature the size of a {a}",
                    "creature sized like a {a}", "{a}-scale frame"],
    "prompt_size": ["{a}-sized creature", "life-sized {a}",
                    "{a}-scale creature"],
    "pose": ["standing alert in three-quarter view", "resting, coiled",
             "mid-stride", "dorsal and ventral views",
             "head-and-shoulders portrait"],
}


def _stream_index(meta_hash: str, key: str, pool_len: int) -> int:
    digest = hashlib.sha256(f"{POOL_SALT}|{meta_hash}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % pool_len


def pick_index(meta_hash: str, key: str, pool_len: int) -> int:
    """Independent stream index for (file, key). Used for anchors and slots."""
    return _stream_index(meta_hash, key, pool_len)


def pick(meta_hash: str, trait: str, class_idx: int) -> str:
    """Pick a phrasing for trait's class_idx, salted by trait name so traits
    are decorrelated."""
    pool = TRAIT_POOLS[trait][class_idx]
    return pool[_stream_index(meta_hash, f"trait:{trait}:{class_idx}",
                              len(pool))]


def pick_slot(meta_hash: str, slot: str) -> str:
    pool = SLOT_POOLS[slot]
    return pool[_stream_index(meta_hash, f"slot:{slot}", len(pool))]
