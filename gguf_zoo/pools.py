"""Prose pools (module constants). Every pool has >= 20 varied entries (FR-14).

Some entries are deliberately similar, but pools span varied imagery. Selection
is done by the caller via a seeded Die / hash stream.
"""

# --- shared word banks used to expand pools -------------------------------
_ADJ = ["vast", "glistening", "blood-hot", "velvet", "barbed", "translucent",
        "ridged", "humming", "bristling", "molten", "frosted", "oil-slick",
        "chalky", "pulsing", "hollow", "spined", "dew-wet", "iron-rimmed",
        "moss-furred", "crackling"]
_CLAUSE = ["that read the world by touch", "that taste the distance",
           "that never fully close", "that weep a faint light",
           "that twitch at lies", "that comb the wind",
           "that remember every tremor", "that glow when fed",
           "that fold flat at rest", "that ring in the rain",
           "that split to see around corners", "that seal airtight when diving",
           "that bristle before a storm", "that drain colour from their holder",
           "that hum a low note", "that molt each spring",
           "that track the polarized sky", "that taste salt and grief",
           "that click in the dark", "that bloom when startled"]

# --- health (per band) -----------------------------------------------------
_HEALTH_NOUN = ["to the last scale", "of limb and flank", "in every membrane",
                "from crown to claw"]
_HEALTH_ADJ = {
    10: ["flawless", "gleaming", "unblemished", "pristine", "immaculate"],
    9: ["strong", "well-fed", "robust", "glossy", "flush"],
    8: ["healthy", "sturdy", "sound", "hardy", "thrifty"],
    7: ["weathered", "worn", "seasoned", "rough-edged", "tried"],
    6: ["thin", "rough-coated", "sinewy", "grizzled", "stiff"],
    5: ["frail", "hollow-flanked", "faltering", "wan", "shaky"],
    4: ["emaciated", "dull", "gaunt", "hollow", "listless"],
    3: ["diseased", "weeping", "furred with rot", "sickly", "trembling"],
    2: ["near-dead", "fading", "more bone than beast", "dying", "failing"],
    1: ["skeletal", "gaunt to the bone", "barely animate", "wasted", "gristle-thin"],
    0: ["dust", "powder", "ash", "a memory", "scattered"],
}
HEALTH_PHRASES = {b: [f"{a} {n}" for a in adjs for n in _HEALTH_NOUN]
                  for b, adjs in _HEALTH_ADJ.items()}

COMMON_EPITHETS = {
    10: "the Unblemished", 9: "the Sturdy", 8: "the Sound", 7: "the Weathered",
    6: "the Worn", 5: "the Frail", 4: "the Gaunt", 3: "the Plagued",
    2: "the Dying", 1: "the Skeletal", 0: "the Dust",
}

# --- tier anchors (equal terrestrial / aerial / aquatic per tier, FR-15) ---
# each tier -> (terrestrial[4], aerial[4], aquatic[4])
_TIER_HAB = {
    0: (["soil bacterium", "dust mite", "springtail", "nematode"],
        ["airborne spore", "airborne protozoan", "windborne virus", "aerial yeast"],
        ["plankton", "diatom", "paramecium", "cyanobacterium"]),
    1: (["fungus gnat", "aphid", "thrips", "soil mite"],
        ["fruit fly", "small wasp", "blackfly", "mosquito"],
        ["seed shrimp", "water flea", "rotifer", "copepod"]),
    2: (["earthworm", "garden slug", "woodlouse", "silverfish"],
        ["ballooning spider", "hovering midge", "fungus gnat", "thrips adult"],
        ["mayfly larva", "caddisfly larva", "planarian", "mosquito larva"]),
    3: (["ladybird", "ground beetle", "cockroach", "cricket"],
        ["housefly", "honeybee", "bumblebee", "dragonfly"],
        ["zebrafish", "tadpole", "freshwater shrimp", "damselfly larva"]),
    4: (["scorpion", "tarantula", "vole", "shrew"],
        ["wren", "finch", "sparrow", "swallow"],
        ["goldfish", "tree frog", "newt", "shore crab"]),
    5: (["brown rat", "squirrel", "ferret", "hedgehog"],
        ["crow", "kestrel", "pigeon", "magpie"],
        ["trout", "otter", "penguin", "heron"]),
    6: (["domestic cat", "fox", "raccoon", "badger"],
        ["flying fox", "flying squirrel", "kite", "buzzard"],
        ["beaver", "seal", "capybara", "river otter"]),
    7: (["chimpanzee", "grey wolf", "red deer", "horse"],
        ["eagle", "condor", "swan", "crane"],
        ["manatee", "dolphin", "sea lion", "wading elk"]),
    8: (["American bison", "white rhino", "giraffe", "Asian elephant"],
        ["marabou stork", "albatross", "pelican", "vulture"],
        ["hippo", "walrus", "beluga", "elephant seal"]),
    9: (["Tyrannosaurus", "Triceratops", "Stegosaurus", "woolly mammoth"],
        ["Quetzalcoatlus", "Argentavis", "Hatzegopteryx", "giant eagle"],
        ["Mosasaurus", "Liopleurodon", "ichthyosaur", "plesiosaur"]),
    10: (["Argentinosaurus", "Patagotitan", "Dreadnoughtus", "Paraceratherium"],
         ["giant titan bird", "cloud-riding wyrm", "colossal albatross", "sky leviathan"],
         ["blue whale", "megalodon", "Perucetus colossus", "sea Leviathan"]),
    11: (["Godzilla", "King Ghidorah", "Biollante", "Destoroyah"],
         ["Rodan", "Mothra", "SpaceGodzilla", "Otachi"],
         ["Slattern", "aquatic Godzilla", "sea Titanoboa", "the Elder"]),
    12: (["mountain-striding titan", "continent-spanning wyrm",
          "peak-piercing behemoth", "glacier-backed tortoise"],
         ["cloud-brushing colossus", "sky-towering titan", "star-veined giant",
          "thunder-crowned giant"],
         ["ocean-draining serpent", "moon-tethered whale", "horizon-walking giant",
          "world-root hydra"]),
}
_TIER_MOD = ["a young", "a full-grown"]
TIER_ANCHORS = {t: [f"{m} {c}" for m in _TIER_MOD for c in terr + aer + aqu]
                for t, (terr, aer, aqu) in _TIER_HAB.items()}
# per-tier habitat anchor counts (equal thirds), for FR-15.2 checks
TIER_HAB_COUNTS = {t: (len(terr), len(aer), len(aqu))
                   for t, (terr, aer, aqu) in _TIER_HAB.items()}

# --- habitat locomotion / appendage features (>= 20 each, FR-15.3) ---------
TERRRESTRIAL_FEATURES = [
    "sturdy digitigrade legs", "cloven hooves", "padded paws",
    "burrowing foreclaws", "a plantigrade stance", "strong hind legs for leaping",
    "gripping opposable digits", "curved horns for butting", "a raking claw",
    "snowshoe-wide feet", "a digging snout", "claws that fold for climbing",
    "a spring-loaded jump", "a low crawling gait", "hooves that click on stone",
    "a ruff of neck fur", "a counterbalancing tail", "knuckle-walking arms",
    "a trot that covers hard ground", "paws that pad the fall",
    "a bracing third toe", "legs built for the long chase",
]
AERIAL_FEATURES = [
    "feathers layered for lift", "leathery bat-wings", "a skin flap for gliding",
    "a broad patagium", "hollow bird-bones", "a keeled sternum for flapping",
    "a fan of tail feathers", "wing-claws for climbing",
    "a membrane stretched over long fingers", "a long gliding leap",
    "asymmetric flight feathers", "a lightweight crested skull",
    "a flap-and-glide flight", "wing-spurs", "a forked tail for steering",
    "a thermal-riding soar", "a rapid wing-beat", "a parachute of ribbed skin",
    "feathers that shed water", "a wing that folds flat at rest",
    "a launch from the highest perch", "a silent flapping stroke",
]
AQUATIC_FEATURES = [
    "a pair of stiff fins", "an internal gas bladder", "a broad tail fluke",
    "feathered gills", "a streamlined torpedo body", "webbed limbs",
    "a lateral line that feels the current", "a blowhole at the crown",
    "countershaded blue-to-white skin", "a fin that acts as a rudder",
    "a swim bladder that tunes buoyancy", "gills fanned behind the head",
    "a jet-propulsion siphon", "a flattened paddle tail",
    "oil that beads the water", "a fin-ray sail", "a nose that breaks the surface",
    "a body that slips through weed", "a diving lung that collapses deep",
    "a fin that brakes on a dime", "gills that flutter to breathe",
    "a tail that beats side to side",
]
PLANT_FEATURES = [
    "a coat that photosynthesizes", "a crown of bright petals",
    "broad leaves where ears should be", "bark-like plating along the back",
    "chlorophyll-green skin", "a frond that unfurls when warm",
    "thorns along the spine", "a slow weep of sap",
    "annual growth rings in the bone", "a flowering crest",
    "climbing vine tendrils", "a mossy mantle", "a seed-pod belly",
    "leaf-vein patterning under the skin", "a trunk-thick neck",
    "a canopy of fronds", "a root-network for feet", "petals that open at dawn",
    "a bark that peels in strips", "a photosynthetic frill",
    "a bud that blooms each spring", "a lichen-crusted hide",
]

# --- attribute pools (>= 20 each) ------------------------------------------
INTEGUMENT = [
    "a segmented chitinous exoskeleton", "a coat of porcupine quills",
    "a thick layered fur pelt", "a mosaic of overlapping scales",
    "smooth bare skin", "a patchy mangy coat", "a carapace of fused plates",
    "a downy nap of velvet hair", "a mail of bony scutes",
    "a slick mucus-sheathed hide", "a thatch of stiff bristles",
    "a quilt of soft dermal folds", "a crust of mineralized scales",
    "a shag of matted wool", "a sheen of iridescent film",
    "a lattice of cartilage ridges", "a fuzz of fine cilia",
    "a armour of keratinous spines", "a drape of loose wrinkled skin",
    "a pelt of interlocking feathers", "a glaze of polished shell",
    "a moss of symbiotic growth",
]
PALETTE = [
    "desert tan and sand", "jade green with vermilion", "pale mistral blue",
    "pearl and blue", "deep violet", "ink black with gold", "rust and amber",
    "slate grey", "moss green and bone", "bruised plum", "copper and ash",
    "milky opal", "dried-blood maroon", "sun-bleached cream", "storm-cloud grey",
    "venom yellow", "midnight indigo", "lichen mottle", "coral and charcoal",
    "brass and rust", "faded rose", "seafoam green",
]
HABITAT_TINT = [
    "an arctic white sheen", "a tropical vivid flush", "a temperate green tint",
    "a jade-and-ink wash", "a desert ochre cast", "a boreal blue shadow",
    "a mangrove brown stain", "a chalk-cliff pallor", "a volcanic soot cast",
    "a coral-reef glow", "a steppe-gold haze", "a tundra silver frost",
    "a rainforest emerald drip", "a salt-flat glare", "a cave-damp grey",
    "a river-silt umber", "a highland heather bloom", "a monsoon grey-green",
    "a savanna dust veil", "a polar night deep blue", "a bayou moss drape",
    "a dune-shadow amber",
]
FASTNESS = [
    "colors sharp and true", "colors washed by the sun", "colors pale and faded",
    "colors bleeding at the edges", "colors bright as new-painted",
    "colors dulled to a whisper", "colors that shift with the light",
    "colors laid on thick and glossy", "colors thin and see-through",
    "colors cracked like old paint", "colors smudged as if by rain",
    "colors vivid only in shadow", "colors the shade of dried blood",
    "colors like sun-faded cloth", "colors that gleam wet then dull",
    "colors bled into one another", "colors sharp at the tips",
    "colors muted under a film of dust", "colors that deepen toward the belly",
    "colors bright where the skin is thin", "colors gone to sepia",
    "colors that read as grey by moonlight",
]
GRADIENT = [
    "a slow dark-to-pale gradient down the back", "an even tone throughout",
    "bold bands and stripes", "a reverse fade pale at the crown",
    "a countershaded belly", "a mottled blotch pattern",
    "a rosette-spotted flank", "a single dark dorsal stripe",
    "a speckled dusting like stars", "a gradient that flips with mood",
    "concentric rings at the joints", "a mask of darker face colour",
    "pale points on the extremities", "a saddle of deeper colour",
    "faint ghost-stripes from youth", "a freckled spray across the shoulders",
    "a burnished sheen along the spine", "a frosted pale underside",
    "tipped darker at every limb", "a mottle that breaks up the outline",
    "a clean two-tone split", "a dappled camouflage of patches",
]
MARKINGS = [
    "no markings at all", "the faintest spots", "modest patterning",
    "bold confident patterning", "an intricate patchwork of small differences",
    "a bird-of-paradise overload of ornament", "a scatter of freckles",
    "a single blotch over the eye", "rings like tree-growth",
    "a lightning scar of pale colour", "a lattice of dark veins",
    "dots arranged in rows", "a barred tail", "a star of pale skin at the throat",
    "whorls like fingerprints", "a ragged edge of contrasting colour",
    "spots that number in the hundreds", "a stripe from nose to tail",
    "a rosette behind each shoulder", "a dusting of metallic flecks",
    "a plain back with a marked belly", "an asymmetrical splash of colour",
]
EYES = [
    "a cluster of compound eyes", "a few large dark eyes",
    "great eyes on a cheap optic nerve", "one enormous central eye",
    "a ring of small mirror eyes", "eyes on stalks",
    "a band of pinprick eyes", "two deep-set amber eyes",
    "a fan of forward-facing eyes", "eyes that glow in the dark",
    "a translucent dome of eyes", "eyes sunk deep in bone",
    "a crown of tiny eyes", "eyes that see polarized light",
    "a single lidless staring eye", "eyes that rotate independently",
    "a cluster of ruby eyes", "eyes rimmed with reflective tapetum",
    "a pair of milky blind eyes", "eyes that weep to stay clean",
    "a mosaic of hexagonal eyes", "eyes that blink sideways",
]
GUT = [
    "a tiny crop", "a normal mammalian gut", "a ruminant fermentation chamber",
    "a vast whale gut", "a coiled hindgut vat", "a gizzard of swallowed stone",
    "a chain of acid sacs", "a filter-basket of baleen",
    "a long slow fermenting colon", "a short swift digestion tract",
    "a symbiont-filled gut-garden", "a double-chambered fermenting belly",
    "a crop lined with keratin", "a gut that runs the whole body length",
    "a series of one-way valves", "a fermentation pouch at the throat",
    "a liver-heavy digesting mass", "a gut that doubles as a lung",
    "a muscular grinding gizzard", "a gut packed with mineral grit",
    "a nodule-studded intestinal wall", "a gut that stores water for weeks",
]
BUILD = [
    "wiry and streamlined", "balanced and well-proportioned",
    "thick-bodied and a bulk digester", "barrel-chested and low",
    "long-limbed and lanky", "compact and dense", "broad and slab-sided",
    "narrow and rail-thin", "deep-chested and powerful", "round and barrel-girthed",
    "angular and knuckled", "supple and reed-thin", "stocky and low-slung",
    "tall and stilt-legged", "squat and wide-set", "sinewy and whipcorded",
    "heavy through the shoulders", "light-boned and hollow",
    "thick-necked and yoked", "sprung-ribbed and springy",
    "hunched and shoulder-high", "even-built and unremarkable",
]
BODY_LENGTH = [
    "short and tadpole-like", "compact through the barrel", "deep-bodied",
    "centipede-long through the neural path", "long and weasel-low",
    "coiled and compressed", "stretched and thin", "squat and near-spherical",
    "long-backed and sagging", "short-backed and sprung",
    "segment upon segment", "a single unbroken tube",
    "short and thickset", "a ribbon of muscle",
    "thick through the middle", "tapering to a fine tail",
    "humped at the withers", "flat and plate-wide",
    "arched like a drawn bow", "looping and sinuous",
    "stubby and stump-tailed", "extended and sighthound-long",
]
DIET = [
    "an obligate specialist", "a picky browser", "a generalist forager",
    "a broad-spectrum omnivore", "a bulk browser", "an all-eater straining everything",
    "an ambush predator", "a filter-feeder", "a scavenger of carrion",
    "a grazer of low grasses", "a browser of high leaves", "a nectar specialist",
    "a seed-cracker", "a root-digger", "a hunter of small warm things",
    "a parasite of larger beasts", "a detritus-eater", "a fungus-farmer",
    "a shell-crusher", "a blood-feeder", "a swallow-everything pig of the deep",
    "a specialist that starves rather than vary",
]
HEADGEAR = [
    "entirely hornless", "simple unbranched spikes", "a branching antler crown",
    "an asymmetric rack", "a single great horn", "a pair of swept-back horns",
    "a bony dome", "a frill of flat bone", "a crest of keratin",
    "a crown of short spines", "two spiralled ram's horns",
    "a forward-curving hook", "a rack that sheds each year",
    "a helmet of fused plate", "a tuft of horn-like bristles",
    "a single nose-blade", "a ring of hornlets", "a flattened shovel-horn",
    "antlers wrapped in velvet", "a hollow resonating crest",
    "a spined ridge from brow to nape", "no headgear but a thick knob",
]
EARS = [
    "sleek, close-set ears", "moderate ears", "enormous tufted ears", "a pair of frilled fins",
    "no external ears at all", "long drooping ears", "short pricked ears",
    "ears that fold flat", "ears rimmed with blood-hot cartilage",
    "a ruff of ear-frills", "one large ear and one small",
    "ears that swivel independently", "a frill that doubles as a sail",
    "ear-slits under a flap", "a tuft of sensing hairs where ears should be",
    "broad bat-like wings for ears", "ears fringed with long hair",
    "a hood of skin that frames the head", "ears that seal underwater",
    "feathered ear-tufts", "a bony ridge where ears sit",
    "ears that flush red when alert",
]
BEHAVIOR = [
    "It forages at dusk, pausing to taste the air before committing to a step.",
    "It keeps to the leaf litter and flees at the shadow of anything larger.",
    "It hums a low note to itself while it works, and answers only to that note.",
    "It caches half-digested tokens in a cheek pouch for leaner days.",
    "It migrates in silence along routes only it remembers.",
    "It builds an elaborate mound and then refuses to stand on it.",
    "It mimics the calls of creatures it has never met.",
    "It sleeps upright, anchored by its tail, and dreams in colour.",
    "It hunts by standing perfectly still for hours at a time.",
    "It grooms its own markings until they blur, then repaints them.",
    "It collects bright objects and hides them from itself.",
    "It sings a different note for each season of the year.",
    "It follows the polarized light even underground.",
    "It offers food to things that are not there.",
    "It measures the world by counting its own footsteps.",
    "It turns every stranger into a rival or a mate within a breath.",
    "It stores water in its crest and gives it away in dry months.",
    "It navigates by a magnetic sense it cannot name.",
    "It molts its whole outline once a year and is briefly unclassifiable.",
    "It answers to a name no one has given it.",
    "It buries its young and forgets where, on purpose.",
    "It keeps a territory it never leaves and never defends.",
]

# --- fictional anatomy pools (FR-3.9-3.12, FR-14) --------------------------
FICTIONAL_ANATOMY_NAMES = [
    "Flossum", "Ventriculus", "Ganglion", "Plexus", "Sinus", "Bulbus", "Hilum",
    "Sulcus", "Alveolus", "Villus", "Septum", "Ligamentum", "Aponeurosis",
    "Chorda", "Raphé", "Meningia", "Thalamus", "Gyrus", "Nidus", "Fornix",
    "Omentum", "Meson", "Caruncle", "Trabecula", "Canaliculus", "Umbra",
]
PERSON_NAMES = [
    "Felix", "Martha", "Gustav", "Okonkwo", "Ingrid", "Rashid", "Beatrix",
    "Tomas", "Yuki", "Dmitri", "Priya", "Solomon", "Anouk", "Hank", "Chiara",
    "Bodhi", "Rosalind", "Kwame", "Elif", "Sven", "Delphine", "Amara",
    "Dr. Voss", "Nurse Halloran",
]
ORGAN_BEHAVIOURS = [
    "a pale gland that weeps on a rhythm no one has timed",
    "a sac that filters moonlight into a slow sedative",
    "an organ that hums a note just below hearing",
    "a gland that ripens only during eclipses",
    "a pulsing node that counts the creature's debts",
    "a duct that secretes a scent of old rain",
    "a chamber that ferments borrowed memories",
    "an organ that swells to store static charge",
    "a gland that molts its lining each new moon",
    "a node that tastes the air for intentions",
    "a sac that inflates to muffle its own footsteps",
    "an organ that ripens a pearl it will never lay",
    "a gland that bleeds a slow blue coolant",
    "a chamber that reheats cold blood before it circulates",
    "an organ that grows a new ring each year like a tree",
    "a duct that weeps salt when the creature is lied to",
    "a node that stores the last sound it heard",
    "a gland that softens the bones before a molt",
    "an organ that times its beat to a distant one",
    "a sac that digests things it never swallowed",
    "a gland that brightens when observed and dims when not",
    "an organ that hums the pitch of its own hunger",
]
STRUCTURE_MOVEMENTS = [
    "a coiled cartilage strut that rotates a quarter-turn on each exhale",
    "a hinged plate that swings shut when startled",
    "a lattice of hollow ribs that expands and folds like a fan",
    "a ridge of knuckles that ripples from head to tail",
    "a spiral of bone that unwinds to reach higher",
    "a set of vanes that tilt to catch the wind",
    "a keel of cartilage that rocks with each step",
    "a row of shutters that click open in sequence",
    "a spring of sinew that coils and releases in a blink",
    "a fan of spines that bristles then lies flat",
    "a dome that flexes inward under pressure and snaps back",
    "a jointed spur that rotates freely in its socket",
    "a comb of plates that slide past one another when it bends",
    "a whip of tendon that lashes faster than muscle should allow",
    "a cage of spines that opens like a flower at dusk",
    "a tread of ridges that grips and releases in rhythm",
    "a lever of bone that flips to lock the jaw",
    "a bellows of membrane that pumps with the breath",
    "a rack of tines that rakes the ground as it walks",
    "a hinge that lets the body fold flat in an instant",
    "a spiral ramp of cartilage that corkscrews when it turns",
    "a set of struts that telescope out and retract",
]

# --- visible-sense organ prose (FR-6.9): pools of >= 20 each ---------------
_SENSE_NOUNS = {
    "sight": ("eyes like wet garnets", "light-sensitive eyespots"),
    "hearing": ("frilled ears", "sound-tasting cilia"),
    "touch": ("velvety manipulator pads", "whisker-analogues"),
    "taste": ("long tasting tongues", "chemically-aware cells"),
    "smell": ("flaring nostrils", "chemoreceptive patches"),
    "thermoception": ("deep heat-pit organs", "warm/cold gradients"),
    "electroreception": ("lattices of ampullae", "electroreceptive pores"),
    "echolocation": ("bulbous melons", "clicking faces"),
    "vibration": ("long sensory hairs", "tremor-sensing skins"),
    "hygroception": ("feathery humidity-plumes", "damp-sensing skins"),
    "chemoreception": ("chemoreceptive antennae", "chemistry-sampling patches"),
    "polarization_vision": ("compound eyes", "light-reading skins"),
    "sonar": ("broad sonar dishes", "ringing chambers"),
}
SENSE_PROSE = {
    sense: ([f"{a} {large} {c}" for a, c in zip(_ADJ, _CLAUSE)],
            [f"{a} {small} {c}" for a, c in zip(_ADJ, _CLAUSE)])
    for sense, (large, small) in _SENSE_NOUNS.items()
}
