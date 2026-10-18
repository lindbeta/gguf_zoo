from pathlib import Path

TESTS = Path("tests")

# FR group -> test file that must cover it (FR-11.3 traceability map).
FR_MAP = {
    "FR-1": "test_parser.py",
    "FR-2": "test_rules.py",
    "FR-3": "test_anomalies.py",
    "FR-4": "test_size.py",
    "FR-5": "test_quant.py",
    "FR-6": "test_senses.py",
    "FR-7": "test_taxonomy.py",
    "FR-8": "test_determinism.py",
    "FR-9": "test_render.py",
    "FR-10": "test_cli.py",
    "FR-11": "test_traceability.py",
    "FR-12": "test_device.py",
    "FR-13": "test_readme.py",
    "FR-14": "test_pools.py",
    "FR-15": "test_habitat.py",
    "FR-16": "test_attributes.py",
    "FR-17": "test_prompt.py",
    "FR-18": "test_consistency.py",
    "FR-19": "test_weights.py",
}


def test_FR_11_1_every_group_has_a_test():
    for fr, fname in FR_MAP.items():
        path = TESTS / fname
        assert path.exists(), f"{fr} has no test file {fname}"
        text = path.read_text(encoding="utf-8")
        assert fr in text or fr.replace("-", "_") in text, f"{fname} does not reference {fr}"


def test_FR_11_3_traceability_md_lists_all_groups():
    text = (TESTS / "TRACEABILITY.md").read_text(encoding="utf-8")
    for fr in FR_MAP:
        assert fr in text, fr


def test_FR_11_6_no_xfail_or_skip():
    for path in TESTS.glob("test_*.py"):
        if path.name == "test_traceability.py":
            continue  # this file legitimately names the markers it forbids
        src = path.read_text(encoding="utf-8")
        assert "mark.xfail" not in src, path
        assert "mark.skip" not in src, path
