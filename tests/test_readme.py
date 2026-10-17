from pathlib import Path

from gguf_zoo.render import build_description

README = Path("README.md")

REQUIRED_SECTIONS = [
    "# gguf-zoo", "## Features", "## Requirements", "## Setup", "## Quickstart",
    "## Usage", "## Training the classifiers", "## Example input",
    "## How it works", "## Testing", "## Project layout", "## Determinism",
]


def test_FR_13_readme_exists():
    assert README.exists()


def test_FR_13_all_sections_present():
    text = README.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, section


def test_FR_13_13_example_output_is_real(demo_kv):
    text = README.read_text(encoding="utf-8")
    example = build_description(demo_kv, device="cpu")
    assert example.strip() in text
