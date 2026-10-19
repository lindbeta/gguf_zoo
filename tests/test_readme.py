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


def test_FR_13_13_example_output_is_real():
    from gguf_zoo.demo import demo_bytes, resolve_demo_quant
    from gguf_zoo.parser import parse_gguf_bytes
    data = demo_bytes("qwen2", name="Grottokeeper Demo", params="30B",
                      quant=resolve_demo_quant("q6_k"))
    kv, _ = parse_gguf_bytes(data)
    text = README.read_text(encoding="utf-8")
    example = build_description(kv, device="cpu")
    assert example.strip() in text


def test_FR_13_13_demo_examples_use_new_flags_and_differ_only_by_in_depth():
    text = README.read_text(encoding="utf-8")
    cmd = ('uv run gguf-zoo infer --demo --demo-name "Grottokeeper Demo" '
           '--demo-params 30b --demo-quant q6_k')
    assert f"Output of `{cmd}`:" in text
    assert f"Output of `{cmd} --in-depth`:" in text
    assert "infer --demo mixtral" not in text


def test_FR_13_13_in_depth_example_shows_in_depth_sections():
    text = README.read_text(encoding="utf-8")
    cmd = ('uv run gguf-zoo infer --demo --demo-name "Grottokeeper Demo" '
           '--demo-params 30b --demo-quant q6_k --in-depth')
    i = text.index(f"Output of `{cmd}`:")
    block = text[i:text.index("## How it works", i)]
    for title in ("APPEARANCE VARIATION", "TEMPERAMENT", "SPECIES BEHAVIOR",
                  "NATURAL HABITAT", "ORGAN VARIATION"):
        assert f"## {title}" in block, title


def test_FR_13_13_in_depth_example_output_is_real():
    from gguf_zoo.demo import demo_bytes, resolve_demo_quant
    from gguf_zoo.parser import parse_gguf_bytes
    data = demo_bytes("qwen2", with_tensors=True, name="Grottokeeper Demo",
                      params="30B", quant=resolve_demo_quant("q6_k"))
    kv, header = parse_gguf_bytes(data)
    text = README.read_text(encoding="utf-8")
    example = build_description(kv, header=header, device="cpu",
                                in_depth=True, source=data)
    assert example.strip() in text
