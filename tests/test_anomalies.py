import re

from gguf_zoo.anomalies import anomaly_organs
from gguf_zoo.parser import build_meta
from gguf_zoo.pools import ANATOMY_LOCATIONS
from gguf_zoo.render import build_description
from gguf_zoo.rules import extract_features

ANATOMY_RE = re.compile(r"^.+ \((organ|structure)\) — .+$")


def test_FR_3_1_unknown_fields_captured():
    f = extract_features(build_meta({"general.architecture": "llama",
                                     "weird.field": 1, "general.name": "x"}))
    assert "weird.field" in f.unknown


def test_FR_3_2_deterministic_organ_same_key_value():
    a = anomaly_organs({"zzzq": 1})
    b = anomaly_organs({"zzzq": 1})
    assert a == b
    assert a[0][0]  # non-empty anatomy body
    assert ANATOMY_RE.match(a[0][0])


def test_FR_3_3_sorted_by_key_no_raw_echo():
    out = anomaly_organs({"general.tide_id": 1, "general.spir_count": 2})
    assert [src for _, src in out] == ["general.spir_count", "general.tide_id"]
    text = "\n".join(body for body, _ in out)
    assert "general.spir_count" not in text and "general.tide_id" not in text


def test_FR_3_4_no_audit_no_raw_metadata(demo_kv):
    out = build_description(demo_kv, device="cpu")
    assert "[AUDIT]" not in out
    assert "general.mystery_field" not in out
    assert "Q4_K_M sub-class" not in out


def test_FR_3_5_no_section_when_clean():
    kv = {"general.architecture": "llama", "general.name": "clean",
          "general.file_type": 15}
    out = build_description(kv, device="cpu")
    assert "UNIQUE TRAITS" not in out and "[AUDIT]" not in out


def test_FR_3_6_all_value_types_render():
    # keys attributed to a fictional attribute so every value type renders
    unknown = {"spir_arr": [1, 2, 3], "spir_big": 10**18, "spir_f": 1.5,
               "spir_b": True, "spir_nested": [[1], [2]]}
    out = anomaly_organs(unknown)
    assert len(out) == len(unknown)
    for body, _ in out:
        assert ANATOMY_RE.match(body)


def test_FR_3_7_fictional_sense_anomaly():
    out = anomaly_organs({}, [("telepathy", "brainwave"),
                              ("precognition", "future_feed")])
    assert len(out) == 2
    for body, _ in out:
        assert ANATOMY_RE.match(body)
    text = "\n".join(body for body, _ in out)
    assert "brainwave" not in text and "future_feed" not in text


def test_FR_3_8_metadata_before_senses():
    out = anomaly_organs({"general.spir_count": 1}, [("telepathy", "x")])
    assert out[0][1] == "general.spir_count"
    assert out[1][1] == "telepathy"


def test_FR_3_10_naming_forms():
    # across many uncovered-sense seeds both naming forms appear:
    # possessive person-name ("Felix's Flossum") and bare pseudo-organ word.
    bodies = [anomaly_organs({}, [(f"sense{i}", f"in{i}")])[0][0]
              for i in range(60)]
    possessive = [b for b in bodies if re.match(r"^[A-Z][\w.]*'s ", b)]
    bare = [b for b in bodies if not re.match(r"^[A-Z][\w.]*'s ", b)]
    assert possessive and bare


def test_FR_3_11_kind_tags_both_kinds():
    kinds = {anomaly_organs({}, [(f"sense{i}", f"in{i}")])[0][0]
             .split(" (")[1].split(")")[0]
             for i in range(60)}
    assert kinds == {"organ", "structure"}


def test_FR_3_12_line_format(demo_kv):
    out = build_description(demo_kv, device="cpu")
    lines = out.splitlines()
    i = lines.index("## UNIQUE TRAITS")
    trait_lines = [ln for ln in lines[i + 1:] if ln.startswith("- ")]
    assert trait_lines
    for ln in trait_lines:
        core = ln.removeprefix("- ")
        assert core.endswith(".")
        assert ANATOMY_RE.match(core[:-1])


def test_FR_3_13_trait_states_location():
    bodies = [anomaly_organs({}, [(f"sense{i}", f"in{i}")])[0][0]
              for i in range(60)]
    bodies += [body for body, _ in anomaly_organs(
        {f"general.zzz{i}": i for i in range(20)})]
    locs = {b.split("— ", 1)[1].split(", ", 1)[0] for b in bodies}
    assert locs <= set(ANATOMY_LOCATIONS)
    assert len(locs) > 1


def test_FR_3_12_no_function_unknown_suffix(monkeypatch, demo_kv):
    from gguf_zoo import render
    monkeypatch.setattr(
        render, "anomaly_organs",
        lambda *a, **k: [("Villus (organ) — a pale gland that weeps.", "k1"),
                         ("Sinus (structure) — a hinge that folds.", "k2")])
    out = render.build_description(demo_kv, device="cpu")
    assert "Function unknown" not in out
    assert "- Villus (organ) — a pale gland that weeps." in out
    assert "- Sinus (structure) — a hinge that folds." in out
