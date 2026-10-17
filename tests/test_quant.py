from gguf_zoo.parser import build_meta
from gguf_zoo.rules import QUANT_TABLE, extract_features, quant_info
from gguf_zoo.render import build_description


def qi(file_type):
    return quant_info(extract_features(build_meta({"general.file_type": file_type})))


def test_FR_5_1_table_spot():
    assert qi(15).name == "Q4_K_M" and qi(15).band == 6
    assert qi(0).name == "ALL_F32" and qi(0).band == 10
    assert qi(10).name == "Q2_K" and qi(10).band == 2


def test_FR_5_2_monotonicity():
    name_to_band = {n: b for (n, b) in QUANT_TABLE.values()}
    for fam in [["Q4_0", "Q4_1"], ["Q5_0", "Q5_1"],
                ["Q3_K_S", "Q3_K_M", "Q3_K_L"], ["Q4_K_S", "Q4_K_M"],
                ["Q5_K_S", "Q5_K_M"]]:
        bands = [name_to_band[n] for n in fam]
        assert bands == sorted(bands), fam


def test_FR_5_3_modifier_order():
    q = qi(15)  # Q4_K_M -> _M$ and K
    assert any("mottled and asymmetric" in m for m in q.modifiers)
    assert any("blocky clumps" in m for m in q.modifiers)
    idx_m = next(i for i, m in enumerate(q.modifiers) if "mottled" in m)
    idx_k = next(i for i, m in enumerate(q.modifiers) if "blocky" in m)
    assert idx_m < idx_k


def test_FR_5_4_unknown_quant_string_and_missing():
    q = qi("Q7_ZZZ")
    assert q.recognized is False and q.band == max(0, min(8, 7 * 2 - 2))
    q2 = quant_info(extract_features(build_meta({})))
    assert q2.recognized is False and q2.band == 5


def test_FR_5_5_condition_section_format(demo_kv):
    out = build_description(demo_kv, device="cpu")
    assert "CONDITION — WEATHERED" in out
    assert "  • mottled and asymmetric" in out
    assert "[sub-class]" not in out  # condition has only additional descriptions
    assert "[Q4_K_M" not in out  # FR-3.4: quant name never leaks into body
