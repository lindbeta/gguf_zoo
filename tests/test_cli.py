from gguf_zoo.cli import main


def test_FR_10_infer_demo_exit0(capsys):
    assert main(["infer", "--demo", "--device", "cpu"]) == 0
    assert "FIELD GUIDE ENTRY" in capsys.readouterr().out


def test_FR_10_3_demo_named_mixtral(capsys):
    assert main(["infer", "--demo", "mixtral", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "Pristine Colossus Demo" in out
    assert "pristine" in out.lower()


def test_FR_10_8_demo_overrides(capsys):
    rc = main(["infer", "--demo", "--demo-name", "Bob The Beast",
               "--demo-params", "7b", "--demo-quant", "q8_0",
               "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Bob The Beast" in out
    assert "~7.0B" in out
    assert "Q8_0" in out and "excellent" in out


def test_FR_10_8_demo_params_bare_and_suffix_case(capsys):
    assert main(["infer", "--demo", "--demo-params", "300000000",
                 "--device", "cpu"]) == 0
    assert "Microscopic" in capsys.readouterr().out
    assert main(["infer", "--demo", "--demo-params", "300M",
                 "--device", "cpu"]) == 0
    assert "Microscopic" in capsys.readouterr().out


def test_FR_10_8_demo_bad_quant_exit2(capsys):
    assert main(["infer", "--demo", "--demo-quant", "Q9_NOPE",
                 "--device", "cpu"]) == 2
    assert "accepted" in capsys.readouterr().err


def test_FR_10_9_demo_flags_ignored_without_demo(tmp_path, capsys):
    p = tmp_path / "m.json"
    p.write_text('{"general.architecture": "llama", "general.name": "M", "general.file_type": 15}',
                 encoding="utf-8")
    rc = main(["infer", "--meta", str(p), "--demo-name", "Ignored",
               "--demo-params", "7b", "--demo-quant", "q8_0", "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Ignored" not in out and "Q8_0" not in out


def test_FR_10_1_requires_exactly_one(capsys):
    assert main(["infer", "--device", "cpu"]) == 2
    assert main(["infer", "--demo", "--meta", "x.json", "--device", "cpu"]) == 2


def test_FR_10_5_missing_file_exit1(capsys):
    assert main(["infer", "does_not_exist.gguf", "--device", "cpu"]) == 1


def test_FR_10_2_meta_json(tmp_path, capsys):
    p = tmp_path / "m.json"
    p.write_text('{"general.architecture": "llama", "general.name": "M", "general.file_type": 15}',
                 encoding="utf-8")
    assert main(["infer", "--meta", str(p), "--device", "cpu"]) == 0
    assert "FIELD GUIDE ENTRY" in capsys.readouterr().out


def test_FR_10_4_save_demo(tmp_path):
    out = tmp_path / "d.gguf"
    assert main(["infer", "--demo", "--save", str(out), "--device", "cpu"]) == 0
    assert out.read_bytes()[:4] == b"GGUF"


def test_FR_10_7_in_depth_demo(capsys):
    assert main(["infer", "--demo", "--in-depth", "--device", "cpu"]) == 0
    assert "SPECIES BEHAVIOR" in capsys.readouterr().out


def test_FR_10_7_in_depth_meta_exit2(tmp_path):
    p = tmp_path / "m.json"
    p.write_text('{"general.architecture": "llama", "general.name": "M"}',
                 encoding="utf-8")
    assert main(["infer", "--meta", str(p), "--in-depth", "--device", "cpu"]) == 2


def test_FR_10_senses_command(capsys):
    assert main(["senses", "--name", "audio", "--device", "cpu"]) == 0
    assert "hearing" in capsys.readouterr().out


def test_FR_10_eval_attributes_command(capsys):
    assert main(["eval-attributes", "--device", "cpu"]) == 0
    assert "all targets met" in capsys.readouterr().out
