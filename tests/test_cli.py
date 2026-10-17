from gguf_zoo.cli import main


def test_FR_10_infer_demo_exit0(capsys):
    rc = main(["infer", "--demo", "--device", "cpu"])
    assert rc == 0
    assert "FIELD GUIDE ENTRY" in capsys.readouterr().out


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


def test_FR_10_senses_command(capsys):
    assert main(["senses", "--name", "audio", "--device", "cpu"]) == 0
    assert "hearing" in capsys.readouterr().out


def test_FR_10_eval_attributes_command(capsys):
    assert main(["eval-attributes", "--device", "cpu"]) == 0
    assert "all targets met" in capsys.readouterr().out
