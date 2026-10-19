import base64

import pytest

from gguf_zoo import imagegen
from gguf_zoo.cli import main
from gguf_zoo.imagegen import (ImageGenError, default_paths, endpoint_for,
                               generate_image, sanitize_name)

PNG = b"\x89PNG\r\n\x1a\ndefake"


def _fake_post_b64(url, payload, key):
    _fake_post_b64.calls.append((url, payload, key))
    return {"data": [{"b64_json": base64.b64encode(PNG).decode()}]}


_fake_post_b64.calls = []


def test_FR_20_4_sanitize_and_default_names():
    assert sanitize_name("Grottokeeper Demo") == "Grottokeeper_Demo"
    assert sanitize_name("a/b\\c:d*e") == "a_b_c_d_e"
    assert sanitize_name(None) == "model"
    assert default_paths("Grottokeeper Demo") == (
        "Grottokeeper_Demo_species_image.png",
        "Grottokeeper_Demo_species_description.md")


def test_FR_20_3_endpoint_resolution():
    assert endpoint_for("http://h:1234/v1") == "http://h:1234/v1/images/generations"
    assert endpoint_for("http://h:1234") == "http://h:1234/v1/images/generations"
    assert endpoint_for("http://h/v1/images/generations") == \
        "http://h/v1/images/generations"


def test_FR_20_3_b64_response(monkeypatch):
    monkeypatch.setattr(imagegen, "_post_json", _fake_post_b64)
    img, ext = generate_image("p", "http://h/v1", "m", "k")
    assert (img, ext) == (PNG, "png")
    url, payload, key = _fake_post_b64.calls[-1]
    assert payload["prompt"] == "p" and payload["model"] == "m"
    assert key == "k"


def test_FR_20_3_url_response(monkeypatch):
    monkeypatch.setattr(imagegen, "_post_json",
                        lambda u, p, k: {"data": [{"url": "http://h/i.jpg"}]})
    monkeypatch.setattr(imagegen, "_get_bytes", lambda u: PNG)
    img, ext = generate_image("p", "http://h/v1", "m")
    assert (img, ext) == (PNG, "jpg")


def test_FR_20_3_empty_response_raises(monkeypatch):
    monkeypatch.setattr(imagegen, "_post_json", lambda u, p, k: {"data": []})
    with pytest.raises(ImageGenError):
        generate_image("p", "http://h/v1", "m")


def test_FR_20_2_missing_model_exit2(capsys):
    assert main(["infer", "--demo", "--image-url", "http://h/v1",
                 "--device", "cpu"]) == 2
    assert "--image-model" in capsys.readouterr().err


def test_FR_20_2_bad_url_exit2(capsys):
    assert main(["infer", "--demo", "--image-url", "ftp://h",
                 "--image-model", "m", "--device", "cpu"]) == 2


def test_FR_20_4_cli_default_names(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(imagegen, "_post_json", _fake_post_b64)
    rc = main(["infer", "--demo", "--image-url", "http://h/v1",
               "--image-model", "m", "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr()
    img = tmp_path / "MysteryBeast_Qwen2_Demo_species_image.png"
    desc = tmp_path / "MysteryBeast_Qwen2_Demo_species_description.md"
    assert img.read_bytes() == PNG
    # FR-20.5: the description file equals the printed entry
    assert desc.read_text(encoding="utf-8").strip() == out.out.strip()
    assert "FIELD GUIDE ENTRY" in desc.read_text(encoding="utf-8")


def test_FR_20_4_cli_custom_names(tmp_path, monkeypatch):
    monkeypatch.setattr(imagegen, "_post_json", _fake_post_b64)
    img = tmp_path / "beast.png"
    desc = tmp_path / "beast.md"
    rc = main(["infer", "--demo", "--image-url", "http://h/v1",
               "--image-model", "m", "--save-image", str(img),
               "--save-description", str(desc), "--device", "cpu"])
    assert rc == 0
    assert img.read_bytes() == PNG
    assert "FIELD GUIDE ENTRY" in desc.read_text(encoding="utf-8")


def test_FR_20_3_api_failure_exit1(monkeypatch, capsys):
    def boom(url, payload, key):
        raise ImageGenError("boom")
    monkeypatch.setattr(imagegen, "_post_json", boom)
    rc = main(["infer", "--demo", "--image-url", "http://h/v1",
               "--image-model", "m", "--device", "cpu"])
    assert rc == 1
    assert "boom" in capsys.readouterr().err
