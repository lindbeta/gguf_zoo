import os
import tempfile

# Force deterministic CPU-trained artifacts for the whole test session.
os.environ["GGUF_ZOO_ARTIFACT"] = os.path.join(
    tempfile.gettempdir(), "ggufzoo_test_clf.pt")
os.environ["GGUF_ZOO_ATTR_ARTIFACT"] = os.path.join(
    tempfile.gettempdir(), "ggufzoo_test_attr_clf.pt")

import pytest  # noqa: E402

from gguf_zoo.attributes import train_attribute_classifier  # noqa: E402
from gguf_zoo.classifier import train_classifier  # noqa: E402
from gguf_zoo.demo import build_gguf, demo_pairs  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    train_classifier(seed=0, device="cpu", out_path=os.environ["GGUF_ZOO_ARTIFACT"])
    train_attribute_classifier(seed=0, device="cpu",
                               out_path=os.environ["GGUF_ZOO_ATTR_ARTIFACT"])
    return os.environ["GGUF_ZOO_ARTIFACT"]


@pytest.fixture
def demo_kv():
    from gguf_zoo.parser import parse_gguf_bytes
    kv, _ = parse_gguf_bytes(build_gguf(demo_pairs()))
    return kv


@pytest.fixture
def demo_bytes():
    return build_gguf(demo_pairs())
