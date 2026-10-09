"""With the real models and LFW (set FACEREC_RUN_SLOW=1 after `facerec download` and `facerec evaluate` once)."""

import os

import numpy as np
import pytest

from facerec.detection import YUNET_MODEL, load_detector
from facerec.evaluate import load_index
from facerec.images import read_image
from facerec.recognizer import SFACE_MODEL, new_recognizer

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("FACEREC_RUN_SLOW") != "1", reason="set FACEREC_RUN_SLOW=1 to run"),
]


@pytest.fixture(scope="module")
def lfw():
    from facerec.evaluate import lfw_folder

    if not (YUNET_MODEL.exists() and SFACE_MODEL.exists()):
        pytest.skip("run `facerec download` first")
    return load_index(lfw_folder(download=False))


def face(detector, recognizer, path):
    image = read_image(path)
    return recognizer.prepare(image, detector.detect(image)[0])


def test_real_models_recognise_lfw_people_and_reject_a_stranger(lfw):
    detector, recognizer = load_detector(), new_recognizer()
    people = ["Colin_Powell", "Serena_Williams"]
    enrol = [(face(detector, recognizer, p), name) for name in people for p in lfw[name][:10]]
    recognizer.train([f for f, _ in enrol], [n for _, n in enrol])
    for name in people:
        assert [recognizer.predict(face(detector, recognizer, p)).name for p in lfw[name][10:15]] == [name] * 5
    stranger = recognizer.predict(face(detector, recognizer, lfw["Gerhard_Schroeder"][0]))
    assert stranger.name is None


def test_the_embedding_is_a_unit_vector_of_128_numbers(lfw):
    detector, recognizer = load_detector(), new_recognizer()
    vector = face(detector, recognizer, lfw["Colin_Powell"][0])
    assert vector.shape == (128,) and np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
