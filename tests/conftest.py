"""Shared fixtures. No real face and no model file: each 'person' is a synthetic texture, a fake detector finds
it and a fake network stands in for SFace.

Real faces are personal data: they never enter the repository, not even as test files.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from facerec.detection import Box
from facerec.recognizer import SFaceRecognizer

# Where the synthetic face sits in every synthetic photo, with a YuNet-like raw row (box, 5 landmarks, score).
FACE_BOX = Box(40, 30, 240, 230, 0.99, (40.0, 30.0, 200.0, 200.0) + (0.0,) * 10 + (0.99,))


def texture(person: int) -> np.ndarray:
    """A person's 'face': shapes drawn at positions of their own, the same every time for the same person.

    LBPH compares local texture histograms cell by cell, so people must differ in spatial structure: random
    noise alone gives the same histograms for everyone.
    """
    rng = np.random.default_rng(person)
    face = cv2.resize(rng.integers(40, 216, (6, 6)).astype(np.uint8), (200, 200), interpolation=cv2.INTER_CUBIC)
    for _ in range(12):
        center = tuple(int(v) for v in rng.integers(10, 190, 2))
        cv2.circle(face, center, int(rng.integers(5, 30)), int(rng.integers(0, 256)), int(rng.choice([-1, 2, 4])))
    for _ in range(6):
        start, end = (tuple(int(v) for v in rng.integers(0, 200, 2)) for _ in range(2))
        cv2.line(face, start, end, int(rng.integers(0, 256)), int(rng.integers(1, 5)))
    return cv2.GaussianBlur(face, (0, 0), 1.5)


def photo(person: int, shot: int = 0, size=(280, 320)) -> np.ndarray:
    """A BGR photo of `person`: their texture in FACE_BOX with some noise and lighting change."""
    rng = np.random.default_rng(1000 * person + shot)
    canvas = np.full((size[0], size[1]), 128, np.uint8)
    face = texture(person).astype(np.int16) + rng.normal(0, 6, (200, 200)) + rng.integers(-15, 15)
    canvas[FACE_BOX.top:FACE_BOX.bottom, FACE_BOX.left:FACE_BOX.right] = np.clip(face, 0, 255)
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


class FakeDetector:
    """Finds the synthetic face, except in an all-black picture (a photo without any face)."""

    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [] if image.max() == 0 else [FACE_BOX]


class FakeNet:
    """Stands in for SFace: 'aligns' by cropping the box; the 128 'features' are the centred pixels of a thumbnail."""

    def alignCrop(self, image, row):
        x, y, w, h = (int(v) for v in row[:4])
        return cv2.resize(image[y:y + h, x:x + w], (112, 112))

    def feature(self, aligned):
        grey = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        thumbnail = cv2.resize(grey, (16, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        return (thumbnail - thumbnail.mean())[None, :]


def fake_recognizer(threshold=0.45) -> SFaceRecognizer:
    return SFaceRecognizer(threshold, net=FakeNet())


def save(image_bgr: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).save(path)
    return path


@pytest.fixture
def faces_folder(tmp_path):
    """faces/Ada (6 photos + one without a face + a text file), faces/Grace (6 photos)."""
    folder = tmp_path / "faces"
    for person, name in ((1, "Ada"), (2, "Grace")):
        for shot in range(6):
            save(photo(person, shot), folder / name / f"{shot}.png")
    save(np.zeros((100, 100, 3), np.uint8), folder / "Ada" / "black.png")
    (folder / "Ada" / "notes.txt").write_text("not a photo")
    return folder


@pytest.fixture
def detector():
    return FakeDetector()
