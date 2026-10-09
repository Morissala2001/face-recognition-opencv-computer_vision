"""Face recognition with SFace: whose face is it, or is it someone unknown?

A neural network turns each aligned face into 128 numbers. Two photos of the same person give vectors pointing
in almost the same direction, so faces are compared with the cosine similarity. Distance = 1 - cosine: 0 for
identical vectors, larger for different people. A face is named after the closest enrolled face when their
distance is at most `threshold`, otherwise it is Unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detection import MODELS, Box, require

SFACE_MODEL = MODELS / "face_recognition_sface_2021dec.onnx"
SFACE_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/"
             "face_recognition_sface_2021dec.onnx")
DEFAULT_MODEL = MODELS / "faces.npz"  # the faces you enrolled: personal data, never committed
DEFAULT_THRESHOLD = 0.45  # chosen on LFW, see the README
OPENCV_THRESHOLD = 1 - 0.363  # OpenCV's documented cosine threshold for SFace, as a distance
THRESHOLD_RANGE = (0.0, 1.2, 0.005)  # for sliders and evaluation sweeps


@dataclass(frozen=True)
class Match:
    name: str | None  # None means "Unknown": the closest known face is still too far
    distance: float  # to the closest enrolled face: the lower, the more alike
    closest: str  # the closest known person, even when too far to be named


class SFaceRecognizer:
    """SFace (OpenCV model zoo, Apache-2.0) and the nearest enrolled face."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, model: str | Path = SFACE_MODEL, net=None):
        self.threshold = threshold
        self.net = net if net is not None else cv2.FaceRecognizerSF.create(str(require(model)), "")
        self.embeddings = np.zeros((0, 128), np.float32)
        self.row_names: list[str] = []  # the person of each enrolled face

    @property
    def names(self) -> list[str]:
        return sorted(set(self.row_names))

    def fresh(self) -> SFaceRecognizer:
        """An empty recognizer sharing the same network (loading it is the slow part)."""
        return SFaceRecognizer(self.threshold, net=self.net)

    def prepare(self, image: np.ndarray, box: Box) -> np.ndarray | None:
        """Align the face on its landmarks (eyes level, fixed size), then turn it into a unit vector."""
        if box.raw is None:
            raise ValueError("The face must come from YuNet: its landmarks are needed to align it.")
        feature = self.net.feature(self.net.alignCrop(image, np.array(box.raw, np.float32))).ravel()
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else None

    def train(self, faces: list[np.ndarray], names: list[str]) -> SFaceRecognizer:
        """'Training' is only remembering the vectors: no network weight changes, so enrolling takes seconds."""
        if not faces:
            raise ValueError("No face to learn from.")
        self.embeddings = np.stack(faces).astype(np.float32)
        self.row_names = list(names)
        return self

    def predict(self, face: np.ndarray) -> Match:
        if len(self.embeddings) == 0:  # nobody enrolled: everybody is Unknown
            return Match(None, 2.0, "")
        similarities = self.embeddings @ face
        best = int(np.argmax(similarities))
        distance = 1.0 - float(similarities[best])
        closest = self.row_names[best]
        return Match(closest if distance <= self.threshold else None, distance, closest)

    def save(self, path: str | Path = DEFAULT_MODEL) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:  # a file object: np.savez would otherwise append ".npz" to other names
            np.savez(file, embeddings=self.embeddings, names=np.array(self.row_names))
        return path

    def load(self, path: str | Path = DEFAULT_MODEL) -> SFaceRecognizer:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No enrolled faces at {path}. Run `facerec enroll <faces folder>` first.")
        with np.load(path) as data:
            self.embeddings, self.row_names = data["embeddings"], data["names"].tolist()
        return self


def new_recognizer(threshold: float = DEFAULT_THRESHOLD) -> SFaceRecognizer:
    """An empty recognizer (one place to replace it, for instance by one with a fake network in the tests)."""
    return SFaceRecognizer(threshold)
