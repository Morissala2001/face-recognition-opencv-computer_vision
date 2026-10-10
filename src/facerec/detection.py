"""Face detection with YuNet: where are the faces in a picture, and where are their eyes, nose and mouth?"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

MODELS = Path("models")
YUNET_MODEL = MODELS / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


@dataclass(frozen=True)
class Box:
    """A face, in pixels of the picture: left, top, right, bottom, and the detector's confidence.

    `raw` keeps YuNet's own output row (box, 5 landmarks: eyes, nose tip, mouth corners, score): the recognizer
    needs the landmarks to align the face before reading it.
    """

    left: int
    top: int
    right: int
    bottom: int
    score: float = 1.0
    raw: tuple[float, ...] | None = None

    @property
    def area(self) -> int:
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Box]:
        """The faces of a BGR picture (as OpenCV reads it), most confident first."""
        ...


def clip(box: Box, width: int, height: int) -> Box:
    """Keep a box inside the picture, so that a face at the edge of the frame gives a valid drawing."""
    return Box(min(max(box.left, 0), width), min(max(box.top, 0), height),
               min(max(box.right, 0), width), min(max(box.bottom, 0), height), box.score, box.raw)


def require(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file at {path}. Run `facerec download` first.")
    return path


class YuNetDetector:
    """YuNet, a tiny face detector (0.2 MB) from OpenCV's model zoo (MIT license).

    Large pictures are shrunk to `max_side` pixels before detection: YuNet is trained on faces of moderate size
    and misses a face that fills a 2,500-pixel phone photo. Boxes and landmarks are scaled back to the original.
    """

    def __init__(self, model: str | Path = YUNET_MODEL, min_score: float = 0.6, max_side: int = 640):
        self.net = cv2.FaceDetectorYN.create(str(require(model)), "", (320, 320), min_score, 0.3, 5000)
        self.max_side = max_side

    def detect(self, image: np.ndarray) -> list[Box]:
        height, width = image.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        small = image if scale == 1.0 else cv2.resize(image, (round(width * scale), round(height * scale)),
                                                       interpolation=cv2.INTER_AREA)
        self.net.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.net.detect(small)
        boxes = []
        for row in [] if faces is None else faces:
            row = np.concatenate([row[:14] / scale, row[14:]])  # back to the original picture's pixels
            x, y, w, h = row[:4]
            box = clip(Box(round(x), round(y), round(x + w), round(y + h), float(row[14]), tuple(map(float, row))),
                       width, height)
            if box.area > 0:
                boxes.append(box)
        return sorted(boxes, key=lambda b: b.score, reverse=True)


def load_detector(model: str | Path = YUNET_MODEL) -> Detector:
    """The face detector (one place to replace it, for instance by a fake one in the tests)."""
    return YuNetDetector(model)


def download(url: str, path: str | Path) -> Path:
    """Download a model file once. It appears only when complete: an interrupted download leaves no broken model."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.{os.getpid()}.part")
        try:
            urllib.request.urlretrieve(url, part)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(path)
    return path
