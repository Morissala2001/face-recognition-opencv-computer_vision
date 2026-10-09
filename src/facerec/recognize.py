"""Recognising every face of a picture, drawing the result, and the live webcam loop."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import open_camera
from .detection import Box, Detector

UNKNOWN = "Unknown"
KNOWN_COLOR = (200, 90, 20)  # BGR: blue
UNKNOWN_COLOR = (40, 40, 200)  # BGR: red


@dataclass(frozen=True)
class Recognition:
    box: Box
    name: str | None  # None: Unknown
    distance: float

    @property
    def label(self) -> str:
        distance = f"{self.distance:.2f}" if self.distance < 2 else f"{self.distance:.0f}"  # SFace or LBPH scale
        return f"{self.name or UNKNOWN}  {distance}"


def recognize(image: np.ndarray, detector: Detector, recognizer) -> list[Recognition]:
    """Every face found in the picture, with a name or Unknown."""
    results = []
    for box in detector.detect(image):
        face = recognizer.prepare(image, box)
        if face is not None:
            match = recognizer.predict(face)
            results.append(Recognition(box, match.name, match.distance))
    return results


def draw(image: np.ndarray, recognitions: list[Recognition]) -> np.ndarray:
    """A copy of the picture with a box around each face and its name (and distance) in a banner below it."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = max(0.5, max(height, width) / 1000)  # readable on a 640 px webcam frame and on a 4000 px photo
    thickness = max(1, round(2 * scale))
    for r in recognitions:
        color = KNOWN_COLOR if r.name else UNKNOWN_COLOR
        b = r.box
        cv2.rectangle(canvas, (b.left, b.top), (b.right, b.bottom), color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(r.label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        banner = text_height + baseline + 2 * thickness
        top = b.bottom if b.bottom + banner <= height else max(b.top - banner, 0)  # below the face, or above if no room
        right = max(b.right, b.left + text_width + 4 * thickness)
        cv2.rectangle(canvas, (b.left, top), (right, top + banner), color, -1)
        cv2.putText(canvas, r.label, (b.left + 2 * thickness, top + text_height + thickness),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def run_webcam(detector: Detector, recognizer, camera: int | str = 0,
               window: str = "Face recognition - press q to quit") -> int:
    """Recognise faces live until `q` (or `x`) is pressed. Returns the number of frames shown."""
    capture = open_camera(camera)
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # Shown at every frame, with or without a face (the original showed it only when a face was found).
            cv2.imshow(window, draw(frame, recognize(frame, detector, recognizer)))
            frames += 1
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("x")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return frames
