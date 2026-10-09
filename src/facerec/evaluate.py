"""Measuring the recognizer on LFW (Labeled Faces in the Wild), a public set of 13,233 photos of 5,749 people.

The question is the one asked at home: a few people are enrolled with about 20 photos each, then faces appear.
Are the known people named correctly, and are the others (never enrolled) called Unknown?

One scenario: 5 people drawn at random among those with at least 25 photos; 20 photos each to enrol, up to 10
others to test; 200 faces of people never enrolled. The 'Unknown' threshold is chosen on 10 scenarios and
reported on 10 other ones, so the threshold is never tuned on the faces it is measured on. The same seed always
gives the same scenarios.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .detection import Detector
from .images import read_image
from .recognizer import OPENCV_THRESHOLD, THRESHOLD_RANGE

ENROLL_PHOTOS = 20  # what the original tutorial advises: about 20 photos per person
MAX_TEST_PHOTOS = 10
MIN_TEST_PHOTOS = 5


def lfw_folder(download: bool = True) -> Path:
    """The LFW photos (funneled version, about 230 MB), downloaded once by scikit-learn into its data folder."""
    from sklearn.datasets import fetch_lfw_people, get_data_home

    home = Path(get_data_home()) / "lfw_home"
    folder = home / "lfw_funneled"
    if not folder.exists():
        if not download:
            raise FileNotFoundError(f"LFW is not in {home}")
        fetch_lfw_people(min_faces_per_person=70, download_if_missing=True)  # small load: downloads and extracts
    return folder


def load_index(folder: str | Path) -> dict[str, list[Path]]:
    """Person name -> their photos."""
    return {p.name: sorted(p.glob("*.jpg")) for p in sorted(Path(folder).iterdir()) if p.is_dir()}


class FaceCache:
    """Detects and prepares each photo once: the same photo appears in several scenarios."""

    def __init__(self, detector: Detector, recognizer):
        self.detector, self.recognizer = detector, recognizer
        self.faces: dict[Path, np.ndarray | None] = {}
        self.seconds = 0.0

    def __call__(self, path: Path) -> np.ndarray | None:
        if path not in self.faces:
            start = time.perf_counter()
            image = read_image(path)
            boxes = self.detector.detect(image) if image is not None else []
            self.faces[path] = self.recognizer.prepare(image, boxes[0]) if boxes else None
            self.seconds += time.perf_counter() - start
        return self.faces[path]


@dataclass
class Scenario:
    enrol: list[tuple[Path, str]]
    known: list[tuple[Path, str]]  # test photos of enrolled people
    unknown: list[Path]  # photos of people never enrolled


@dataclass
class Outcome:
    """Raw results of one scenario; any threshold can be applied afterwards."""

    known: list[tuple[str, str, float]] = field(default_factory=list)  # true name, closest name, distance
    unknown: list[float] = field(default_factory=list)  # distance to the closest enrolled face
    missed: int = 0  # test photos where no face was detected


def draw_scenario(index: dict[str, list[Path]], rng: np.random.Generator, people: int = 5,
                  unknown: int = 200) -> Scenario:
    candidates = [name for name, photos in index.items() if len(photos) >= ENROLL_PHOTOS + MIN_TEST_PHOTOS]
    others = [name for name, photos in index.items() if len(photos) < ENROLL_PHOTOS + MIN_TEST_PHOTOS]
    enrol, known = [], []
    for name in rng.choice(candidates, size=people, replace=False):
        photos = list(rng.permutation(index[name]))
        enrol += [(Path(p), name) for p in photos[:ENROLL_PHOTOS]]
        known += [(Path(p), name) for p in photos[ENROLL_PHOTOS:ENROLL_PHOTOS + MAX_TEST_PHOTOS]]
    strangers = rng.choice(others, size=unknown, replace=False)  # one photo per stranger
    return Scenario(enrol, known, [Path(rng.choice(index[name])) for name in strangers])


def run_scenario(scenario: Scenario, faces: FaceCache) -> Outcome:
    pairs = [(faces(path), name) for path, name in scenario.enrol]
    pairs = [(face, name) for face, name in pairs if face is not None]
    recognizer = faces.recognizer.fresh().train([f for f, _ in pairs], [n for _, n in pairs])
    outcome = Outcome()
    for path, name in scenario.known:
        face = faces(path)
        if face is None:
            outcome.missed += 1
            continue
        match = recognizer.predict(face)
        outcome.known.append((name, match.closest, match.distance))
    for path in scenario.unknown:
        face = faces(path)
        if face is None:
            outcome.missed += 1
            continue
        outcome.unknown.append(recognizer.predict(face).distance)
    return outcome


def rates(outcomes: list[Outcome], threshold: float) -> dict[str, float]:
    """Shares over all the outcomes: known faces named right / named wrong / called Unknown, strangers rejected."""
    known = [(true, closest, d) for o in outcomes for true, closest, d in o.known]
    unknown = np.array([d for o in outcomes for d in o.unknown])
    right = np.mean([closest == true and d <= threshold for true, closest, d in known])
    wrong = np.mean([closest != true and d <= threshold for true, closest, d in known])
    rejected = np.mean(unknown > threshold)
    return {"threshold": round(float(threshold), 4), "known_named_right": float(right),
            "known_named_wrong": float(wrong), "known_called_unknown": float(1 - right - wrong),
            "strangers_rejected": float(rejected), "balanced_accuracy": float((right + rejected) / 2),
            "closed_set_accuracy": float(np.mean([closest == true for true, closest, _ in known]))}


def best_threshold(outcomes: list[Outcome], thresholds) -> float:
    """The threshold with the best balance between naming known people and rejecting strangers."""
    return float(max(thresholds, key=lambda t: rates(outcomes, t)["balanced_accuracy"]))


def evaluate(index: dict[str, list[Path]], detector: Detector, recognizer, people: int = 5, repeats: int = 20,
             unknown: int = 200, seed: int = 0) -> dict:
    """Calibrate the threshold on the first half of the scenarios, report on the second half."""
    faces = FaceCache(detector, recognizer)
    rng = np.random.default_rng(seed)
    outcomes = [run_scenario(draw_scenario(index, rng, people, unknown), faces) for _ in range(repeats)]
    calibration, test = outcomes[:repeats // 2], outcomes[repeats // 2:]
    grid = np.arange(*THRESHOLD_RANGE)
    chosen = best_threshold(calibration, grid)
    detected = [f is not None for f in faces.faces.values()]
    return {
        "people": people, "repeats": repeats, "unknown_per_scenario": unknown,
        "photos_processed": len(detected), "no_face_detected": 1 - float(np.mean(detected)),
        "ms_per_photo": 1000 * faces.seconds / max(len(detected), 1),
        "chosen_threshold": chosen,
        "test": rates(test, chosen),
        "test_with_opencv_threshold": rates(test, OPENCV_THRESHOLD),
        "sweep": [rates(test, t) for t in grid[:: max(1, len(grid) // 12)]],
    }
