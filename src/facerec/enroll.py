"""Enrolment: learn the known people from a folder of photos, one subfolder per person."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np

from .detection import Detector
from .images import IMAGE_SUFFIXES, list_images, read_image


@dataclass
class Enrollment:
    faces: list[np.ndarray] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)  # unreadable photos and photos where no face was found

    @property
    def per_person(self) -> Counter:
        return Counter(self.names)


def collect_faces(folder: str | Path, detector: Detector, recognizer) -> Enrollment:
    """One face per photo: the most confident one, prepared for the recognizer. The folder name is the name.

    faces/
    ├── Ada/      ← about 20 photos, one clearly visible face in each
    └── Grace/
    """
    folder = Path(folder)
    people = sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
    if not people:
        raise ValueError(f"No person folder in {folder}: create one subfolder per person, named after them.")
    enrollment = Enrollment()
    for person in people:
        for photo in list_images(person):
            image = read_image(photo)
            boxes = detector.detect(image) if image is not None else []
            face = recognizer.prepare(image, boxes[0]) if boxes else None
            if face is None:  # the original notebook crashed here when a photo had no detectable face
                enrollment.skipped.append(photo)
                continue
            enrollment.faces.append(face)
            enrollment.names.append(person.name)
    return enrollment


def enroll(folder: str | Path, detector: Detector, recognizer):
    """Collect the faces of every person folder and train the recognizer on them."""
    enrollment = collect_faces(folder, detector, recognizer)
    return recognizer.train(enrollment.faces, enrollment.names), enrollment


@dataclass
class PhotoFaces:
    faces: list[np.ndarray] = field(default_factory=list)
    used: list[str] = field(default_factory=list)  # the photos the faces come from
    no_face: list[str] = field(default_factory=list)
    several_faces: list[str] = field(default_factory=list)  # used, with their clearest face: check them
    unreadable: list[str] = field(default_factory=list)


def faces_in_photos(photos: list[tuple[str, bytes]], detector: Detector, recognizer) -> PhotoFaces:
    """The face of each photo, prepared for the recognizer, in memory only: nothing is written anywhere.

    `photos` are (file name, content) pairs, as uploaded. The online demo learns the visitors' faces this way.
    """
    found = PhotoFaces()
    for filename, content in photos:
        image = read_image(BytesIO(content))
        if image is None:
            found.unreadable.append(filename)
            continue
        boxes = detector.detect(image)
        face = recognizer.prepare(image, boxes[0]) if boxes else None
        if face is None:
            found.no_face.append(filename)
            continue
        if len(boxes) > 1:
            found.several_faces.append(filename)
        found.faces.append(face)
        found.used.append(filename)
    return found


_FORBIDDEN = set('<>:"/\\|?*')  # characters a Windows folder name cannot hold (and path separators)
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def folder_name(name: str) -> str:
    """A person's name made safe as a folder name: no path separators or forbidden characters, no leading dot."""
    cleaned = "".join(c for c in name if c not in _FORBIDDEN and c.isprintable()).strip().strip(".").strip()
    if not cleaned or cleaned.upper() in _RESERVED:
        raise ValueError("Give the person a name (letters, digits, spaces).")
    return cleaned


@dataclass
class AddedPhotos:
    person: Path
    saved: list[str] = field(default_factory=list)
    no_face: list[str] = field(default_factory=list)  # not saved: nothing to learn from them
    several_faces: list[str] = field(default_factory=list)  # saved, but the clearest face will be used: check them
    unreadable: list[str] = field(default_factory=list)


def add_person(folder: str | Path, name: str, photos: list[tuple[str, bytes]], detector: Detector) -> AddedPhotos:
    """Save the photos of a person (new or already known) in `folder/<name>/`, keeping only those with a face.

    `photos` are (file name, content) pairs, as uploaded. The files are kept as they are (EXIF orientation too).
    """
    if not photos:
        raise ValueError("Add at least one photo of the person.")
    person = Path(folder) / folder_name(name)
    result = AddedPhotos(person)
    for filename, content in photos:
        image = read_image(BytesIO(content))
        if image is None:
            result.unreadable.append(filename)
            continue
        faces = detector.detect(image)
        if not faces:
            result.no_face.append(filename)
            continue
        if len(faces) > 1:
            result.several_faces.append(filename)
        person.mkdir(parents=True, exist_ok=True)
        target = _free_path(person, filename)
        target.write_bytes(content)
        result.saved.append(target.name)
    return result


def _free_path(folder: Path, filename: str) -> Path:
    """`folder/filename`, with _2, _3... added if a photo of that name is already there."""
    original = Path(filename)
    stem = "".join(c for c in original.stem if c not in _FORBIDDEN and c.isprintable()).strip() or "photo"
    suffix = original.suffix.lower() if original.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    path, n = folder / f"{stem}{suffix}", 2
    while path.exists():
        path, n = folder / f"{stem}_{n}{suffix}", n + 1
    return path
