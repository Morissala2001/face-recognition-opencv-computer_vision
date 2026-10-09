"""Reading pictures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# iPhones save their photos as HEIC by default: let Pillow open them like any other picture
register_heif_opener()

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def read_image(source) -> np.ndarray | None:
    """A picture as a BGR array (OpenCV's layout), or None if it cannot be read.

    Accepts a path, a file object or a PIL image, including iPhone HEIC photos. Phone photos are turned upright
    using their EXIF orientation (HEIC ones are already upright when decoded), and paths with accents work on
    Windows too (`cv2.imread` fails on them).
    """
    try:
        image = source if isinstance(source, Image.Image) else Image.open(source)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError):
        return None
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
