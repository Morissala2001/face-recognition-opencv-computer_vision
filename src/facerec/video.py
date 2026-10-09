"""Videos: recognise the faces frame by frame, write an annotated copy and tell who appears, and when."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .camera import open_camera
from .detection import Detector
from .recognize import draw, recognize

# Codecs tried in order for each file type. H.264 ('avc1') plays everywhere but is not always available
# (on Windows it goes through Media Foundation); VP8 in WebM plays in every browser and works on any system.
CODECS = {".mp4": ["avc1", "mp4v"], ".webm": ["VP80"], ".avi": ["MJPG"]}


@dataclass
class Sighting:
    """When a person was seen: first and last time (in seconds) and on how many analysed frames."""

    first: float
    last: float
    frames: int = 1


@dataclass
class VideoReport:
    frames: int = 0  # frames read
    analysed: int = 0  # frames where faces were recognised (every `every`-th frame)
    fps: float = 25.0
    people: dict[str, Sighting] = field(default_factory=dict)
    unknown_frames: int = 0  # analysed frames with at least one Unknown face
    output: Path | None = None

    @property
    def duration(self) -> float:
        return self.frames / self.fps


def open_writer(path: str | Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """A video writer for `path`, with the first codec of its file type that this OpenCV build can use."""
    path = Path(path)
    codecs = CODECS.get(path.suffix.lower())
    if codecs is None:
        raise ValueError(f"Cannot write {path.suffix} videos: use one of {', '.join(CODECS)}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in codecs:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"No codec available to write {path.suffix} videos with this OpenCV build.")


def fit(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    height, width = frame.shape[:2]
    if not max_side or max(height, width) <= max_side:
        return frame
    scale = max_side / max(height, width)
    return cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def process_video(source: str | Path, detector: Detector, recognizer, output: str | Path | None = None,
                  every: int = 1, max_side: int | None = None, max_seconds: float | None = None,
                  progress: Callable[[int, int], None] | None = None) -> VideoReport:
    """Recognise the faces of every `every`-th frame and, if `output` is given, write the annotated video.

    `source` is a video file or a stream address (rtsp://, http://...; a live stream needs `max_seconds`, it has
    no end). Frames in between keep the last names drawn, so the annotated video stays smooth when `every` > 1.
    `max_side` shrinks the written frames (detection always uses the full picture).
    """
    if "://" in str(source):
        capture = open_camera(str(source))  # with a time limit if the stream does not answer
    else:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open the video {source}.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    report = VideoReport(fps=fps if 0 < fps < 1000 else 25.0, output=Path(output) if output else None)
    total = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)  # 0 for a live stream: unknown
    if max_seconds:
        limit = round(max_seconds * report.fps)
        total = min(total, limit) if total else limit
    writer, results = None, []
    try:
        while not (max_seconds and report.frames >= total):
            ok, frame = capture.read()
            if not ok:
                break
            if report.frames % max(every, 1) == 0:
                results = recognize(frame, detector, recognizer)
                report.analysed += 1
                seconds = report.frames / report.fps
                for name in {r.name for r in results if r.name}:
                    seen = report.people.get(name)
                    if seen is None:
                        report.people[name] = Sighting(seconds, seconds)
                    else:
                        seen.last, seen.frames = seconds, seen.frames + 1
                report.unknown_frames += any(r.name is None for r in results)
            if output:
                annotated = fit(draw(frame, results), max_side)
                if writer is None:
                    writer = open_writer(output, report.fps, (annotated.shape[1], annotated.shape[0]))
                writer.write(annotated)
            report.frames += 1
            if progress:
                progress(report.frames, total)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return report


def timestamp(seconds: float) -> str:
    """'1:05' for 65 seconds."""
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
