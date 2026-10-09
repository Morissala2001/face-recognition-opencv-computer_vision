"""Cameras: one of the computer's, or a phone's camera.

A phone usually takes much sharper pictures than a laptop webcam. Two ways to use it:
- iVCam (iPhone, Android) and its computer client turn the phone into a camera of the computer: a number (often 1);
- DroidCam (Android, iPhone) streams it on the local Wi-Fi at http://<phone IP>:4747/video, read like any camera.
"""

from __future__ import annotations

import cv2
import numpy as np

HD = (1920, 1080)  # asked from computer cameras: OpenCV would otherwise use 640 x 480, wasting a phone's sharpness


def camera_source(value: str) -> int | str:
    """'0', '1'... is a camera of the computer (iVCam's phone is one of them); anything else is a video address,
    such as DroidCam's stream on the same Wi-Fi (http://<phone IP>:4747/video)."""
    return int(value) if str(value).strip().isdigit() else str(value).strip()


def open_camera(source: int | str, timeout_ms: int = 5000) -> cv2.VideoCapture:
    """Open a computer camera or a video address; an unreachable phone fails after `timeout_ms`, not ~30 s."""
    if isinstance(source, int):
        capture = cv2.VideoCapture(source)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, HD[0])  # the camera keeps its closest supported size
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HD[1])
    else:
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                                                            cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])
    if not capture.isOpened():
        capture.release()
        hint = ("try another camera number (--camera 1, 2...)" if isinstance(source, int) else
                "check that DroidCam is open on the phone, that no other program is connected to it, and that "
                "the phone and this computer share the same Wi-Fi")
        raise RuntimeError(f"Camera {source} cannot be opened: {hint}")
    return capture


def grab_frame(source: int | str) -> np.ndarray:
    """One picture from a camera or a phone's video stream."""
    capture = open_camera(source)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"No picture received from {source}.")
    return frame


def computer_cameras(count: int = 4) -> list[tuple[int, np.ndarray]]:
    """A picture from each camera of the computer that answers, to tell them apart (webcam, iVCam's phone...)."""
    found = []
    for index in range(count):
        try:
            found.append((index, grab_frame(index)))
        except RuntimeError:
            continue
    return found
