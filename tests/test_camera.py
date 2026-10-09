"""Cameras of the computer and phone cameras streamed over Wi-Fi (a local MJPEG server plays the phone)."""

import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest
from PIL import Image

from facerec.camera import camera_source, grab_frame, open_camera


def test_numbers_are_computer_cameras_and_anything_else_an_address():
    assert camera_source("0") == 0 and camera_source(" 2 ") == 2
    assert camera_source(" http://192.168.1.20:4747/video ") == "http://192.168.1.20:4747/video"


@pytest.fixture
def phone():
    """An HTTP server streaming JPEG frames like DroidCam: a red square on black."""
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[100:300, 200:400] = (0, 0, 255)
    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(buffer, "JPEG")
    jpeg = buffer.getvalue()

    class Stream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n"
                                     % (len(jpeg), jpeg))
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/video"
    server.shutdown()


def test_a_picture_is_grabbed_from_a_phone_stream(phone):
    frame = grab_frame(camera_source(phone))
    assert frame.shape == (480, 640, 3)
    assert frame[200, 300, 2] > 240 and frame[200, 300, :2].max() < 20  # the red square, in BGR order


def test_a_live_stream_is_recorded_for_a_given_time(phone, tmp_path):
    from conftest import FakeDetector, fake_recognizer

    from facerec.video import process_video

    report = process_video(phone, FakeDetector(), fake_recognizer(), tmp_path / "live.avi", max_seconds=0.4)
    assert report.frames == round(0.4 * report.fps)  # a live stream has no end: stopped by the time limit
    assert (tmp_path / "live.avi").stat().st_size > 0


class ClosedCamera:
    def __init__(self, *args):
        self.released = False

    def isOpened(self):
        return False

    def set(self, prop, value):
        return True

    def release(self):
        self.released = True


def test_errors_say_what_to_check(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", ClosedCamera)
    with pytest.raises(RuntimeError, match="--camera 1, 2"):
        open_camera(3)
    with pytest.raises(RuntimeError, match="same Wi-Fi"):
        open_camera("http://192.168.1.20:4747/video")


def test_a_stream_that_sends_nothing_is_reported(monkeypatch):
    class Silent(ClosedCamera):
        def isOpened(self):
            return True

        def read(self):
            return False, None

    monkeypatch.setattr(cv2, "VideoCapture", Silent)
    with pytest.raises(RuntimeError, match="No picture received"):
        grab_frame("http://192.168.1.20:4747/video")
