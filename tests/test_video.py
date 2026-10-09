"""Videos: synthetic clips where Ada appears, then Grace, then a stranger, then nobody."""

import cv2
import numpy as np
import pytest
from conftest import FakeDetector, fake_recognizer, photo

import facerec.cli as cli
import facerec.detection
import facerec.recognizer
from facerec.enroll import enroll
from facerec.video import open_writer, process_video, timestamp

FPS = 10


def make_clip(path, frames):
    writer = open_writer(path, FPS, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def clip(tmp_path):
    """1 s of Ada, 1 s of Grace, 0.5 s of a stranger, 0.5 s of an empty room."""
    frames = ([photo(1, 100 + i) for i in range(10)] + [photo(2, 100 + i) for i in range(10)]
              + [photo(9, i) for i in range(5)] + [np.zeros_like(photo(1))] * 5)
    return make_clip(tmp_path / "clip.avi", frames)


@pytest.fixture
def recognizer(faces_folder):
    recognizer, _ = enroll(faces_folder, FakeDetector(), fake_recognizer())
    return recognizer


def test_who_appears_and_when(clip, recognizer, tmp_path):
    seen = []
    report = process_video(clip, FakeDetector(), recognizer, tmp_path / "named.avi",
                           progress=lambda done, total: seen.append((done, total)))
    assert (report.frames, report.analysed, report.fps) == (30, 30, FPS)
    assert report.duration == 3.0
    assert set(report.people) == {"Ada", "Grace"}
    ada, grace = report.people["Ada"], report.people["Grace"]
    assert (ada.first, ada.last, ada.frames) == (0.0, 0.9, 10)
    assert (grace.first, grace.last, grace.frames) == (1.0, 1.9, 10)
    assert report.unknown_frames == 5
    assert seen[-1] == (30, 30)
    written = cv2.VideoCapture(str(tmp_path / "named.avi"))
    assert int(written.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    written.release()


def test_analysing_one_frame_out_of_n_still_writes_every_frame(clip, recognizer, tmp_path):
    report = process_video(clip, FakeDetector(), recognizer, tmp_path / "named.avi", every=3, max_side=160)
    assert (report.frames, report.analysed) == (30, 10)
    written = cv2.VideoCapture(str(tmp_path / "named.avi"))
    assert int(written.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    assert written.get(cv2.CAP_PROP_FRAME_WIDTH) == 160  # shrunk to max_side
    written.release()


def test_videos_for_the_browser_are_written_in_webm(clip, recognizer, tmp_path):
    process_video(clip, FakeDetector(), recognizer, tmp_path / "named.webm")
    back = cv2.VideoCapture(str(tmp_path / "named.webm"))
    ok, _ = back.read()
    back.release()
    assert ok


def test_errors(tmp_path, recognizer):
    with pytest.raises(RuntimeError, match="Cannot open the video"):
        process_video(tmp_path / "missing.mp4", FakeDetector(), recognizer)
    with pytest.raises(ValueError, match=".mp4, .webm, .avi"):
        open_writer(tmp_path / "out.gif", FPS, (32, 32))


def test_timestamps():
    assert timestamp(0) == "0:00" and timestamp(65.4) == "1:05" and timestamp(600) == "10:00"


def test_cli_video_summary(clip, faces_folder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(facerec.detection, "load_detector", lambda *args, **kwargs: FakeDetector())
    monkeypatch.setattr(facerec.recognizer, "new_recognizer", lambda threshold=0.45: fake_recognizer(threshold))
    model = tmp_path / "faces.npz"
    cli.main(["enroll", str(faces_folder), "--model", str(model)])
    capsys.readouterr()
    cli.main(["video", str(clip), "--model", str(model), "--save", str(tmp_path / "out.mp4")])
    out = capsys.readouterr().out
    assert "30 frames (0:03)" in out
    assert "Ada                  0:00 -> 0:00   on 10 analysed frames" in out
    assert "Grace                0:01 -> 0:01   on 10 analysed frames" in out
    assert "Unknown              on 5 analysed frames" in out
    assert (tmp_path / "out.mp4").exists()
