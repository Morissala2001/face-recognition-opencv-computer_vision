"""The web app guides the user from nothing to enrolled faces (fake detector and network, synthetic faces)."""

from pathlib import Path

import pytest
from conftest import FakeDetector, fake_recognizer

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import facerec.detection  # noqa: E402
import facerec.recognizer  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    def make_app(faces=None, models=True, public=False):
        st.cache_resource.clear()
        if public:
            monkeypatch.setenv("FACEREC_PUBLIC", "1")
        for name in ("YUNET", "SFACE"):
            path = tmp_path / f"{name.lower()}.onnx"
            if models:
                path.write_bytes(b"fake")
            monkeypatch.setenv(f"FACEREC_{name}", str(path))
        monkeypatch.setenv("FACEREC_MODEL", str(tmp_path / "models" / "faces.npz"))
        monkeypatch.setenv("FACEREC_FACES", str(faces or tmp_path / "no_faces"))
        monkeypatch.setattr(facerec.detection, "load_detector", lambda *args, **kwargs: FakeDetector())
        monkeypatch.setattr(facerec.recognizer, "new_recognizer", lambda *args, **kwargs: fake_recognizer())
        return AppTest.from_file(str(APP), default_timeout=60).run()

    return make_app


def test_without_the_models_the_app_says_how_to_get_them(make_app):
    app = make_app(models=False)
    assert not app.exception
    assert "Missing model files" in app.error[0].value and "facerec download" in app.code[0].value


def test_without_any_photo_the_app_explains_how_to_start(make_app):
    app = make_app()
    assert "Nobody has been learnt yet" in app.warning[0].value


def test_the_faces_folder_is_learnt_with_one_click(make_app, faces_folder, tmp_path):
    app = make_app(faces_folder)
    assert "Ada, Grace" in app.info[0].value
    next(b for b in app.button if b.label == "Learn the faces").click().run()
    assert not app.exception
    assert (tmp_path / "models" / "faces.npz").exists()
    assert "Ada (6 faces), Grace (6 faces)" in app.sidebar.caption[0].value
    assert "black.png" in app.sidebar.caption[1].value
    assert app.sidebar.caption[-1].value == "Ada, Grace"  # the people it knows
    assert "Drop a photo" in app.info[0].value


def test_a_picture_from_the_phone_is_recognised(make_app, faces_folder, monkeypatch):
    import facerec.camera
    from conftest import photo

    addresses = []
    monkeypatch.setattr(facerec.camera, "grab_frame", lambda source: addresses.append(source) or photo(2, 50))
    app = make_app(faces_folder)
    next(b for b in app.button if b.label == "Learn the faces").click().run()
    app.button_group[0].set_value("Phone or stream").run()
    take = next(b for b in app.button if b.label == "Take a picture")
    assert take.disabled  # no address yet
    app.text_input[0].set_value(" 1 ").run()  # iVCam: the phone is camera 1 of the computer
    next(b for b in app.button if b.label == "Take a picture").click().run()
    assert not app.exception
    assert addresses == [1]
    assert app.dataframe[0].value["Name"].tolist() == ["Grace"]
    app.sidebar.slider[0].set_value(0.0).run()  # the picture is kept while the threshold changes
    assert app.dataframe[0].value["Name"].tolist() == ["Unknown"]
    app.text_input[0].set_value("http://192.168.1.20:4747/video").run()  # DroidCam: an address
    next(b for b in app.button if b.label == "Take a picture").click().run()
    assert addresses[-1] == "http://192.168.1.20:4747/video"


def test_a_live_stream_is_watched_until_it_ends_and_the_camera_released(make_app, faces_folder, monkeypatch):
    import numpy as np

    import facerec.camera
    from conftest import photo

    class Stream:
        def __init__(self):
            self.frames, self.released = [photo(1, 60), np.zeros((280, 320, 3), np.uint8), photo(2, 60)], False

        def read(self):
            return (True, self.frames.pop(0)) if self.frames else (False, None)

        def release(self):
            self.released = True

    streams = []
    monkeypatch.setattr(facerec.camera, "open_camera", lambda source: streams.append(Stream()) or streams[-1])
    app = make_app(faces_folder)
    next(b for b in app.button if b.label == "Learn the faces").click().run()
    app.button_group[0].set_value("Phone or stream").run()
    app.text_input[0].set_value("rtsp://192.168.1.30:554/stream").run()
    next(b for b in app.button if b.label == "Watch live").click().run()
    assert not app.exception
    assert [c.value for c in app.main.caption if c.value.startswith("In front")] == ["In front of the camera: Grace"]
    assert "Live view ended after 3 frames" in app.info[0].value
    assert streams[0].released


def test_the_cameras_of_the_computer_are_previewed(make_app, faces_folder, monkeypatch):
    import facerec.camera
    from conftest import photo

    def grab(index):
        if index == 2:
            raise RuntimeError("no such camera")
        return photo(index + 1)

    monkeypatch.setattr(facerec.camera, "grab_frame", grab)
    app = make_app(faces_folder)
    next(b for b in app.button if b.label == "Learn the faces").click().run()
    app.button_group[0].set_value("Phone or stream").run()
    next(b for b in app.button if b.label == "Show the cameras of this computer").click().run()
    assert not app.exception
    captions = [caption for image in app.get("image") for caption in image.captions]
    assert captions == ["Camera 0 (320 x 280)", "Camera 1 (320 x 280)", "Camera 3 (320 x 280)"]


def test_the_add_a_person_form_explains_what_is_missing(make_app):
    app = make_app()
    assert "Add a person" in app.warning[0].value
    app.sidebar.text_input[0].set_value("Awa Diallo")
    next(b for b in app.sidebar.button if b.label == "Add and learn").click().run()
    assert not app.exception
    assert "at least one photo" in app.sidebar.error[0].value


def test_online_the_visitor_tries_the_astronauts_and_nothing_is_saved(make_app, tmp_path):
    app = make_app(public=True)
    assert "Online demo" in app.sidebar.info[0].value
    assert "Nobody has been learnt yet" in app.warning[0].value
    next(b for b in app.button if b.label == "Try it with NASA astronauts").click().run()
    assert not app.exception
    assert app.sidebar.caption[-1].value == "Christina Koch, Victor Glover"  # the people it knows
    assert len(app.dataframe[0].value) >= 1  # the example crew photo is analysed
    assert "Phone or stream" not in app.button_group[0].options  # the server cannot see the visitor's phone
    assert not (tmp_path / "no_faces").exists() and not (tmp_path / "models").exists()  # nothing written
    other = make_app(public=True)  # another visitor knows nobody
    assert "Nobody has been learnt yet" in other.warning[0].value
    next(b for b in app.sidebar.button if b.label == "Forget everyone").click().run()
    assert "Nobody has been learnt yet" in app.warning[0].value


def test_online_the_form_explains_what_is_missing_and_writes_nothing(make_app, tmp_path):
    app = make_app(public=True)
    app.sidebar.text_input[0].set_value("Awa Diallo")
    next(b for b in app.sidebar.button if b.label == "Add and learn").click().run()
    assert not app.exception
    assert "at least one photo" in app.sidebar.error[0].value
    assert not (tmp_path / "no_faces").exists()


def test_online_the_models_are_downloaded_once(make_app, monkeypatch):
    downloads = []

    def download(url, path):
        downloads.append(url)
        path.write_bytes(b"fake")
        return path

    monkeypatch.setattr(facerec.detection, "download", download)
    app = make_app(models=False, public=True)
    assert not app.exception and not app.error
    assert [url.rsplit("/", 1)[-1] for url in downloads] == ["face_detection_yunet_2023mar.onnx",
                                                              "face_recognition_sface_2021dec.onnx"]
