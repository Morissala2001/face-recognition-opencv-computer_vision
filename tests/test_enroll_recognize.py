import cv2
import numpy as np
import pytest
from conftest import FACE_BOX, FakeDetector, fake_recognizer, photo, save

import facerec.recognize as recognize_module
from facerec.detection import Box
from facerec.enroll import add_person, collect_faces, enroll, folder_name
from facerec.recognize import KNOWN_COLOR, UNKNOWN_COLOR, Recognition, draw, recognize, run_webcam


def test_enrolment_skips_photos_without_a_face_instead_of_crashing(faces_folder, detector):
    enrollment = collect_faces(faces_folder, detector, fake_recognizer())
    assert enrollment.per_person == {"Ada": 6, "Grace": 6}
    assert [p.name for p in enrollment.skipped] == ["black.png"]  # the original notebook crashed on such a photo
    assert detector.calls == 13  # the text file is not even opened


def test_enrolment_needs_person_folders(tmp_path, detector):
    with pytest.raises(ValueError, match="one subfolder per person"):
        collect_faces(tmp_path, detector, fake_recognizer())


def test_enrolled_people_are_recognised(faces_folder, detector):
    recognizer, _ = enroll(faces_folder, detector, fake_recognizer())
    assert [r.name for r in recognize(photo(1, 20), detector, recognizer)] == ["Ada"]
    assert [r.name for r in recognize(photo(2, 20), detector, recognizer)] == ["Grace"]
    assert recognize(np.zeros((50, 50, 3), np.uint8), detector, recognizer) == []


def test_drawing_names_known_faces_in_blue_and_strangers_in_red():
    image = photo(1)
    known = draw(image, [Recognition(FACE_BOX, "Ada", 0.12)])
    stranger = draw(image, [Recognition(FACE_BOX, None, 0.81)])
    assert known.shape == image.shape and not np.array_equal(known, image)
    assert tuple(known[FACE_BOX.bottom + 3, FACE_BOX.left + 1]) == KNOWN_COLOR  # banner below the face
    assert tuple(stranger[FACE_BOX.bottom + 3, FACE_BOX.left + 1]) == UNKNOWN_COLOR
    assert Recognition(FACE_BOX, None, 0.8123).label == "Unknown  0.81"


def test_the_banner_goes_above_a_face_touching_the_bottom():
    image = np.zeros((200, 200, 3), np.uint8)
    drawn = draw(image, [Recognition(Box(50, 60, 150, 200), "Ada", 0.1)])
    assert tuple(drawn[57, 51]) == KNOWN_COLOR


class FakeCamera:
    def __init__(self, frames, opened=True):
        self.frames, self.opened, self.released = list(frames), opened, False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        return True

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        self.released = True


@pytest.fixture
def screen(monkeypatch):
    shown, keys = [], []
    monkeypatch.setattr(cv2, "imshow", lambda window, frame: shown.append(frame))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: keys.pop(0) if keys else -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return shown, keys


def test_the_webcam_window_is_refreshed_even_without_a_face(faces_folder, detector, monkeypatch, screen):
    """Regression: the original refreshed the window only inside the loop over faces."""
    recognizer, _ = enroll(faces_folder, detector, fake_recognizer())
    camera = FakeCamera([photo(1, 30), np.zeros((280, 320, 3), np.uint8), photo(2, 30)])
    monkeypatch.setattr(recognize_module.cv2, "VideoCapture", lambda index: camera)
    assert run_webcam(detector, recognizer) == 3
    assert len(screen[0]) == 3 and camera.released


def test_the_webcam_stops_on_q_and_reports_a_missing_camera(faces_folder, detector, monkeypatch, screen):
    recognizer, _ = enroll(faces_folder, detector, fake_recognizer())
    screen[1].append(ord("q"))
    monkeypatch.setattr(recognize_module.cv2, "VideoCapture", lambda index: FakeCamera([photo(1)] * 5))
    assert run_webcam(FakeDetector(), recognizer) == 1
    monkeypatch.setattr(recognize_module.cv2, "VideoCapture", lambda index: FakeCamera([], opened=False))
    with pytest.raises(RuntimeError, match="--camera"):
        run_webcam(detector, recognizer, camera=3)


def png(image) -> bytes:
    _, data = cv2.imencode(".png", image)
    return data.tobytes()


class GroupDetector(FakeDetector):
    """Like FakeDetector, but sees two faces in wide pictures (a group photo)."""

    def detect(self, image):
        faces = super().detect(image)
        return faces * 2 if image.shape[1] > 400 else faces


def test_adding_a_person_keeps_only_photos_with_a_face(tmp_path):
    photos = [("a.png", png(photo(5, 1))), ("b.png", png(photo(5, 2))),
              ("empty.png", png(np.zeros((90, 90, 3), np.uint8))), ("group.png", png(photo(5, 3, size=(280, 500)))),
              ("notes.png", b"not a picture")]
    added = add_person(tmp_path / "faces", "  ../Awa: Diallo? ", photos, GroupDetector())
    assert added.person == tmp_path / "faces" / "Awa Diallo"  # no path trick, no forbidden character
    assert added.saved == ["a.png", "b.png", "group.png"]
    assert added.no_face == ["empty.png"] and added.unreadable == ["notes.png"]
    assert added.several_faces == ["group.png"]
    assert sorted(p.name for p in added.person.iterdir()) == ["a.png", "b.png", "group.png"]


def test_photos_are_added_to_a_known_person_without_overwriting(tmp_path, detector):
    add_person(tmp_path, "Мори", [("selfie.png", png(photo(3, 1)))], detector)
    again = add_person(tmp_path, "Мори", [("selfie.png", png(photo(3, 2))), ("x", png(photo(3, 3)))], detector)
    assert again.saved == ["selfie_2.png", "x.jpg"]
    assert len(list((tmp_path / "Мори").iterdir())) == 3


def test_a_name_and_photos_are_needed(tmp_path, detector):
    with pytest.raises(ValueError, match="at least one photo"):
        add_person(tmp_path, "Ada", [], detector)
    for name in ("   ", "../..", "CON", "nul"):
        with pytest.raises(ValueError, match="Give the person a name"):
            folder_name(name)
    assert folder_name("Jean-Pierre O'Neil") == "Jean-Pierre O'Neil"


def test_added_people_are_recognised_after_enrolment(tmp_path, detector):
    for person, name in ((1, "Ada"), (4, "Bob")):
        add_person(tmp_path, name, [(f"{s}.png", png(photo(person, s))) for s in range(5)], detector)
    recognizer, enrollment = enroll(tmp_path, detector, fake_recognizer())
    assert enrollment.per_person == {"Ada": 5, "Bob": 5}
    assert [r.name for r in recognize(photo(4, 30), detector, recognizer)] == ["Bob"]


def test_iphone_heic_photos_can_be_added_and_recognised(tmp_path, detector):
    heic = [(f"IMG_{s}.HEIC", (save(photo(6, s), tmp_path / "phone" / f"{s}.heic")).read_bytes()) for s in range(4)]
    added = add_person(tmp_path / "faces", "Mori", heic, detector)
    assert added.saved == [f"IMG_{s}.heic" for s in range(4)]  # kept as HEIC, as they came from the phone
    recognizer, enrollment = enroll(tmp_path / "faces", detector, fake_recognizer())
    assert enrollment.per_person == {"Mori": 4}
    assert [r.name for r in recognize(photo(6, 30), detector, recognizer)] == ["Mori"]
