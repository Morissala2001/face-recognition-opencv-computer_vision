import numpy as np
import pytest
from conftest import FACE_BOX, fake_recognizer, photo, save, texture
from PIL import Image

from facerec.detection import Box, clip
from facerec.images import list_images, read_image
from facerec.recognizer import SFaceRecognizer


def test_boxes_are_kept_inside_the_picture():
    assert clip(Box(-20, -5, 700, 300), 640, 480) == Box(0, 0, 640, 300)
    assert clip(Box(700, 10, 800, 50), 640, 480).area == 0
    assert clip(FACE_BOX, 640, 480).raw == FACE_BOX.raw  # the landmarks travel with the box


class FakeYuNet:
    """Like YuNet, finds faces only in moderate-size inputs: one face at a fixed place of what it is given."""

    def __init__(self):
        self.sizes = []

    def setInputSize(self, size):
        self.sizes.append(size)

    def detect(self, image):
        height, width = image.shape[:2]
        if max(height, width) > 1000:
            return 1, None
        row = [width / 4, height / 4, width / 2, height / 2] + [width / 2, height / 2] * 5 + [0.9]
        return 1, np.array([row], np.float32)


def test_large_photos_are_shrunk_for_detection_and_boxes_scaled_back(tmp_path, monkeypatch):
    """Regression: a face filling a 2,560-pixel phone photo was missed (6 of 22 found on real photos)."""
    import facerec.detection as detection

    fake = FakeYuNet()
    monkeypatch.setattr(detection.cv2, "FaceDetectorYN", type("Factory", (), {"create": staticmethod(lambda *a: fake)}))
    (tmp_path / "yunet.onnx").write_bytes(b"")
    detector = detection.YuNetDetector(tmp_path / "yunet.onnx")
    [box] = detector.detect(np.zeros((2560, 1920, 3), np.uint8))
    assert fake.sizes == [(480, 640)]  # the longest side brought to 640 pixels
    assert (box.left, box.top, box.right, box.bottom) == (480, 640, 1440, 1920)  # in the photo's own pixels
    assert box.raw[4:6] == (960.0, 1280.0) and box.raw[14] == pytest.approx(0.9)  # landmarks too, score unchanged
    detector.detect(np.zeros((250, 250, 3), np.uint8))
    assert fake.sizes[-1] == (250, 250)  # small pictures are left as they are


def test_pictures_are_read_upright_from_any_path(tmp_path):
    path = save(photo(1), tmp_path / "Ünïcödé dossier" / "photo.jpg")  # cv2.imread fails on such paths on Windows
    assert read_image(path).shape == (280, 320, 3)
    turned = tmp_path / "turned.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90°": how phones store portrait photos
    Image.new("RGB", (40, 20)).save(turned, exif=exif)
    assert read_image(turned).shape == (40, 20, 3)  # height and width swapped back
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    assert read_image(tmp_path / "broken.jpg") is None


def test_iphone_heic_photos_are_read(tmp_path):
    original = photo(1)
    path = save(original, tmp_path / "IMG_0001.HEIC")
    assert path.read_bytes()[4:12] == b"ftypheic"  # a real HEIC file, as an iPhone writes it
    image = read_image(path)
    assert image.shape == original.shape
    assert np.abs(image.astype(int) - original).mean() < 5  # HEIC is lossy, but close to the original
    (tmp_path / "broken.heic").write_bytes(path.read_bytes()[:200])
    assert read_image(tmp_path / "broken.heic") is None


def test_only_pictures_are_listed(tmp_path):
    for name in ("b.PNG", "a.jpg", "c.txt", "d.webp", "e.HEIC", "f.heif"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.jpg", "b.PNG", "d.webp", "e.HEIC", "f.heif"]


def faces_of(recognizer, person, shots=range(5)):
    return [recognizer.prepare(photo(person, s), FACE_BOX) for s in shots]


@pytest.fixture
def trained():
    r = fake_recognizer()
    return r.train(faces_of(r, 1) + faces_of(r, 2), ["Ada"] * 5 + ["Grace"] * 5)


def test_faces_become_unit_vectors_and_need_landmarks():
    r = fake_recognizer()
    vector = r.prepare(photo(1), FACE_BOX)
    assert vector.shape == (128,) and np.linalg.norm(vector) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="landmarks"):
        r.prepare(photo(1), Box(40, 30, 240, 230))


def test_known_faces_are_named_and_a_stranger_is_unknown(trained):
    ada, grace = faces_of(trained, 1, [10])[0], faces_of(trained, 2, [10])[0]
    assert trained.predict(ada).name == "Ada" and trained.predict(grace).name == "Grace"
    stranger = trained.predict(trained.prepare(photo(7), FACE_BOX))
    assert stranger.name is None and stranger.distance > trained.threshold
    assert stranger.closest in {"Ada", "Grace"}  # the closest person is still reported
    trained.threshold = 2.0  # the largest possible distance: everybody gets a name
    assert trained.predict(trained.prepare(photo(7), FACE_BOX)).name is not None


def test_enrolled_faces_survive_saving_and_loading(tmp_path, trained):
    path = trained.save(tmp_path / "models" / "faces.npz")
    reloaded = trained.fresh().load(path)
    assert reloaded.names == ["Ada", "Grace"] and reloaded.net is trained.net
    assert reloaded.predict(faces_of(reloaded, 2, [9])[0]).name == "Grace"
    r = fake_recognizer()
    r.train(faces_of(r, 3), ["Мори"] * 5).save(tmp_path / "ru.npz")
    assert fake_recognizer().load(tmp_path / "ru.npz").names == ["Мори"]


def test_with_nobody_enrolled_every_face_is_unknown():
    r = fake_recognizer()
    assert r.predict(r.prepare(photo(1), FACE_BOX)).name is None  # used to crash on an empty argmax


def test_training_and_loading_errors(tmp_path):
    with pytest.raises(ValueError, match="No face"):
        fake_recognizer().train([], [])
    with pytest.raises(FileNotFoundError, match="facerec enroll"):
        fake_recognizer().load(tmp_path / "missing.npz")
    with pytest.raises(FileNotFoundError, match="facerec download"):
        SFaceRecognizer(model=tmp_path / "missing.onnx")


def test_synthetic_people_really_differ():
    assert not np.array_equal(texture(1), texture(2))
