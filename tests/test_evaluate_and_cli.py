import json

import numpy as np
import pytest
from conftest import FakeDetector, fake_recognizer, photo, save

import facerec.cli as cli
import facerec.detection
import facerec.evaluate as evaluation
import facerec.recognizer
from facerec.evaluate import FaceCache, Outcome, best_threshold, draw_scenario, evaluate, load_index, rates, run_scenario


@pytest.fixture(scope="module")
def mini_lfw(tmp_path_factory):
    """An LFW-like folder: 8 people with 30 photos, 40 people with 2 photos."""
    folder = tmp_path_factory.mktemp("lfw")
    for person in range(48):
        for shot in range(30 if person < 8 else 2):
            save(photo(person + 1, shot, size=(250, 250)), folder / f"Person_{person:02d}" / f"{shot:04d}.jpg")
    return folder


def test_a_scenario_enrols_20_photos_per_person_and_never_enrols_a_stranger(mini_lfw):
    index = load_index(mini_lfw)
    scenario = draw_scenario(index, np.random.default_rng(0), people=3, unknown=10)
    enrolled = {name for _, name in scenario.enrol}
    assert len(scenario.enrol) == 60 and len(enrolled) == 3
    assert len(scenario.known) == 30 and {name for _, name in scenario.known} == enrolled
    assert not {p for p, _ in scenario.enrol} & {p for p, _ in scenario.known}  # tested photos were never learnt
    assert len(scenario.unknown) == 10 and not {p.parent.name for p in scenario.unknown} & enrolled


def test_rates_add_up_and_the_threshold_balances_both_errors():
    outcome = Outcome(known=[("A", "A", 0.10), ("A", "B", 0.30), ("B", "B", 0.60)], unknown=[0.20, 0.80])
    r = rates([outcome], 0.50)
    assert r["known_named_right"] == pytest.approx(1 / 3) and r["known_named_wrong"] == pytest.approx(1 / 3)
    assert r["known_called_unknown"] == pytest.approx(1 / 3) and r["strangers_rejected"] == 0.5
    assert r["closed_set_accuracy"] == pytest.approx(2 / 3)
    # 0.10 names the first known face and rejects both strangers: (1/3 + 1) / 2 beats (2/3 + 1/2) / 2 at 0.60
    assert 0.10 <= best_threshold([outcome], np.arange(0, 1.2, 0.01)) < 0.20


def test_synthetic_people_are_recognised_and_strangers_rejected(mini_lfw):
    faces = FaceCache(FakeDetector(), fake_recognizer())
    outcome = run_scenario(draw_scenario(load_index(mini_lfw), np.random.default_rng(1), 3, 10), faces)
    threshold = best_threshold([outcome], np.arange(0, 1.2, 0.01))
    assert rates([outcome], threshold)["known_named_right"] == 1.0
    assert rates([outcome], threshold)["strangers_rejected"] >= 0.9
    calls = faces.detector.calls
    run_scenario(draw_scenario(load_index(mini_lfw), np.random.default_rng(1), 3, 10), faces)
    assert faces.detector.calls == calls  # every photo is detected only once


def test_the_full_evaluation_is_reproducible(mini_lfw):
    first = evaluate(load_index(mini_lfw), FakeDetector(), fake_recognizer(), people=3, repeats=4, unknown=10, seed=3)
    second = evaluate(load_index(mini_lfw), FakeDetector(), fake_recognizer(), people=3, repeats=4, unknown=10, seed=3)
    assert first["test"] == second["test"] and first["chosen_threshold"] == second["chosen_threshold"]
    assert first["no_face_detected"] == 0 and first["test_with_opencv_threshold"]["threshold"] == pytest.approx(0.637)


@pytest.fixture
def run(monkeypatch, capsys):
    monkeypatch.setattr(facerec.detection, "load_detector", lambda *args, **kwargs: FakeDetector())
    monkeypatch.setattr(facerec.recognizer, "new_recognizer", lambda threshold=0.45: fake_recognizer(threshold))

    def run(*args):
        cli.main(list(map(str, args)))
        return capsys.readouterr()

    return run


def test_cli_enroll_then_recognize(run, faces_folder, tmp_path):
    model = tmp_path / "model.yml"
    out = run("enroll", faces_folder, "--model", model)
    assert "Ada                  6 faces" in out.out and "black.png" in out.err and model.exists()
    picture = save(photo(2, 40), tmp_path / "group.jpg")
    out = run("recognize", picture, "--model", model, "--save", tmp_path / "out" / "a.jpg")
    assert out.out.startswith("Grace") and (tmp_path / "out" / "a.jpg").exists()


def test_cli_recognize_needs_a_model(run, tmp_path):
    picture = save(photo(2), tmp_path / "p.jpg")
    with pytest.raises(FileNotFoundError, match="facerec enroll"):
        run("recognize", picture, "--model", tmp_path / "none.yml")


def test_cli_evaluate_writes_a_report(run, mini_lfw, monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "lfw_folder", lambda: mini_lfw)
    run("evaluate", "--people", 3, "--repeats", 2, "--unknown", 10, "--output", tmp_path / "r.json")
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert {"chosen_threshold", "test", "sweep"} <= set(report)
