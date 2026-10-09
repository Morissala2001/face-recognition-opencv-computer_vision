"""Command line: `facerec download | enroll | recognize | webcam | video | evaluate`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from . import detection, recognizer as recognizer_module
from .camera import camera_source
from .enroll import enroll
from .images import read_image
from .recognize import draw, recognize, run_webcam
from .recognizer import DEFAULT_MODEL, DEFAULT_THRESHOLD, SFACE_MODEL, SFACE_URL
from .video import process_video, timestamp


def cmd_download(args) -> None:
    for url, path in ((detection.YUNET_URL, detection.YUNET_MODEL), (SFACE_URL, SFACE_MODEL)):
        path = detection.download(url, path)
        print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")


def cmd_enroll(args) -> None:
    recognizer, enrollment = enroll(args.faces, detection.load_detector(),
                                    recognizer_module.new_recognizer(args.threshold))
    for name, n in sorted(enrollment.per_person.items()):
        print(f"{name:20s} {n} faces")
    for photo in enrollment.skipped:
        print(f"skipped (no face found): {photo}", file=sys.stderr)
    print(f"Saved to {recognizer.save(args.model)}")


def trained(args):
    return recognizer_module.new_recognizer(args.threshold).load(args.model)


def cmd_recognize(args) -> None:
    image = read_image(args.photo)
    if image is None:
        sys.exit(f"Cannot read {args.photo}")
    results = recognize(image, detection.load_detector(), trained(args))
    if not results:
        print("No face found.")
    for r in results:
        print(f"{r.label:28s} box {r.box.left},{r.box.top},{r.box.right},{r.box.bottom}")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), draw(image, results))
        print(f"Annotated picture: {args.save}")


def cmd_webcam(args) -> None:
    print(f"{run_webcam(detection.load_detector(), trained(args), args.camera)} frames shown")


def cmd_video(args) -> None:
    def progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            print(f"\r{done}/{total or '?'} frames", end="", file=sys.stderr, flush=True)

    report = process_video(args.video, detection.load_detector(), trained(args), args.save, args.every,
                           max_seconds=args.seconds, progress=progress)
    print(file=sys.stderr)
    print(f"{report.frames} frames ({timestamp(report.duration)}), faces recognised on {report.analysed}")
    for name, seen in sorted(report.people.items(), key=lambda item: item[1].first):
        print(f"{name:20s} {timestamp(seen.first)} -> {timestamp(seen.last)}   on {seen.frames} analysed frames")
    if not report.people:
        print("Nobody it knows appears in this video.")
    if report.unknown_frames:
        print(f"{'Unknown':20s} on {report.unknown_frames} analysed frames")
    if args.save:
        print(f"Annotated video: {args.save}")


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate, lfw_folder, load_index

    report = evaluate(load_index(lfw_folder()), detection.load_detector(), recognizer_module.new_recognizer(),
                      args.people, args.repeats, args.unknown, args.seed)
    print(json.dumps(report, indent=1))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="facerec", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download YuNet and SFace (39 MB)").set_defaults(func=cmd_download)

    def add_model_options(p):
        p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="file of the enrolled faces")
        p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                       help="largest distance still named; above it the face is Unknown (default: measured on LFW)")

    p = sub.add_parser("enroll", help="learn the people of a folder (one subfolder of photos per person)")
    p.add_argument("faces", type=Path)
    add_model_options(p)
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("recognize", help="name the faces of a photo")
    p.add_argument("photo", type=Path)
    p.add_argument("--save", type=Path, help="write the annotated picture here")
    add_model_options(p)
    p.set_defaults(func=cmd_recognize)

    p = sub.add_parser("webcam", help="recognise faces live (press q to quit)")
    p.add_argument("--camera", type=camera_source, default=0,
                   help="0, 1... for a camera of this computer (iVCam's phone is one of them), or the DroidCam "
                        "address of a phone on the same Wi-Fi, e.g. http://192.168.1.20:4747/video")
    add_model_options(p)
    p.set_defaults(func=cmd_webcam)

    p = sub.add_parser("video", help="name the faces of a video file or stream and tell who appears, and when")
    p.add_argument("video", help="a video file, or a stream address (rtsp://..., http://...)")
    p.add_argument("--seconds", type=float, help="stop after this many seconds of video (needed for a live stream)")
    p.add_argument("--save", type=Path, help="write the annotated video here (.mp4, .webm or .avi)")
    p.add_argument("--every", type=int, default=1,
                   help="recognise faces on one frame out of N (faster; the names stay drawn in between)")
    add_model_options(p)
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("evaluate", help="measure the recognizer on LFW (downloads about 230 MB once)")
    p.add_argument("--people", type=int, default=5, help="people enrolled in each scenario")
    p.add_argument("--repeats", type=int, default=20, help="scenarios: half to choose the threshold, half to report")
    p.add_argument("--unknown", type=int, default=200, help="faces of strangers in each scenario")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, help="also write the report as JSON")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
