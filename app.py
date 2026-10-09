"""Web app: a photo, the computer's camera, a phone's camera or a video file, and see who is in it.

    uv run streamlit run app.py

It needs the two models (`uv run facerec download`) and the enrolled faces: either `uv run facerec enroll faces/`,
or the "Learn the faces" button when a faces/ folder exists. Pictures are processed in memory and never saved.
"""

import os
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from facerec import camera, detection, recognizer as recognizer_module
from facerec.enroll import add_person, enroll
from facerec.images import read_image
from facerec.recognize import UNKNOWN, draw, recognize
from facerec.recognizer import DEFAULT_MODEL, DEFAULT_THRESHOLD, SFACE_MODEL, THRESHOLD_RANGE
from facerec.video import fit, process_video, timestamp

MODEL = Path(os.environ.get("FACEREC_MODEL", DEFAULT_MODEL))
FACES = Path(os.environ.get("FACEREC_FACES", "faces"))
REQUIRED = [Path(os.environ.get("FACEREC_YUNET", detection.YUNET_MODEL)),
            Path(os.environ.get("FACEREC_SFACE", SFACE_MODEL))]

st.set_page_config(page_title="Face recognition", page_icon=":material/face:")
# A browser that translates the page (Chrome into French, for instance) rewrites the texts that Streamlit's React
# code manages, which crashes it ("removeChild ... is not a child of this node"). Ask browsers not to translate.
st.html("<script>document.documentElement.setAttribute('translate', 'no');"
        "document.documentElement.classList.add('notranslate');</script>", unsafe_allow_javascript=True)


@st.cache_resource(show_spinner="Loading the models…")
def get_models():
    return detection.load_detector(), recognizer_module.new_recognizer()


def learn_faces() -> None:
    detector, recognizer = get_models()
    with st.spinner("Detecting and learning the faces of each person folder…"):
        recognizer, enrollment = enroll(FACES, detector, recognizer.fresh())
        recognizer.save(MODEL)
    st.session_state.enrolled = dict(enrollment.per_person)
    st.session_state.skipped = [p.name for p in enrollment.skipped]


def add_and_learn() -> None:
    """The 'Add a person' form: keep the photos that show a face, then learn everyone again."""
    photos = [(f.name, f.getvalue()) for f in st.session_state.new_photos or []]
    try:
        added = add_person(FACES, st.session_state.new_name, photos, get_models()[0])
    except ValueError as error:
        st.session_state.added = str(error)
        return
    if added.saved:
        learn_faces()
    st.session_state.added = added


people = sorted(p.name for p in FACES.iterdir() if p.is_dir()) if FACES.is_dir() else []

st.title("Face recognition")
st.caption("Names the people it has learnt, and says Unknown for everyone else.")

if missing := [p for p in REQUIRED if not p.exists()]:
    st.error("Missing model files: " + ", ".join(f"`{p}`" for p in missing) + ". Download them first (39 MB):")
    st.code("uv run facerec download", language="bash")
    st.stop()

with st.sidebar:
    st.header("Settings")
    low, high, step = THRESHOLD_RANGE
    threshold = st.slider("Unknown above this distance", low, high, DEFAULT_THRESHOLD, step,
                          help="1 - cosine similarity with the closest learnt face: the lower, the more alike. "
                               "Lower it to call more faces Unknown, raise it to name more faces (and make more "
                               f"mistakes). {DEFAULT_THRESHOLD} was chosen on the LFW benchmark.")
    with st.expander("Add a person", icon=":material/person_add:", expanded=not people):
        with st.form("add_person", clear_on_submit=True, border=False):
            st.text_input("Name", key="new_name", placeholder="Awa Diallo",
                          help="Shown on the faces it recognises. A known name adds photos to that person.")
            st.file_uploader("15 to 20 photos, one clearly visible face in each", key="new_photos",
                             type=["jpg", "jpeg", "png", "webp", "heic", "heif"], accept_multiple_files=True,
                             help="Varied photos: front and slightly sideways, different lights, with and without "
                                  "glasses. Ask the person first: faces are personal data.")
            st.form_submit_button("Add and learn", type="primary", on_click=add_and_learn, width="stretch")
        added = st.session_state.pop("added", None)
        if isinstance(added, str):
            st.error(added)
        elif added is not None:
            if added.saved:
                st.success(f"{added.person.name}: {len(added.saved)} photo(s) added and learnt.")
            if added.no_face:
                st.warning("No face found, not kept: " + ", ".join(added.no_face))
            if added.several_faces:
                st.warning("Several faces, the clearest one is used (check it is the right person): "
                           + ", ".join(added.several_faces))
            if added.unreadable:
                st.warning("Not a readable picture: " + ", ".join(added.unreadable))
    if people:
        st.button(f"Learn the faces again from `{FACES}/`", on_click=learn_faces, width="stretch")
    if "enrolled" in st.session_state:
        st.caption("Learnt: " + ", ".join(f"{name} ({n} faces)" for name, n in st.session_state.enrolled.items()))
        if st.session_state.skipped:
            st.caption("Skipped, no face found: " + ", ".join(st.session_state.skipped))

if not MODEL.exists():
    if people:
        st.info(f"Found {len(people)} person folder(s) in `{FACES}/`: {', '.join(people)}.")
        st.button("Learn the faces", type="primary", on_click=learn_faces)
    else:
        st.warning("Nobody has been learnt yet. Use **Add a person** in the sidebar, or create "
                   f"`{FACES}/<name>/` with about 20 photos per person (one clearly visible face in each) and run:")
        st.code(f"uv run facerec enroll {FACES}", language="bash")
    st.stop()

detector, base = get_models()
recognizer = base.fresh().load(MODEL)  # quick: a few kilobytes of vectors
recognizer.threshold = threshold
st.sidebar.write("People it knows")
st.sidebar.caption(", ".join(recognizer.names))

PHOTO, COMPUTER, PHONE, VIDEO = "Photo file", "Computer camera", "Phone or stream", "Video file"
LIVE_SECONDS = 300  # the live view stops by itself after 5 minutes
# The computer camera is opt-in: a camera widget asks the browser for camera access as soon as it is displayed.
kind = st.segmented_control("Source", [PHOTO, COMPUTER, PHONE, VIDEO], default=PHOTO, key="source",
                            label_visibility="collapsed") or PHOTO
image = None
if kind == VIDEO:
    clip = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm"], label_visibility="collapsed")
    if clip is None:
        st.info("Drop a video. It is analysed on this computer, then deleted.")
        st.stop()
    every = st.select_slider("Recognise faces on", options=[1, 2, 3, 5, 10], value=1,
                             format_func=lambda n: "every frame" if n == 1 else f"1 frame out of {n}",
                             help="Fewer frames is faster; the names stay drawn on the frames in between.")
    if st.button("Analyse the video", type="primary", icon=":material/movie:"):
        bar = st.progress(0.0, text="Analysing the video…")

        def progress(done: int, total: int) -> None:
            if total and (done % 10 == 0 or done == total):
                bar.progress(min(done / total, 1.0), text=f"Frame {done} of {total}")

        with tempfile.TemporaryDirectory() as folder:  # the video and its copy are deleted right after
            source, named = Path(folder) / f"input{Path(clip.name).suffix}", Path(folder) / "named.webm"
            source.write_bytes(clip.getvalue())
            report = process_video(source, detector, recognizer, named, every, max_side=1280, progress=progress)
            st.session_state.video = {"name": clip.name, "threshold": threshold, "data": named.read_bytes(),
                                      "report": report}
        bar.empty()
    result = st.session_state.get("video")
    if result is None or result["name"] != clip.name:
        st.stop()
    report = result["report"]
    st.video(result["data"], format="video/webm")
    rows = [{"Name": name, "First seen": timestamp(seen.first), "Last seen": timestamp(seen.last),
             "Analysed frames": seen.frames} for name, seen in sorted(report.people.items(), key=lambda p: p[1].first)]
    if report.unknown_frames:
        rows.append({"Name": UNKNOWN, "First seen": "", "Last seen": "", "Analysed frames": report.unknown_frames})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", alt="Who appears in the video, and when")
    else:
        st.warning("No face found in this video.")
    st.caption(f"{report.frames} frames ({timestamp(report.duration)}), faces recognised on {report.analysed} of "
               f"them, with the threshold {result['threshold']}. To use another threshold, move the slider and "
               "analyse again.")
    st.stop()
elif kind == PHOTO:
    uploaded = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
                                label_visibility="collapsed")
    if uploaded is not None and (image := read_image(uploaded)) is None:
        st.error("This file cannot be read as a picture.")
        st.stop()
elif kind == COMPUTER:
    shot = st.camera_input("Take a photo", label_visibility="collapsed")
    image = read_image(shot) if shot is not None else None
else:
    with st.expander("How to connect a phone or a stream", expanded="phone_frame" not in st.session_state):
        st.markdown(
            "- **iVCam** (iPhone, Android): open iVCam on the phone and on this computer. The phone becomes a "
            "camera of this computer: type its **number** below (often `1`; *Show the cameras* tells you which).\n"
            "- **DroidCam** (Android, iPhone): with the phone on the same Wi-Fi, type the address DroidCam shows "
            "followed by `/video`, for instance `http://192.168.1.20:4747/video`.\n"
            "- **A network camera or a video stream**: its address, such as `rtsp://...` for most IP and security "
            "cameras, or an `http(s)://...` stream.\n\n"
            "Phone apps serve one program at a time, and anyone on the same Wi-Fi can open a DroidCam stream "
            "while it runs: stop the app when you are done.")
    source_text = st.text_input("Camera number, phone or stream address", key="phone_source",
                                placeholder="1   or   http://192.168.1.20:4747/video   or   rtsp://...")
    take, live, show = st.columns(3)
    if take.button("Take a picture", type="primary", disabled=not source_text.strip(),
                   icon=":material/photo_camera:", width="stretch"):
        try:
            st.session_state.phone_frame = camera.grab_frame(camera.camera_source(source_text))
        except RuntimeError as error:
            st.error(str(error))
    live.button("Watch live", disabled=not source_text.strip(), icon=":material/live_tv:", width="stretch",
                on_click=lambda: st.session_state.update(live=True))
    if st.session_state.get("live"):
        st.button("Stop", icon=":material/stop:", on_click=lambda: st.session_state.update(live=False))
        screen, who = st.empty(), st.empty()
        try:
            capture = camera.open_camera(camera.camera_source(source_text))
        except RuntimeError as error:
            st.session_state.live = False
            st.error(str(error))
            st.stop()
        try:  # a click on Stop (or anything else) reruns the script, which ends this loop and releases the camera
            start, frames = time.monotonic(), 0
            while time.monotonic() - start < LIVE_SECONDS:
                ok, frame = capture.read()
                if not ok:
                    break
                results = recognize(frame, detector, recognizer)
                screen.image(cv2.cvtColor(fit(draw(frame, results), 960), cv2.COLOR_BGR2RGB), width="stretch",
                             alt="The live picture, with the faces outlined and named")
                names = sorted({r.name or UNKNOWN for r in results})
                who.caption("In front of the camera: " + (", ".join(names) if names else "nobody"))
                frames += 1
        finally:
            capture.release()
        st.session_state.live = False
        st.info(f"Live view ended after {frames} frames (it stops by itself after {LIVE_SECONDS // 60} minutes).")
        st.stop()
    if show.button("Show the cameras of this computer", icon=":material/videocam:", width="stretch"):
        found = camera.computer_cameras()
        if not found:
            st.warning("No camera answered.")
        for column, (index, frame) in zip(st.columns(max(len(found), 1)), found):
            column.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Camera {index} "
                         f"({frame.shape[1]} x {frame.shape[0]})", alt=f"What camera {index} sees")
    image = st.session_state.get("phone_frame")

if image is None:
    st.info({PHOTO: "Drop a photo of one or more people.", COMPUTER: "Take a photo with this computer's camera.",
             PHONE: "Take a picture, or watch live."}[kind] + " It is processed in memory and never saved.")
    st.stop()

results = recognize(image, detector, recognizer)
st.image(cv2.cvtColor(draw(image, results), cv2.COLOR_BGR2RGB), width="stretch",
         alt=f"The photo with {len(results)} face(s) outlined and named")
if results:
    st.dataframe(pd.DataFrame([{"Name": r.name or UNKNOWN, "Distance": round(r.distance, 3)} for r in results]),
                 hide_index=True, width="stretch", alt="The faces found, with their name and distance")
else:
    st.warning("No face found in this picture.")
