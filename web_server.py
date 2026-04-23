import os
import sys
import time
import atexit
import sqlite3
import threading
from functools import wraps

import cv2
import mediapipe as mp
from flask import (
    Flask, Response, jsonify, render_template,
    request, redirect, url_for, session
)
from werkzeug.security import generate_password_hash, check_password_hash

from pose_analyzer import get_joint_angles, EXERCISE_CLASSES
from ui_widgets import render_dashboard, draw_no_pose, get_exercise_tips, COLORS


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
RESOURCE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))

if getattr(sys, "frozen", False):
    DATA_DIR = os.path.dirname(sys.executable)
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(DATA_DIR, "users.db")

app = Flask(
    __name__,
    template_folder=os.path.join(RESOURCE_DIR, "templates"),
    static_folder=os.path.join(RESOURCE_DIR, "static"),
)

app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key-123456")


# -----------------------------------------------------------------------------
# DATABASE
# -----------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def get_user_by_email(email):
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (email.lower().strip(),)
    ).fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()
    conn.close()
    return user


def create_user(username, email, password):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (
                username.strip(),
                email.lower().strip(),
                generate_password_hash(password)
            )
        )
        conn.commit()
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.lower().strip(),)
        ).fetchone()
        conn.close()
        return user
    except sqlite3.IntegrityError:
        conn.close()
        return None


# -----------------------------------------------------------------------------
# AUTH HELPERS
# -----------------------------------------------------------------------------
def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_user_by_id(user_id)


def page_login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user:
            session.clear()
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper


def api_login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = get_current_user()
        if not user:
            session.clear()
            return jsonify({
                "ok": False,
                "error": "Unauthorized",
                "redirect": url_for("login")
            }), 401
        return fn(*args, **kwargs)
    return wrapper


@app.context_processor
def inject_user():
    return {"current_user": get_current_user()}


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------
EXERCISE_NAMES = list(EXERCISE_CLASSES.keys())
analysers = {name: cls() for name, cls in EXERCISE_CLASSES.items()}

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

CUSTOM_LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(80, 255, 120), thickness=2, circle_radius=5
)
CUSTOM_CONNECTION_STYLE = mp_drawing.DrawingSpec(
    color=(245, 245, 245), thickness=2
)

pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = None
runtime_lock = threading.Lock()

runtime = {
    "current_exercise": EXERCISE_NAMES[0],
    "paused": False,
}

latest_state = {
    "exercise": EXERCISE_NAMES[0],
    "paused": False,
    "phase": "READY",
    "score": 0,
    "reps": 0,
    "fps": 0,
    "feedback": ["Position yourself so your full body is visible."],
    "status_title": "POSITION YOURSELF SO YOUR FULL BODY IS VISIBLE",
    "status_subtitle": "Make sure the entire body is in the camera frame",
}


# -----------------------------------------------------------------------------
# CAMERA
# -----------------------------------------------------------------------------
def open_camera():
    global cap

    if cap is not None and cap.isOpened():
        return

    if os.name == "nt":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def close_camera():
    global cap
    try:
        if cap is not None and cap.isOpened():
            cap.release()
    except Exception:
        pass
    cap = None


atexit.register(close_camera)


# -----------------------------------------------------------------------------
# STATE HELPERS
# -----------------------------------------------------------------------------
def get_runtime():
    with runtime_lock:
        return runtime["current_exercise"], runtime["paused"]


def set_exercise(name):
    with runtime_lock:
        runtime["current_exercise"] = name


def set_paused(value: bool):
    with runtime_lock:
        runtime["paused"] = bool(value)


def toggle_pause():
    with runtime_lock:
        runtime["paused"] = not runtime["paused"]
        return runtime["paused"]


def get_latest_state():
    with runtime_lock:
        return dict(latest_state)


def update_latest_state(new_data):
    with runtime_lock:
        latest_state.update(new_data)


# -----------------------------------------------------------------------------
# UI HELPERS
# -----------------------------------------------------------------------------
def build_angle_items(analyser, angles):
    joints = getattr(analyser, "joints_of_interest", None)
    if not joints:
        joints = list(angles.keys())

    items = []
    for joint in joints[:6]:
        value = float(angles.get(joint, 0) or 0)

        if 70 <= value <= 170:
            color = COLORS["green"]
        elif 45 <= value <= 195:
            color = COLORS["orange"]
        else:
            color = COLORS["red"]

        items.append({
            "label": str(joint).replace("_", " ").upper(),
            "value": value,
            "color": color
        })

    return items


# -----------------------------------------------------------------------------
# STREAM
# -----------------------------------------------------------------------------
def generate_dashboard_stream():
    global cap

    open_camera()

    prev_time = time.time()
    fps = 0.0

    while True:
        open_camera()

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        camera_view = frame.copy()

        current_exercise, paused = get_runtime()
        analyser = analysers[current_exercise]

        pose_detected = False
        score = 0
        feedback = []
        angles = {}
        phase = "READY"

        status_title = "POSITION YOURSELF SO YOUR FULL BODY IS VISIBLE"
        status_subtitle = "Make sure the entire body is in the camera frame"

        if not paused:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                pose_detected = True

                mp_drawing.draw_landmarks(
                    camera_view,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=CUSTOM_LANDMARK_STYLE,
                    connection_drawing_spec=CUSTOM_CONNECTION_STYLE,
                )

                landmarks = results.pose_landmarks.landmark
                angles = get_joint_angles(landmarks)
                score, feedback = analyser.update(landmarks, angles)
                phase = (analyser.phase or "READY").upper()

                if feedback:
                    status_title = str(feedback[0]).upper()
                    status_subtitle = "Adjust your posture and continue"
                else:
                    status_title = f"{current_exercise.upper()} | {phase}"
                    status_subtitle = "Good form. Keep going."
            else:
                draw_no_pose(camera_view)

        else:
            overlay = camera_view.copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (camera_view.shape[1], camera_view.shape[0]),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, 0.35, camera_view, 0.65, 0, camera_view)

            cv2.putText(
                camera_view,
                "PAUSED",
                (camera_view.shape[1] // 2 - 70, camera_view.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                COLORS["warning"],
                3,
                cv2.LINE_AA
            )

            phase = "PAUSED"
            status_title = "WORKOUT PAUSED"
            status_subtitle = "Press Space or click Resume"

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        angle_items = build_angle_items(analyser, angles)

        ui_state = {
            "exercise_names": EXERCISE_NAMES,
            "selected_index": EXERCISE_NAMES.index(current_exercise),
            "exercise": current_exercise,
            "phase": phase,
            "score": score,
            "reps": int(getattr(analyser, "rep_count", 0)),
            "rep_goal": 20,
            "fps": fps,
            "feedback": feedback if feedback else (
                ["Good posture detected."] if pose_detected
                else ["Position yourself so your full body is visible."]
            ),
            "tips": get_exercise_tips(current_exercise),
            "angles": angle_items,
            "status_title": status_title,
            "status_subtitle": status_subtitle,
        }

        update_latest_state({
            "exercise": current_exercise,
            "paused": paused,
            "phase": phase,
            "score": int(score),
            "reps": int(getattr(analyser, "rep_count", 0)),
            "fps": int(fps),
            "feedback": [str(x) for x in ui_state["feedback"]],
            "status_title": str(status_title),
            "status_subtitle": str(status_subtitle),
        })

        dashboard = render_dashboard(camera_view, ui_state, size=(1600, 900))

        ok, buffer = cv2.imencode(
            ".jpg",
            dashboard,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88]
        )
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# -----------------------------------------------------------------------------
# AUTH ROUTES
# -----------------------------------------------------------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if get_current_user():
        return redirect(url_for("index"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password or not confirm_password:
            error = "Please fill in all fields."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif get_user_by_email(email):
            error = "An account with this email already exists."
        else:
            user = create_user(username, email, password)
            if user:
                session["user_id"] = user["id"]
                return redirect(url_for("index"))
            error = "Signup failed. Please try again."

    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if get_current_user():
        return redirect(url_for("index"))

    error = None

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = get_user_by_email(email)

        if not user or not check_password_hash(user["password_hash"], password):
            error = "Invalid email or password."
        else:
            session["user_id"] = user["id"]
            return redirect(url_for("index"))

    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# -----------------------------------------------------------------------------
# APP ROUTES
# -----------------------------------------------------------------------------
@app.route("/")
@page_login_required
def index():
    return render_template("index.html", exercises=EXERCISE_NAMES)


@app.route("/video_feed")
@page_login_required
def video_feed():
    return Response(
        generate_dashboard_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/state")
@api_login_required
def api_state():
    return jsonify(get_latest_state())


@app.route("/api/exercise", methods=["POST"])
@api_login_required
def api_exercise():
    data = request.get_json(silent=True) or {}
    name = data.get("exercise", "")

    if name not in EXERCISE_NAMES:
        return jsonify({"ok": False, "error": "Invalid exercise"}), 400

    set_exercise(name)
    update_latest_state({"exercise": name})
    return jsonify({"ok": True, "exercise": name})


@app.route("/api/reset", methods=["POST"])
@api_login_required
def api_reset():
    current_exercise, _ = get_runtime()
    analysers[current_exercise].reset()
    update_latest_state({"reps": 0, "score": 0})
    return jsonify({"ok": True, "exercise": current_exercise})


@app.route("/api/toggle_pause", methods=["POST"])
@api_login_required
def api_toggle_pause():
    paused = toggle_pause()
    update_latest_state({"paused": paused})
    return jsonify({"ok": True, "paused": paused})


@app.route("/api/pause", methods=["POST"])
@api_login_required
def api_pause():
    set_paused(True)
    update_latest_state({"paused": True})
    return jsonify({"ok": True, "paused": True})


@app.route("/api/resume", methods=["POST"])
@api_login_required
def api_resume():
    set_paused(False)
    update_latest_state({"paused": False})
    return jsonify({"ok": True, "paused": False})


# -----------------------------------------------------------------------------
# STARTUP
# -----------------------------------------------------------------------------
init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)