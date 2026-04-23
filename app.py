
#!/usr/bin/env python3
"""
Gym Pose Analyzer + AI Voice Assistant
Cleaned OpenCV dashboard version
"""

import argparse
import asyncio
import inspect
import os
import sys
import tempfile
import threading
import time
import uuid

import cv2
import edge_tts
import mediapipe as mp
import pygame

from pose_analyzer import get_joint_angles, EXERCISE_CLASSES
from ui_widgets import render_dashboard, draw_no_pose, get_exercise_tips, COLORS
from calorie_tracker import CalorieTracker
from session_history import save_session, get_personal_bests


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VOICE = "en-IN-PrabhatNeural"
EXERCISE_NAMES = list(EXERCISE_CLASSES.keys())

DASHBOARD_W = 1600
DASHBOARD_H = 900

EXERCISE_MET = {
    "Squat": 5.0,
    "Push-Up": 8.0,
    "Bicep Curl": 3.5,
    "Plank": 3.0,
    "Shoulder Press": 4.5,
    "Deadlift": 6.0,
}


# ---------------------------------------------------------------------------
# AUDIO / VOICE
# ---------------------------------------------------------------------------
AUDIO_ENABLED = True
is_speaking = False

try:
    pygame.mixer.init()
except Exception as e:
    AUDIO_ENABLED = False
    print(f"[WARN] Audio disabled: {e}")


def speak(text: str):
    global is_speaking

    if not AUDIO_ENABLED or not text or is_speaking:
        return

    def run():
        global is_speaking
        is_speaking = True

        filename = os.path.join(
            tempfile.gettempdir(),
            f"voice_{uuid.uuid4().hex}.mp3"
        )

        try:
            async def _speak():
                communicate = edge_tts.Communicate(text, VOICE)
                await communicate.save(filename)

            asyncio.run(_speak())

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            print(f"[WARN] Voice error: {e}")

        finally:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception:
                pass

            is_speaking = False

    threading.Thread(target=run, daemon=True).start()


# ---------------------------------------------------------------------------
# MEDIAPIPE SETUP
# ---------------------------------------------------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

CUSTOM_LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(80, 255, 120), thickness=2, circle_radius=5
)
CUSTOM_CONNECTION_STYLE = mp_drawing.DrawingSpec(
    color=(245, 245, 245), thickness=2
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def get_weight_kg(tracker, default=70.0):
    try:
        value = getattr(tracker, "weight_kg", default)
        return float(value)
    except Exception:
        return float(default)


def estimate_calories(weight_kg: float, exercise_name: str, active_seconds: float) -> float:
    met = EXERCISE_MET.get(exercise_name, 4.0)
    kcal_per_min = met * 3.5 * weight_kg / 200.0
    return kcal_per_min * (active_seconds / 60.0)


def normalize_best_map(raw, exercise_names):
    result = {name: 0 for name in exercise_names}

    if not isinstance(raw, dict):
        return result

    for key, value in raw.items():
        if not isinstance(key, str):
            continue

        for name in exercise_names:
            if key.strip().lower() == name.strip().lower():
                try:
                    result[name] = int(value or 0)
                except Exception:
                    result[name] = 0
                break

    return result


def safe_get_personal_best_map(exercise_names):
    # Case 1: get_personal_bests() returns a dict
    try:
        raw = get_personal_bests()
        bests = normalize_best_map(raw, exercise_names)
        if any(v > 0 for v in bests.values()):
            return bests
    except TypeError:
        pass
    except Exception as e:
        print(f"[WARN] Could not load personal bests (bulk): {e}")

    # Case 2: get_personal_bests(exercise_name) returns int or dict
    result = {name: 0 for name in exercise_names}
    for name in exercise_names:
        try:
            raw = get_personal_bests(name)
            if isinstance(raw, (int, float)):
                result[name] = int(raw)
            elif isinstance(raw, dict):
                normalized = normalize_best_map(raw, exercise_names)
                if name in normalized:
                    result[name] = normalized[name]
        except Exception:
            pass

    return result


def safe_save_session(summary: dict):
    """
    Tries multiple common calling styles so this file can work
    even if your session_history.py uses a slightly different API.
    """
    # 1) save_session(summary_dict)
    try:
        return save_session(summary)
    except TypeError:
        pass
    except Exception as e:
        print(f"[WARN] save_session(summary) failed: {e}")
        return

    # 2) save_session(**kwargs) based on signature
    try:
        sig = inspect.signature(save_session)
        params = sig.parameters

        field_map = {
            "exercise": summary["exercise_name"],
            "exercise_name": summary["exercise_name"],
            "name": summary["exercise_name"],

            "reps": summary["reps"],
            "rep_count": summary["reps"],
            "total_reps": summary["reps"],

            "score": summary["avg_score"],
            "avg_score": summary["avg_score"],
            "average_score": summary["avg_score"],

            "calories": summary["calories"],
            "kcal": summary["calories"],

            "duration": summary["duration_sec"],
            "duration_sec": summary["duration_sec"],
            "active_duration": summary["duration_sec"],
            "active_duration_sec": summary["duration_sec"],
            "session_duration": summary["total_session_duration_sec"],
            "total_session_duration": summary["total_session_duration_sec"],
            "total_session_duration_sec": summary["total_session_duration_sec"],

            "timestamp": summary["timestamp"],
            "date": summary["timestamp"],

            "personal_best": summary["personal_best"],
        }

        kwargs = {}
        for param_name in params:
            if param_name in field_map:
                kwargs[param_name] = field_map[param_name]

        if kwargs:
            return save_session(**kwargs)

    except Exception as e:
        print(f"[WARN] save_session(**kwargs) failed: {e}")

    # 3) Common positional fallback
    try:
        return save_session(
            summary["exercise_name"],
            summary["reps"],
            summary["avg_score"],
            summary["calories"],
            summary["duration_sec"],
        )
    except Exception as e:
        print(f"[WARN] save_session positional fallback failed: {e}")


def save_all_session_summaries(
    exercise_names,
    session_rep_map,
    session_score_map,
    active_seconds_map,
    weight_kg,
    total_session_duration_sec,
    personal_bests,
):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for exercise in exercise_names:
        reps = int(session_rep_map.get(exercise, 0))
        scores = session_score_map.get(exercise, [])
        active_sec = float(active_seconds_map.get(exercise, 0.0))

        if reps <= 0 and active_sec < 1 and not scores:
            continue

        avg_score = float(sum(scores) / len(scores)) if scores else 0.0
        calories = estimate_calories(weight_kg, exercise, active_sec)
        personal_best = max(int(personal_bests.get(exercise, 0)), reps)

        summary = {
            "exercise_name": exercise,
            "reps": reps,
            "avg_score": round(avg_score, 2),
            "calories": round(calories, 2),
            "duration_sec": round(active_sec, 2),
            "total_session_duration_sec": round(total_session_duration_sec, 2),
            "timestamp": timestamp,
            "personal_best": personal_best,
        }

        safe_save_session(summary)
        print(
            f"[SESSION] {exercise}: reps={reps}, "
            f"avg_score={avg_score:.1f}, kcal={calories:.1f}, active={active_sec:.1f}s"
        )


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


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def run(camera_index: int = 0, width: int = 1280, height: int = 720):
    global is_speaking

    last_spoken_rep = -1
    last_feedback_time = 0.0
    last_no_pose_time = 0.0

    calorie_tracker = CalorieTracker(weight_kg=70)
    weight_kg = get_weight_kg(calorie_tracker, 70.0)
    personal_bests = safe_get_personal_best_map(EXERCISE_NAMES)

    session_start_time = time.time()
    session_rep_map = {name: 0 for name in EXERCISE_NAMES}
    session_score_map = {name: [] for name in EXERCISE_NAMES}
    active_seconds_map = {name: 0.0 for name in EXERCISE_NAMES}
    rep_snapshot_map = {name: 0 for name in EXERCISE_NAMES}
    last_score_map = {name: 0.0 for name in EXERCISE_NAMES}

    analysers = {name: cls() for name, cls in EXERCISE_CLASSES.items()}
    current_exercise = EXERCISE_NAMES[0]
    paused = False

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        sys.exit(1)

    prev_loop_time = time.time()
    fps = 0.0

    cv2.namedWindow("Gym Pose Analyzer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gym Pose Analyzer", DASHBOARD_W, DASHBOARD_H)

    try:
        with mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while True:
                loop_now = time.time()
                frame_dt = max(loop_now - prev_loop_time, 1e-6)
                prev_loop_time = loop_now

                if fps == 0.0:
                    fps = 1.0 / frame_dt
                else:
                    fps = 0.9 * fps + 0.1 * (1.0 / frame_dt)

                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                camera_view = frame.copy()

                analyser = analysers[current_exercise]

                pose_detected = False
                score = last_score_map[current_exercise]
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

                        last_score_map[current_exercise] = float(score)
                        session_score_map[current_exercise].append(float(score))
                        active_seconds_map[current_exercise] += frame_dt

                        # Track session reps by delta
                        current_rep_count = int(analyser.rep_count)
                        prev_rep_snapshot = rep_snapshot_map[current_exercise]

                        if current_rep_count > prev_rep_snapshot:
                            delta = current_rep_count - prev_rep_snapshot
                            session_rep_map[current_exercise] += delta
                            rep_snapshot_map[current_exercise] = current_rep_count
                        elif current_rep_count < prev_rep_snapshot:
                            rep_snapshot_map[current_exercise] = current_rep_count

                        # Voice: rep count
                        if analyser.rep_count != last_spoken_rep:
                            last_spoken_rep = analyser.rep_count
                            if analyser.rep_count > 0:
                                speak(f"Rep {analyser.rep_count}")

                        # Voice: feedback every 5 sec
                        if time.time() - last_feedback_time > 5:
                            if feedback:
                                speak(feedback[0])
                                last_feedback_time = time.time()
                            elif analyser.rep_count == 0:
                                speak("Start your exercise")
                                last_feedback_time = time.time()

                        if feedback:
                            status_title = feedback[0].upper()
                            status_subtitle = "Adjust your posture and continue"
                        else:
                            status_title = f"{current_exercise.upper()}  |  {phase}"
                            status_subtitle = "Good form. Keep going."

                    else:
                        draw_no_pose(camera_view)

                        if time.time() - last_no_pose_time > 5:
                            speak("Position yourself so your full body is visible")
                            last_no_pose_time = time.time()

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
                    status_subtitle = "Press SPACE to resume"

                current_calories = estimate_calories(
                    weight_kg,
                    current_exercise,
                    active_seconds_map[current_exercise]
                )

                angle_items = build_angle_items(analyser, angles)
                personal_best = int(personal_bests.get(current_exercise, 0))
                rep_goal = max(20, personal_best) if personal_best > 0 else 20

                ui_feedback = feedback if feedback else (
                    [
                        "Good posture detected.",
                        f"Estimated calories: {current_calories:.1f} kcal"
                    ] if pose_detected else [
                        "Position yourself so your full body is visible."
                    ]
                )

                ui_state = {
                    "exercise_names": EXERCISE_NAMES,
                    "selected_index": EXERCISE_NAMES.index(current_exercise),
                    "exercise": current_exercise,
                    "phase": phase,
                    "score": score,
                    "reps": analyser.rep_count,
                    "rep_goal": rep_goal,
                    "fps": fps,
                    "feedback": ui_feedback,
                    "tips": get_exercise_tips(current_exercise),
                    "angles": angle_items,
                    "status_title": status_title,
                    "status_subtitle": status_subtitle,
                }

                dashboard = render_dashboard(
                    camera_view,
                    ui_state,
                    size=(DASHBOARD_W, DASHBOARD_H)
                )

                cv2.imshow("Gym Pose Analyzer", dashboard)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

                elif key == ord("r"):
                    analysers[current_exercise].reset()
                    rep_snapshot_map[current_exercise] = 0
                    last_spoken_rep = 0
                    last_score_map[current_exercise] = 0.0
                    speak("Reset done")

                elif key == ord(" "):
                    paused = not paused
                    speak("Paused" if paused else "Resumed")

                elif ord("1") <= key <= ord("6"):
                    idx = key - ord("1")
                    if idx < len(EXERCISE_NAMES):
                        current_exercise = EXERCISE_NAMES[idx]
                        last_spoken_rep = analysers[current_exercise].rep_count
                        speak(current_exercise)

    finally:
        total_session_duration_sec = time.time() - session_start_time

        try:
            save_all_session_summaries(
                exercise_names=EXERCISE_NAMES,
                session_rep_map=session_rep_map,
                session_score_map=session_score_map,
                active_seconds_map=active_seconds_map,
                weight_kg=weight_kg,
                total_session_duration_sec=total_session_duration_sec,
                personal_bests=personal_bests,
            )
        except Exception as e:
            print(f"[WARN] Failed to save session summaries: {e}")

        try:
            cap.release()
        except Exception:
            pass

        cv2.destroyAllWindows()

        if AUDIO_ENABLED:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            try:
                pygame.mixer.quit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    run(args.camera, args.width, args.height)


if __name__ == "__main__":
    main()