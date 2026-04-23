"""
Generates a static PNG preview of the Gym Pose Estimator UI.
Simulates what the app looks like mid-workout.
"""

import cv2
import numpy as np
import mediapipe as mp
import math

from ui_widgets import (
    draw_left_panel, draw_top_bar,
    draw_exercise_selector, COLORS,
    put_text_shadow, FONT, FONT_BOLD,
    draw_rounded_rect, draw_score_gauge,
)

W, H = 1280, 720


def make_preview():
    # Create a realistic-looking background (gym floor texture simulation)
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:] = (28, 28, 38)

    # Add subtle grid/floor lines
    for i in range(0, W, 40):
        cv2.line(bg, (i, 0), (i, H), (35, 35, 48), 1)
    for j in range(0, H, 40):
        cv2.line(bg, (0, j), (W, j), (35, 35, 48), 1)

    # Draw a realistic squat position skeleton
    cx = int(W * 0.58)
    connection_color = (180, 180, 220)
    landmark_color   = (0, 220, 140)
    accent2          = COLORS["accent2"]

    # Key body points for a deep squat (front view)
    # Head
    head      = (cx, 130)
    nose      = (cx, 150)
    # Shoulders
    l_shoulder = (cx - 75, 210)
    r_shoulder = (cx + 75, 210)
    # Elbows (arms out front, squat position)
    l_elbow    = (cx - 120, 300)
    r_elbow    = (cx + 120, 300)
    # Wrists
    l_wrist    = (cx - 110, 340)
    r_wrist    = (cx + 110, 340)
    # Hips (lower because squatting)
    l_hip      = (cx - 55, 370)
    r_hip      = (cx + 55, 370)
    # Knees (wide and bent ~95 deg)
    l_knee     = (cx - 130, 490)
    r_knee     = (cx + 130, 490)
    # Ankles
    l_ankle    = (cx - 90, 600)
    r_ankle    = (cx + 90, 600)

    # Background shadow figure
    def line_s(a, b, t=6):
        cv2.line(bg, a, b, (45, 45, 60), t + 4)
    line_s(head, l_shoulder); line_s(head, r_shoulder)
    line_s(l_shoulder, r_shoulder)
    line_s(l_shoulder, l_elbow); line_s(l_elbow, l_wrist)
    line_s(r_shoulder, r_elbow); line_s(r_elbow, r_wrist)
    line_s(l_shoulder, l_hip); line_s(r_shoulder, r_hip)
    line_s(l_hip, r_hip)
    line_s(l_hip, l_knee); line_s(l_knee, l_ankle)
    line_s(r_hip, r_knee); line_s(r_knee, r_ankle)

    # Draw MediaPipe-style skeleton (bright)
    connections = [
        (head, l_shoulder), (head, r_shoulder),
        (l_shoulder, r_shoulder),
        (l_shoulder, l_elbow), (l_elbow, l_wrist),
        (r_shoulder, r_elbow), (r_elbow, r_wrist),
        (l_shoulder, l_hip), (r_shoulder, r_hip),
        (l_hip, r_hip),
        (l_hip, l_knee), (l_knee, l_ankle),
        (r_hip, r_knee), (r_knee, r_ankle),
    ]
    for a, b in connections:
        cv2.line(bg, a, b, connection_color, 2, cv2.LINE_AA)

    all_landmarks = [
        head, nose, l_shoulder, r_shoulder,
        l_elbow, r_elbow, l_wrist, r_wrist,
        l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle
    ]
    for pt in all_landmarks:
        cv2.circle(bg, pt, 7, landmark_color, -1, cv2.LINE_AA)
        cv2.circle(bg, pt, 9, landmark_color, 2, cv2.LINE_AA)

    # Head circle
    cv2.circle(bg, (cx, 100), 30, (45, 45, 60), -1)
    cv2.circle(bg, (cx, 100), 30, landmark_color, 2, cv2.LINE_AA)

    # Knee angle arcs and annotations
    for knee, hip, ankle, label, side in [
        (l_knee, l_hip, l_ankle, "95 deg", "left"),
        (r_knee, r_hip, r_ankle, "97 deg", "right"),
    ]:
        cv2.circle(bg, knee, 14, accent2, 2, cv2.LINE_AA)
        if side == "left":
            put_text_shadow(bg, label, (knee[0] - 80, knee[1] + 5), 0.55, accent2, 1, FONT_BOLD)
        else:
            put_text_shadow(bg, label, (knee[0] + 18, knee[1] + 5), 0.55, accent2, 1, FONT_BOLD)

    # Hip angle annotation
    cv2.circle(bg, l_hip, 12, (0, 210, 210), 2, cv2.LINE_AA)
    put_text_shadow(bg, "155 deg", (l_hip[0] - 85, l_hip[1] - 8), 0.45, (0, 210, 210), 1, FONT_BOLD)

    # --- Draw the UI panel ---
    angles = {
        "left_knee": 95,
        "right_knee": 97,
        "left_hip": 155,
        "right_hip": 158,
    }
    from pose_analyzer import SquatAnalyzer
    sa = SquatAnalyzer()
    sa.rep_count = 8
    sa.phase = "down"

    feedback = [
        "Good depth -- knees at 95 deg",
        "Back alignment is good.",
        "Even knee tracking.",
    ]

    draw_left_panel(
        bg,
        "Squat",
        sa.rep_count,
        85,
        "down",
        feedback,
        angles,
        sa.joints_of_interest,
    )

    draw_top_bar(bg, 29.8)

    from pose_analyzer import EXERCISE_CLASSES
    draw_exercise_selector(bg, list(EXERCISE_CLASSES.keys()), 0)

    # Watermark
    put_text_shadow(bg, "PREVIEW MODE -- Webcam not active",
                    (W // 2 - 200, H - 55), 0.5, COLORS["text_dim"], 1)

    cv2.imwrite("preview.png", bg)
    print("[INFO] Preview saved to preview.png")


if __name__ == "__main__":
    make_preview()
