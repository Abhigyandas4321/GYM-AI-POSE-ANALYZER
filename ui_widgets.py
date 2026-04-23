import cv2
import numpy as np
from datetime import datetime

# OpenCV uses BGR, not RGB
FONT = cv2.FONT_HERSHEY_DUPLEX

COLORS = {
    # Deep blue futuristic background
    "bg1": (22, 11, 5),       # dark navy top
    "bg2": (36, 20, 10),      # dark navy bottom

    # Panels
    "panel": (26, 16, 9),     # very dark blue
    "panel2": (34, 22, 13),   # slightly lighter blue
    "border": (95, 76, 56),   # steel-blue border
    "shine": (70, 52, 36),    # subtle panel top highlight
    "grid": (28, 18, 10),     # background grid

    # Text
    "white": (245, 245, 245),
    "muted": (175, 165, 155),

    # Accent colors
    "orange": (0, 145, 255),
    "green": (90, 255, 100),
    "red": (80, 80, 255),
    "yellow": (60, 210, 255),

    "warning": (0, 145, 255),
}


EXERCISE_TIPS = {
    "Squat": [
        "Lower your hips back and down.",
        "Keep your chest lifted.",
        "Keep your knees aligned with your toes."
    ],
    "Push-Up": [
        "Keep your body in a straight line.",
        "Lower your chest with control.",
        "Do not flare your elbows too much."
    ],
    "Bicep Curl": [
        "Keep your elbows close to your body.",
        "Avoid swinging your shoulders.",
        "Control both up and down movement."
    ],
    "Plank": [
        "Keep your hips level.",
        "Engage your core.",
        "Do not let your lower back sag."
    ],
    "Shoulder Press": [
        "Start with elbows around 90 degrees.",
        "Press upward in a controlled path.",
        "Avoid arching your lower back."
    ],
    "Deadlift": [
        "Keep your back neutral.",
        "Hinge from the hips.",
        "Keep the weight close to your body."
    ],
}


def get_exercise_tips(name):
    return EXERCISE_TIPS.get(name, [
        "Maintain proper posture.",
        "Move in a controlled manner.",
        "Focus on breathing and alignment."
    ])


def _gradient_background(h, w):
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        t = y / max(h - 1, 1)
        color = (
            int(COLORS["bg1"][0] * (1 - t) + COLORS["bg2"][0] * t),
            int(COLORS["bg1"][1] * (1 - t) + COLORS["bg2"][1] * t),
            int(COLORS["bg1"][2] * (1 - t) + COLORS["bg2"][2] * t),
        )
        bg[y, :] = color

    # subtle grid
    for x in range(0, w, 64):
        cv2.line(bg, (x, 0), (x, h), COLORS["grid"], 1)
    for y in range(0, h, 64):
        cv2.line(bg, (0, y), (w, y), COLORS["grid"], 1)

    # light texture dots
    for yy in range(20, h, 120):
        for xx in range(20, w, 120):
            cv2.circle(bg, (xx, yy), 1, (40, 24, 12), -1, cv2.LINE_AA)

    return bg


def _rounded_rect(img, x1, y1, x2, y2, color, r=16, thickness=-1):
    r = max(2, int(r))

    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)

        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def _draw_panel(img, x, y, w, h, title=None):
    overlay = img.copy()
    _rounded_rect(overlay, x, y, x + w, y + h, COLORS["panel"], r=18, thickness=-1)
    cv2.addWeighted(overlay, 0.95, img, 0.05, 0, img)

    # Main border
    _rounded_rect(img, x, y, x + w, y + h, COLORS["border"], r=18, thickness=2)

    # Inner soft border
    _rounded_rect(img, x + 2, y + 2, x + w - 2, y + h - 2, (55, 42, 28), r=16, thickness=1)

    # Top shine line
    cv2.line(img, (x + 18, y + 10), (x + w - 18, y + 10), COLORS["shine"], 1, cv2.LINE_AA)

    if title:
        cv2.putText(img, title, (x + 16, y + 32), FONT, 0.8, COLORS["orange"], 2, cv2.LINE_AA)


def _text_center(img, text, cx, cy, scale, color, thickness=2):
    size = cv2.getTextSize(text, FONT, scale, thickness)[0]
    x = int(cx - size[0] / 2)
    y = int(cy + size[1] / 2)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _fit_text_scale(text, max_width, base_scale=1.0, thickness=2, min_scale=0.35):
    scale = base_scale
    while scale > min_scale:
        w = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
        if w <= max_width:
            return scale
        scale -= 0.02
    return min_scale


def _put_text_fit(img, text, x, y, max_width, base_scale, color, thickness=2):
    scale = _fit_text_scale(text, max_width, base_scale, thickness)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return scale


def _put_text_center_fit(img, text, cx, cy, max_width, base_scale, color, thickness=2):
    scale = _fit_text_scale(text, max_width, base_scale, thickness)
    _text_center(img, text, cx, cy, scale, color, thickness)
    return scale


def _fit_image(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = (10, 8, 6)

    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _draw_progress_bar(img, x, y, w, h, value, max_value, color):
    value = max(0, min(value, max_value))
    ratio = 0 if max_value == 0 else value / max_value

    _rounded_rect(img, x, y, x + w, y + h, (45, 35, 24), r=5, thickness=-1)

    fill_w = int(w * ratio)
    if fill_w > 0:
        _rounded_rect(img, x, y, x + fill_w, y + h, color, r=5, thickness=-1)

    _rounded_rect(img, x, y, x + w, y + h, (88, 70, 52), r=5, thickness=1)


def _draw_ring(img, center, radius, value, max_value, color, title, sub):
    cx, cy = center

    # ring background
    cv2.circle(img, center, radius, (55, 42, 28), 8, cv2.LINE_AA)

    progress = 0 if max_value <= 0 else max(0.0, min(value / max_value, 1.0))
    sweep = int(360 * progress)
    if sweep > 0:
        cv2.ellipse(img, center, (radius, radius), -90, 0, sweep, color, 8, cv2.LINE_AA)

    _text_center(img, str(int(value)), cx, cy - 5, 1.0, color, 3)
    _text_center(img, title, cx, cy - radius - 18, 0.66, COLORS["white"], 2)
    _text_center(img, sub, cx, cy + radius + 18, 0.52, COLORS["muted"], 1)


def _wrap_draw_text(img, text, x, y, max_width, scale=0.62, color=(255, 255, 255), thickness=1):
    words = str(text).split()
    line = ""
    line_h = cv2.getTextSize("Ag", FONT, scale, thickness)[0][1] + 10

    for word in words:
        test = (line + " " + word).strip()
        if cv2.getTextSize(test, FONT, scale, thickness)[0][0] <= max_width:
            line = test
        else:
            if line:
                cv2.putText(img, line, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
                y += line_h
            line = word

    if line:
        cv2.putText(img, line, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
        y += line_h

    return y


def _draw_bullets(img, items, x, y, max_width, bullet_color, text_color):
    if not items:
        cv2.circle(img, (x, y - 6), 4, bullet_color, -1, cv2.LINE_AA)
        cv2.putText(img, "No feedback available.", (x + 12, y), FONT, 0.62, text_color, 1, cv2.LINE_AA)
        return

    for item in items:
        cv2.circle(img, (x, y - 6), 4, bullet_color, -1, cv2.LINE_AA)
        y = _wrap_draw_text(
            img,
            str(item),
            x + 12,
            y,
            max_width - 12,
            scale=0.68,
            color=text_color,
            thickness=1,
        )
        y += 6


def _draw_tab_label(img, name, cx, y, tab_w, color):
    max_w = tab_w - 18

    if cv2.getTextSize(name, FONT, 0.62, 2)[0][0] <= max_w:
        _text_center(img, name, cx, y + 52, 0.62, color, 2)
        return

    parts = name.split()

    if len(parts) == 2:
        line1, line2 = parts[0], parts[1]
    elif len(parts) > 2:
        mid = len(parts) // 2
        line1 = " ".join(parts[:mid])
        line2 = " ".join(parts[mid:])
    else:
        line1, line2 = name, ""

    s1 = _fit_text_scale(line1, max_w, 0.50, 2, 0.38)
    _text_center(img, line1, cx, y + 42, s1, color, 2)

    if line2:
        s2 = _fit_text_scale(line2, max_w, 0.50, 2, 0.38)
        _text_center(img, line2, cx, y + 64, s2, color, 2)


def _draw_tabs(img, x, y, w, h, exercise_names, selected_idx):
    cv2.putText(img, "SELECT EXERCISE", (x, y - 12), FONT, 0.70, COLORS["orange"], 2, cv2.LINE_AA)
    cv2.putText(img, "(Press Number Key)", (x + 190, y - 12), FONT, 0.48, COLORS["muted"], 1, cv2.LINE_AA)

    gap = 8
    tab_w = (w - gap * (len(exercise_names) - 1)) // len(exercise_names)

    for i, name in enumerate(exercise_names):
        tx = x + i * (tab_w + gap)
        active = (i == selected_idx)

        _rounded_rect(img, tx, y, tx + tab_w, y + h, COLORS["panel2"], r=14, thickness=-1)
        _rounded_rect(
            img,
            tx, y, tx + tab_w, y + h,
            COLORS["orange"] if active else COLORS["border"],
            r=14,
            thickness=2
        )

        if active:
            cv2.line(img, (tx + 14, y + h - 5), (tx + tab_w - 14, y + h - 5), COLORS["orange"], 2, cv2.LINE_AA)

        _text_center(
            img,
            str(i + 1),
            tx + tab_w // 2,
            y + 20,
            0.88,
            COLORS["orange"] if active else COLORS["white"],
            2
        )

        _draw_tab_label(img, name, tx + tab_w // 2, y, tab_w, COLORS["white"])


def _draw_joint_rows(img, x, y, w, angles):
    if not angles:
        cv2.putText(img, "No joint data available", (x, y + 10), FONT, 0.62, COLORS["muted"], 1, cv2.LINE_AA)
        return

    row_h = 58
    for i, item in enumerate(angles[:6]):
        yy = y + i * row_h
        label = str(item["label"])
        value = float(item["value"])
        color = item.get("color", COLORS["green"])

        lbl_scale = _fit_text_scale(label, w - 88, 0.56, 1, 0.42)
        cv2.putText(img, label, (x, yy), FONT, lbl_scale, COLORS["white"], 1, cv2.LINE_AA)
        cv2.putText(img, f"{int(value)}°", (x + w - 72, yy), FONT, 0.76, COLORS["white"], 2, cv2.LINE_AA)
        _draw_progress_bar(img, x, yy + 12, w, 10, min(value, 180), 180, color)


def draw_no_pose(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, h - 75), (w - 20, h - 20), (18, 12, 8), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    text = "Position yourself so your full body is visible"
    scale = _fit_text_scale(text, w - 80, 0.80, 2, 0.50)
    cv2.putText(
        frame,
        text,
        (40, h - 42),
        FONT,
        scale,
        COLORS["green"],
        2,
        cv2.LINE_AA
    )


def render_dashboard(camera_frame, state, size=(1600, 900)):
    W, H = size
    canvas = _gradient_background(H, W)

    left_x, left_w = 20, 250
    center_x, center_w = 290, 980
    right_x, right_w = 1290, 290

    # Top panels
    _draw_panel(canvas, left_x, 20, left_w, 150)
    _draw_panel(canvas, center_x, 20, center_w, 150)
    _draw_panel(canvas, right_x, 35, right_w, 105)

    # Left logo
    _put_text_fit(canvas, "GYM AI", left_x + 20, 84, left_w - 40, 1.45, COLORS["orange"], 4)
    _put_text_fit(canvas, "POSE", left_x + 20, 120, left_w - 40, 0.86, COLORS["white"], 2)
    _put_text_fit(canvas, "ANALYZER", left_x + 20, 150, left_w - 40, 0.86, COLORS["white"], 2)

    # Exercise selector
    _draw_tabs(
        canvas,
        center_x + 16,
        55,
        center_w - 32,
        86,
        state["exercise_names"],
        state["selected_index"]
    )

    # Time/date
    now = datetime.now()
    cv2.putText(canvas, now.strftime("%I:%M:%S %p"), (right_x + 24, 80), FONT, 0.92, COLORS["white"], 2, cv2.LINE_AA)
    cv2.putText(canvas, now.strftime("%d %b %Y"), (right_x + 24, 124), FONT, 0.82, COLORS["white"], 2, cv2.LINE_AA)

    # Left panel: exercise
    _draw_panel(canvas, left_x, 185, left_w, 200)
    cv2.putText(canvas, "EXERCISE", (left_x + 18, 225), FONT, 0.75, COLORS["muted"], 1, cv2.LINE_AA)
    _put_text_fit(canvas, str(state["exercise"]).upper(), left_x + 18, 272, left_w - 36, 1.10, COLORS["white"], 3)
    cv2.putText(canvas, "PHASE", (left_x + 18, 314), FONT, 0.75, COLORS["muted"], 1, cv2.LINE_AA)

    bx, by, bw, bh = left_x + 18, 332, 210, 44
    _rounded_rect(canvas, bx, by, bx + bw, by + bh, COLORS["panel2"], r=12, thickness=-1)
    _rounded_rect(canvas, bx, by, bx + bw, by + bh, COLORS["orange"], r=12, thickness=2)
    _put_text_fit(canvas, str(state["phase"]).upper(), bx + 14, by + 30, bw - 20, 0.95, COLORS["orange"], 2)

    # Left panel: score/reps
    _draw_panel(canvas, left_x, 410, left_w, 180)
    _draw_ring(canvas, (left_x + 72, 500), 48, state["score"], 100, COLORS["green"], "SCORE", "/100")
    _draw_ring(
        canvas,
        (left_x + 182, 500),
        48,
        state["reps"],
        state.get("rep_goal", 20),
        COLORS["orange"],
        "REPS",
        f"/{state.get('rep_goal', 20)}"
    )

    # Left panel: feedback
    _draw_panel(canvas, left_x, 610, left_w, 220, "FEEDBACK")
    _draw_bullets(
        canvas,
        state.get("feedback", []),
        left_x + 18,
        660,
        left_w - 36,
        COLORS["green"],
        COLORS["green"]
    )

    # FPS
    _draw_panel(canvas, left_x, 845, left_w, 35)
    cv2.putText(canvas, f"FPS: {int(state['fps'])}", (left_x + 18, 870), FONT, 0.8, COLORS["orange"], 2, cv2.LINE_AA)

    # Center camera
    _draw_panel(canvas, center_x, 185, center_w, 540)
    cam = _fit_image(camera_frame, center_w - 20, 520)
    canvas[195:715, center_x + 10:center_x + 10 + center_w - 20] = cam
    _rounded_rect(canvas, center_x, 185, center_x + center_w, 725, COLORS["border"], r=18, thickness=2)

    # Bottom status
    _draw_panel(canvas, center_x, 745, center_w, 115)
    _put_text_center_fit(
        canvas,
        str(state.get("status_title", "")),
        center_x + center_w // 2,
        785,
        center_w - 60,
        0.92,
        COLORS["green"],
        2
    )
    _put_text_center_fit(
        canvas,
        str(state.get("status_subtitle", "")),
        center_x + center_w // 2,
        825,
        center_w - 80,
        0.72,
        COLORS["white"],
        1
    )

    # Right panel: joint angles
    _draw_panel(canvas, right_x, 185, right_w, 420, "JOINT ANGLES")
    _draw_joint_rows(canvas, right_x + 18, 238, right_w - 36, state.get("angles", []))

    # Right panel: tips
    _draw_panel(canvas, right_x, 625, right_w, 235, "TIPS")
    _draw_bullets(
        canvas,
        state.get("tips", []),
        right_x + 18,
        675,
        right_w - 36,
        COLORS["yellow"],
        COLORS["white"]
    )

    return canvas