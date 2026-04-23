"""
Core pose analysis engine using MediaPipe.
Calculates joint angles, posture scores, and exercise metrics.
"""

import math
import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose


def get_landmark_coords(landmarks, landmark_name):
    """Extract (x, y, z, visibility) from a named landmark."""
    lm = landmarks[mp_pose.PoseLandmark[landmark_name].value]
    return lm.x, lm.y, lm.z, lm.visibility


def calculate_angle(a, b, c):
    """
    Calculate angle at point B formed by segments BA and BC.
    Points are (x, y) tuples.
    Returns angle in degrees [0, 180].
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (
        math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2) + 1e-9
    )
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def landmark_visible(landmarks, *names, threshold=0.5):
    """Return True if all listed landmarks are visible above threshold."""
    for name in names:
        _, _, _, vis = get_landmark_coords(landmarks, name)
        if vis < threshold:
            return False
    return True


def get_joint_angles(landmarks):
    """Compute a dictionary of useful body joint angles."""
    angles = {}

    lm = landmarks

    def pt(name):
        x, y, _, _ = get_landmark_coords(lm, name)
        return (x, y)

    # --- Arms ---
    if landmark_visible(lm, "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"):
        angles["left_elbow"] = calculate_angle(pt("LEFT_SHOULDER"), pt("LEFT_ELBOW"), pt("LEFT_WRIST"))

    if landmark_visible(lm, "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"):
        angles["right_elbow"] = calculate_angle(pt("RIGHT_SHOULDER"), pt("RIGHT_ELBOW"), pt("RIGHT_WRIST"))

    if landmark_visible(lm, "LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"):
        angles["left_shoulder"] = calculate_angle(pt("LEFT_ELBOW"), pt("LEFT_SHOULDER"), pt("LEFT_HIP"))

    if landmark_visible(lm, "RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"):
        angles["right_shoulder"] = calculate_angle(pt("RIGHT_ELBOW"), pt("RIGHT_SHOULDER"), pt("RIGHT_HIP"))

    # --- Legs ---
    if landmark_visible(lm, "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"):
        angles["left_knee"] = calculate_angle(pt("LEFT_HIP"), pt("LEFT_KNEE"), pt("LEFT_ANKLE"))

    if landmark_visible(lm, "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"):
        angles["right_knee"] = calculate_angle(pt("RIGHT_HIP"), pt("RIGHT_KNEE"), pt("RIGHT_ANKLE"))

    if landmark_visible(lm, "LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"):
        angles["left_hip"] = calculate_angle(pt("LEFT_SHOULDER"), pt("LEFT_HIP"), pt("LEFT_KNEE"))

    if landmark_visible(lm, "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"):
        angles["right_hip"] = calculate_angle(pt("RIGHT_SHOULDER"), pt("RIGHT_HIP"), pt("RIGHT_KNEE"))

    # --- Torso / back alignment ---
    if landmark_visible(lm, "LEFT_SHOULDER", "LEFT_HIP", "LEFT_ANKLE"):
        angles["left_torso"] = calculate_angle(pt("LEFT_SHOULDER"), pt("LEFT_HIP"), pt("LEFT_ANKLE"))

    if landmark_visible(lm, "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_ANKLE"):
        angles["right_torso"] = calculate_angle(pt("RIGHT_SHOULDER"), pt("RIGHT_HIP"), pt("RIGHT_ANKLE"))

    return angles


# ---------------------------------------------------------------------------
# Posture analysers per exercise
# ---------------------------------------------------------------------------

class ExerciseAnalyzer:
    """
    Base class for all exercise analysers.
    Subclasses override `analyse` and set `display_name`, `joints_of_interest`.
    """
    display_name = "Exercise"
    joints_of_interest = []

    def __init__(self):
        self.rep_count = 0
        self.phase = "up"          # "up" | "down"
        self.feedback = []
        self.score = 100
        self._phase_lock = False

    def analyse(self, landmarks, angles):
        """
        Override in subclasses. Must return:
            score (int 0-100),
            feedback (list[str]),
            phase (str)
        """
        return 100, [], self.phase

    def update(self, landmarks, angles):
        self.score, self.feedback, new_phase = self.analyse(landmarks, angles)
        if new_phase != self.phase:
            if new_phase == "up" and self.phase == "down":
                self.rep_count += 1
            self.phase = new_phase
        return self.score, self.feedback

    def reset(self):
        self.rep_count = 0
        self.phase = "up"
        self.feedback = []
        self.score = 100


# ---- Squat ----------------------------------------------------------------

class SquatAnalyzer(ExerciseAnalyzer):
    display_name = "Squat"
    joints_of_interest = ["left_knee", "right_knee", "left_hip", "right_hip"]

    def analyse(self, landmarks, angles):
        feedback = []
        score = 100

        lk = angles.get("left_knee")
        rk = angles.get("right_knee")
        lh = angles.get("left_hip")
        rh = angles.get("right_hip")

        if lk is None and rk is None:
            return 0, ["Position yourself so your full body is visible."], self.phase

        knee_angle = lk if lk is not None else rk
        hip_angle = lh if lh is not None else rh

        # Determine phase
        phase = "down" if knee_angle < 120 else "up"

        # --- Knee depth ---
        if phase == "down":
            if knee_angle > 120:
                feedback.append("Go deeper -- aim for 90 deg or below at the knee.")
                score -= 20
            elif knee_angle < 60:
                feedback.append("Excellent depth!")
            else:
                feedback.append("Good depth.")

        # --- Back straightness (hip angle) ---
        if hip_angle is not None:
            if hip_angle < 150:
                feedback.append("Keep your chest up and back straight.")
                score -= 15
            else:
                feedback.append("Back alignment is good.")

        # --- Knee symmetry ---
        if lk is not None and rk is not None:
            diff = abs(lk - rk)
            if diff > 15:
                feedback.append(f"Uneven knees -- difference: {diff:.0f} deg. Balance your weight.")
                score -= 10

        score = max(0, score)
        return score, feedback, phase


# ---- Push-Up --------------------------------------------------------------

class PushUpAnalyzer(ExerciseAnalyzer):
    display_name = "Push-Up"
    joints_of_interest = ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"]

    def analyse(self, landmarks, angles):
        feedback = []
        score = 100

        le = angles.get("left_elbow")
        re = angles.get("right_elbow")
        lh = angles.get("left_hip")
        rh = angles.get("right_hip")

        elbow_angle = le if le is not None else re
        if elbow_angle is None:
            return 0, ["Could not detect arms. Face sideways for best results."], self.phase

        hip_angle = lh if lh is not None else rh
        phase = "down" if elbow_angle < 100 else "up"

        if phase == "down":
            if elbow_angle > 100:
                feedback.append("Lower your chest closer to the floor.")
                score -= 20
            elif elbow_angle < 70:
                feedback.append("Full range of motion -- great!")
            else:
                feedback.append("Good descent.")

        if hip_angle is not None:
            if hip_angle < 160:
                feedback.append("Keep hips in line -- avoid sagging or piking.")
                score -= 15
            else:
                feedback.append("Core and hips are aligned.")

        if le is not None and re is not None:
            diff = abs(le - re)
            if diff > 15:
                feedback.append(f"Arm symmetry off by {diff:.0f} deg -- even out your push.")
                score -= 10

        score = max(0, score)
        return score, feedback, phase


# ---- Bicep Curl -----------------------------------------------------------

class BicepCurlAnalyzer(ExerciseAnalyzer):
    display_name = "Bicep Curl"
    joints_of_interest = ["left_elbow", "right_elbow"]

    def analyse(self, landmarks, angles):
        feedback = []
        score = 100

        le = angles.get("left_elbow")
        re = angles.get("right_elbow")
        elbow_angle = le if le is not None else re

        if elbow_angle is None:
            return 0, ["Stand facing the camera with arms visible."], self.phase

        phase = "down" if elbow_angle > 140 else "up"

        if phase == "up":
            if elbow_angle > 50:
                feedback.append("Curl higher -- squeeze your bicep at the top.")
                score -= 15
            else:
                feedback.append("Full contraction -- excellent!")

        if phase == "down":
            if elbow_angle < 170:
                feedback.append("Fully extend your arm at the bottom.")
                score -= 10
            else:
                feedback.append("Good full extension.")

        if le is not None and re is not None:
            ls = angles.get("left_shoulder", 0)
            rs = angles.get("right_shoulder", 0)
            if abs(ls - rs) > 20:
                feedback.append("Keep shoulders still -- avoid swinging.")
                score -= 15

        score = max(0, score)
        return score, feedback, phase


# ---- Plank ----------------------------------------------------------------

class PlankAnalyzer(ExerciseAnalyzer):
    display_name = "Plank"
    joints_of_interest = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]

    def __init__(self):
        super().__init__()
        self.phase = "hold"

    def analyse(self, landmarks, angles):
        feedback = []
        score = 100

        lh = angles.get("left_hip")
        rh = angles.get("right_hip")
        lk = angles.get("left_knee")
        rk = angles.get("right_knee")

        hip_angle = lh if lh is not None else rh
        if hip_angle is None:
            return 0, ["Face sideways so your full body is visible."], "hold"

        # Ideal: hip angle close to 180 (straight line shoulder-hip-ankle)
        if hip_angle < 150:
            feedback.append("Hips too high -- bring them down to form a straight line.")
            score -= 25
        elif hip_angle > 200:
            feedback.append("Hips sagging -- engage your core and raise them slightly.")
            score -= 25
        else:
            feedback.append("Perfect plank position!")

        # Knee should be straight
        knee_angle = lk if lk is not None else rk
        if knee_angle is not None and knee_angle < 160:
            feedback.append("Keep your legs straight -- don't bend the knees.")
            score -= 15

        score = max(0, score)
        return score, feedback, "hold"


# ---- Shoulder Press -------------------------------------------------------

class ShoulderPressAnalyzer(ExerciseAnalyzer):
    display_name = "Shoulder Press"
    joints_of_interest = ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"]

    def analyse(self, landmarks, angles):
        feedback = []
        score = 100

        le = angles.get("left_elbow")
        re = angles.get("right_elbow")
        elbow_angle = le if le is not None else re

        if elbow_angle is None:
            return 0, ["Stand facing the camera with both arms visible."], self.phase

        phase = "up" if elbow_angle > 150 else "down"

        if phase == "up":
            if elbow_angle < 160:
                feedback.append("Extend fully overhead -- lock out your arms.")
                score -= 15
            else:
                feedback.append("Full lockout -- perfect!")

        if phase == "down":
            if elbow_angle > 100:
                feedback.append("Lower the weight more -- aim for ~90 deg at the elbow.")
                score -= 10
            else:
                feedback.append("Good starting position.")

        if le is not None and re is not None:
            diff = abs(le - re)
            if diff > 15:
                feedback.append(f"Arms uneven by {diff:.0f} deg -- press symmetrically.")
                score -= 10

        score = max(0, score)
        return score, feedback, phase


# ---- Deadlift -------------------------------------------------------------

class DeadliftAnalyzer(ExerciseAnalyzer):
    display_name = "Deadlift"
    joints_of_interest = ["left_hip", "right_hip", "left_knee", "right_knee"]

    def analyse(self, landmarks, angles):
        feedback = []
        score = 100

        lh = angles.get("left_hip")
        rh = angles.get("right_hip")
        lk = angles.get("left_knee")
        rk = angles.get("right_knee")

        hip_angle = lh if lh is not None else rh
        knee_angle = lk if lk is not None else rk

        if hip_angle is None:
            return 0, ["Stand sideways so your full body is visible."], self.phase

        phase = "up" if hip_angle > 160 else "down"

        if phase == "down":
            if hip_angle < 80:
                feedback.append("Hips too low -- this is more of a squat. Hip hinge more.")
                score -= 15
            elif hip_angle < 140:
                feedback.append("Good hinge position.")

            if knee_angle is not None and knee_angle < 140:
                feedback.append("Slight bend in knees is fine -- don't over-bend.")

        if phase == "up":
            if hip_angle < 170:
                feedback.append("Stand tall -- fully extend hips at the top.")
                score -= 10
            else:
                feedback.append("Full hip extension -- excellent lockout!")

        score = max(0, score)
        return score, feedback, phase


# ---- Registry -------------------------------------------------------------

EXERCISE_CLASSES = {
    "Squat":          SquatAnalyzer,
    "Push-Up":        PushUpAnalyzer,
    "Bicep Curl":     BicepCurlAnalyzer,
    "Plank":          PlankAnalyzer,
    "Shoulder Press": ShoulderPressAnalyzer,
    "Deadlift":       DeadliftAnalyzer,
}
