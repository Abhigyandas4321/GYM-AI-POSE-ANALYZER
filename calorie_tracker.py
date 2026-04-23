# calorie_tracker.py
"""
Calorie burn estimator for Gym Pose Analyzer.
Uses MET-derived per-rep values scaled to body weight.
"""

# Calories per rep at 70 kg bodyweight (empirical MET estimates)
CALORIES_PER_REP: dict = {
    "Squat":          0.35,
    "Push-Up":        0.29,
    "Bicep Curl":     0.14,
    "Plank":          0.00,   # time-based only
    "Shoulder Press": 0.22,
    "Deadlift":       0.40,
}
PLANK_CAL_PER_SEC: float = 0.048   # ≈ 2.9 kcal / min at 70 kg


class CalorieTracker:
    def __init__(self, weight_kg: float = 70.0):
        self.weight_kg   = weight_kg
        self._scale      = weight_kg / 70.0
        self.total_kcal  : float = 0.0
        self.breakdown   : dict  = {}   # {exercise_name: kcal}

    # ── call once per completed rep ─────────────────────────────────────────
    def log_rep(self, exercise: str, reps: int = 1) -> float:
        cal = CALORIES_PER_REP.get(exercise, 0.20) * reps * self._scale
        self.total_kcal              += cal
        self.breakdown[exercise]      = self.breakdown.get(exercise, 0.0) + cal
        return round(cal, 3)

    # ── call while plank is held (pass elapsed seconds since last call) ─────
    def log_plank_time(self, seconds: float) -> float:
        cal = PLANK_CAL_PER_SEC * seconds * self._scale
        self.total_kcal          += cal
        self.breakdown["Plank"]   = self.breakdown.get("Plank", 0.0) + cal
        return round(cal, 3)

    @property
    def total(self) -> float:
        return round(self.total_kcal, 1)

    def reset(self):
        self.total_kcal = 0.0
        self.breakdown.clear()