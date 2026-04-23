# Gym Pose Estimator

Real-time human pose analysis for gym exercises powered by **MediaPipe** and **OpenCV**.

## Features

| Feature | Details |
|---|---|
| **Exercises** | Squat, Push-Up, Bicep Curl, Plank, Shoulder Press, Deadlift |
| **Rep counting** | Automatic phase detection (up / down / hold) |
| **Posture score** | 0–100 per-frame quality score with colour gauge |
| **Joint angles** | Live angle bars for every tracked joint |
| **Feedback** | Contextual cues ("Go deeper", "Chest up", etc.) |
| **FPS display** | Real-time performance counter |

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

### Options

```
python app.py --camera 0      # default webcam
python app.py --camera 1      # external USB cam
python app.py --width 1280 --height 720
```

## Keyboard Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `R` | Reset rep counter |
| `SPACE` | Pause / Resume |
| `1` | Squat |
| `2` | Push-Up |
| `3` | Bicep Curl |
| `4` | Plank |
| `5` | Shoulder Press |
| `6` | Deadlift |

## Generate a Static Preview

```bash
python preview_generator.py   # saves preview.png
```

## Requirements

- Python 3.9+
- Webcam
- mediapipe, opencv-python, numpy
