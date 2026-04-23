# session_history.py

import json
import os

FILE_PATH = "sessions.json"


def save_session(session_data):
    """
    Saves a workout session to a JSON file.
    """
    # Load existing data
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new session
    data.append(session_data)

    # Save back to file
    with open(FILE_PATH, "w") as file:
        json.dump(data, file, indent=4)


def get_personal_bests():
    """
    Returns the best (max) value per exercise.
    Assumes session_data like:
    {
        "exercise": "Bench Press",
        "weight": 80
    }
    """
    if not os.path.exists(FILE_PATH):
        return {}

    with open(FILE_PATH, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            return {}

    bests = {}

    for session in data:
        exercise = session.get("exercise")
        weight = session.get("weight", 0)

        if exercise:
            if exercise not in bests or weight > bests[exercise]:
                bests[exercise] = weight

    return bests