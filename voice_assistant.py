import pyttsx3
import threading

engine = pyttsx3.init()

# Set voice properties (optional)
engine.setProperty('rate', 170)   # speed
engine.setProperty('volume', 1.0)

def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()

    # Run in separate thread (important for real-time app)
    threading.Thread(target=_speak, daemon=True).start()