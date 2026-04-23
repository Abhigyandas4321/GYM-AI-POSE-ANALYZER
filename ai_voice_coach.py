import asyncio
import edge_tts
import time
import os

VOICE = "en-IN-PrabhatNeural"

async def speak(text):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.run()

def speak_sync(text):
    asyncio.run(speak(text))


def voice_loop():
    last_rep = -1

    while True:
        try:
            if os.path.exists("rep_data.txt"):
                with open("rep_data.txt", "r") as f:
                    rep = int(f.read().strip())

                if rep != last_rep:
                    last_rep = rep

                    if rep > 0:
                        speak_sync(f"Rep {rep}")

                    if rep % 5 == 0 and rep != 0:
                        speak_sync("Great job, keep going")

        except:
            pass

        time.sleep(1)


if __name__ == "__main__":
    speak_sync("AI gym coach started")
    voice_loop()