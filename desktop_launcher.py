import threading
import time
import requests
import webview

from waitress import serve
from web_server import app


HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"


def run_server():
    serve(app, host=HOST, port=PORT, threads=8)


def wait_for_server(url, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.25)
    return False


if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    if not wait_for_server(URL):
        raise RuntimeError("Flask server did not start in time.")

    webview.create_window(
        "Gym AI Pose Analyzer",
        URL,
        width=1600,
        height=950,
        min_size=(1200, 750),
        resizable=True,
    )

    webview.start(debug=False)