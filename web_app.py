import time
import cv2
import av
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from pose_analyzer import get_joint_angles, EXERCISE_CLASSES
from ui_widgets import render_dashboard, draw_no_pose, get_exercise_tips, COLORS


# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gym AI Pose Analyzer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Hide Streamlit default UI */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* App background */
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255,138,0,0.10), transparent 25%),
                radial-gradient(circle at top right, rgba(0,180,255,0.08), transparent 20%),
                linear-gradient(180deg, #07101c 0%, #09111f 45%, #0b1220 100%);
            color: #f4f7fb;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1322 0%, #0d1628 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }

        /* Hero */
        .hero-wrap {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
            padding: 22px 26px;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(10,20,35,0.95), rgba(13,19,32,0.92));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            margin-bottom: 18px;
        }

        .hero-left h1 {
            margin: 0;
            font-size: 2.3rem;
            line-height: 1.1;
            color: #f8fbff;
        }

        .hero-left p {
            margin: 10px 0 0 0;
            color: #aeb8c7;
            font-size: 1rem;
        }

        .hero-kicker {
            color: #ff8c00;
            font-weight: 700;
            letter-spacing: 1px;
            font-size: 0.82rem;
            margin-bottom: 6px;
            text-transform: uppercase;
        }

        .hero-badges {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .hero-badge {
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            color: #e7edf7;
            font-size: 0.92rem;
            white-space: nowrap;
        }

        /* Section titles */
        .section-title {
            color: #ff8c00;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin: 10px 0 12px 0;
        }

        /* Info cards */
        .info-card {
            background: linear-gradient(180deg, rgba(10,20,34,0.92), rgba(12,18,30,0.92));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px 18px;
            min-height: 120px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.22);
        }

        .info-card h3 {
            margin: 0 0 8px 0;
            color: #f5f8fd;
            font-size: 1.05rem;
        }

        .info-card p {
            margin: 0;
            color: #adb7c5;
            font-size: 0.95rem;
            line-height: 1.55;
        }

        /* Sidebar brand */
        .brand-box {
            padding: 16px 18px;
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(12,19,32,0.95), rgba(9,15,26,0.92));
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 16px;
        }

        .brand-title {
            color: #ff8c00;
            font-size: 1.9rem;
            font-weight: 900;
            line-height: 1;
            margin: 0;
        }

        .brand-sub {
            color: #f3f6fb;
            font-size: 0.95rem;
            margin-top: 6px;
            letter-spacing: 0.5px;
        }

        /* Widgets */
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            background: linear-gradient(135deg, #ff8c00, #ff6a00);
            color: white;
            border: none;
            font-weight: 700;
            padding: 0.7rem 1rem;
            box-shadow: 0 6px 18px rgba(255,140,0,0.25);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #ff9d26, #ff7d1a);
            color: white;
        }

        .stSelectbox label,
        .stCheckbox label,
        .stMarkdown,
        .stCaption {
            color: #f3f7fc !important;
        }

        div[data-baseweb="select"] > div {
            background: #0d1626 !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            border-radius: 12px !important;
            color: white !important;
        }

        /* Stream area hint */
        .stream-note {
            color: #aeb8c7;
            font-size: 0.92rem;
            margin-top: 6px;
            margin-bottom: 8px;
        }

        /* Tip list */
        .tip-item {
            color: #dce3ee;
            margin-bottom: 10px;
            line-height: 1.5;
        }

        /* Responsive */
        @media (max-width: 900px) {
            .hero-wrap {
                flex-direction: column;
                align-items: flex-start;
            }
            .hero-badges {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
EXERCISE_NAMES = list(EXERCISE_CLASSES.keys())

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

CUSTOM_LANDMARK_STYLE = mp_drawing.DrawingSpec(
    color=(80, 255, 120), thickness=2, circle_radius=5
)
CUSTOM_CONNECTION_STYLE = mp_drawing.DrawingSpec(
    color=(245, 245, 245), thickness=2
)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# VIDEO PROCESSOR
# ---------------------------------------------------------
class GymProcessor(VideoProcessorBase):
    def __init__(self):
        self.analysers = {name: cls() for name, cls in EXERCISE_CLASSES.items()}
        self.current_exercise = EXERCISE_NAMES[0]
        self.pose = mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.prev_time = time.time()
        self.fps = 0.0
        self.paused = False

    def set_exercise(self, name):
        if name in self.analysers:
            self.current_exercise = name

    def reset_current(self):
        self.analysers[self.current_exercise].reset()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        camera_view = img.copy()
        analyser = self.analysers[self.current_exercise]

        pose_detected = False
        score = 0
        feedback = []
        angles = {}
        phase = "READY"

        status_title = "POSITION YOURSELF SO YOUR FULL BODY IS VISIBLE"
        status_subtitle = "Make sure the entire body is in the camera frame"

        if not self.paused:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

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

                if feedback:
                    status_title = feedback[0].upper()
                    status_subtitle = "Adjust your posture and continue"
                else:
                    status_title = f"{self.current_exercise.upper()} | {phase}"
                    status_subtitle = "Good form. Keep going."
            else:
                draw_no_pose(camera_view)
        else:
            phase = "PAUSED"
            status_title = "WORKOUT PAUSED"
            status_subtitle = "Uncheck pause to continue"

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.prev_time, 1e-6))
        self.prev_time = now

        angle_items = build_angle_items(analyser, angles)

        ui_state = {
            "exercise_names": EXERCISE_NAMES,
            "selected_index": EXERCISE_NAMES.index(self.current_exercise),
            "exercise": self.current_exercise,
            "phase": phase,
            "score": score,
            "reps": analyser.rep_count,
            "rep_goal": 20,
            "fps": self.fps,
            "feedback": feedback if feedback else (
                ["Good posture detected."] if pose_detected
                else ["Position yourself so your full body is visible."]
            ),
            "tips": get_exercise_tips(self.current_exercise),
            "angles": angle_items,
            "status_title": status_title,
            "status_subtitle": status_subtitle,
        }

        dashboard = render_dashboard(camera_view, ui_state, size=(1600, 900))
        return av.VideoFrame.from_ndarray(dashboard, format="bgr24")


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-title">GYM AI</div>
            <div class="brand-sub">Pose Analyzer Control Panel</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workout Controls")

    selected_exercise = st.selectbox(
        "Choose Exercise",
        EXERCISE_NAMES,
        index=0,
    )

    pause_workout = st.checkbox("Pause workout", value=False)
    reset_btn = st.button("Reset Current Exercise")

    st.markdown("### Exercise Tips")
    for tip in get_exercise_tips(selected_exercise):
        st.markdown(f"<div class='tip-item'>• {tip}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Use good lighting, keep your full body visible, and stand far enough from the camera.")


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-left">
            <div class="hero-kicker">Real-Time Browser Workout Analyzer</div>
            <h1>Gym AI Pose Analyzer</h1>
            <p>
                Professional-looking browser dashboard for exercise tracking,
                rep counting, posture feedback, and live joint-angle analysis.
            </p>
        </div>
        <div class="hero-badges">
            <div class="hero-badge">Live Webcam</div>
            <div class="hero-badge">MediaPipe Pose</div>
            <div class="hero-badge">Dark Pro UI</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------
st.markdown("<div class='section-title'>Live Analyzer</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='stream-note'>Click <b>START</b> inside the video panel to begin the live camera session.</div>",
    unsafe_allow_html=True,
)

ctx = webrtc_streamer(
    key="gym-analyzer-pro",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GymProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

if ctx.video_processor:
    ctx.video_processor.set_exercise(selected_exercise)
    ctx.video_processor.paused = pause_workout

    if reset_btn:
        ctx.video_processor.reset_current()

st.markdown("<div class='section-title'>Platform Features</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="info-card">
            <h3>Live Pose Tracking</h3>
            <p>
                Detect body landmarks in real time and visualize posture directly
                inside a styled workout dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="info-card">
            <h3>Exercise Intelligence</h3>
            <p>
                Track phases, rep count, joint angles, and exercise-specific
                feedback for multiple workouts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="info-card">
            <h3>Professional Browser UI</h3>
            <p>
                Clean dark layout, sidebar controls, full-width live analyzer,
                and a polished look closer to a real web product.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )