import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# -------------------------------------------------------------------
# PAGE CONFIG & INDUSTRIAL DARK THEME
# -------------------------------------------------------------------
st.set_page_config(
    page_title="SMART INSPECT | Factory AI Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    .top-header {
        background-color: #161b22;
        padding: 14px 24px;
        border-radius: 8px;
        border: 1px solid #30363d;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        color: #c9d1d9;
    }
    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #8b949e;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-num {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
        margin: 6px 0;
    }
    .badge-fail {
        background-color: #da3633;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
        border: 1px solid #f85149;
    }
    .badge-pass {
        background-color: #238636;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 15px;
        display: inline-block;
        border: 1px solid #2ea043;
    }
</style>
""", unsafe_allow_html=True)

if "history_log" not in st.session_state:
    st.session_state.history_log = []

# WebRTC ICE Server Configuration for Cloud Tunneling
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -------------------------------------------------------------------
# DETECTION ENGINE & HUMAN FILTER
# -------------------------------------------------------------------
def is_human_present(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / (image.shape[0] * image.shape[1])
    return skin_ratio > 0.15

def analyze_surface_topology(image, sensitivity=50):
    if is_human_present(image):
        annotated = image.copy()
        cv2.putText(annotated, "HUMAN DETECTED - INSPECTION PAUSED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return annotated, [], 0.0, 0.0, gray

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    annotated = image.copy()
    
    defects = []
    total_defect_area = 0.0
    occupied_mask = np.zeros((h, w), dtype=np.uint8)

    # Burnt Holes Filter
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 30])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned_dark = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_large)
    contours_large, _ = cv2.findContours(cleaned_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_large:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        extent = float(area) / (bw * bh) if (bw * bh) > 0 else 0
        
        if area > 1200 and extent < 0.80:
            total_defect_area += area
            cv2.drawContours(occupied_mask, [cnt], -1, 255, -1)
            defects.append({"id": f"DEF-{len(defects)+1:02d}", "type": "Burnt Hole / Major Void", "area": int(area), "box": (x, y, bw, bh)})
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 220), 3)

    # Micro-cracks Filter
    k_size = max(3, int(sensitivity / 10) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blur = cv2.GaussianBlur(tophat, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(occupied_mask))
    
    contours_small, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_small:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        if bh < 12 or bw < 6 or (bw > (w * 0.4)) or (bh > (h * 0.4)):
            continue
            
        if 150 < area < (h * w * 0.03):
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            defect_type = "Surface Scratch" if (aspect_ratio > 3.5 or aspect_ratio < 0.28) else "Micro-Crack"
            total_defect_area += area
            defects.append({"id": f"DEF-{len(defects)+1:02d}", "type": defect_type, "area": int(area), "box": (x, y, bw, bh)})
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 180, 255), 2)

    severity_score = min(100.0, (total_defect_area / (h * w * 0.005)) * 100)
    return annotated, defects, total_defect_area, severity_score, gray

# -------------------------------------------------------------------
# WEBRTC VIDEO PROCESSOR CLASS
# -------------------------------------------------------------------
class PCBVideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.sensitivity = 50

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        annotated_img, _, _, _, _ = analyze_surface_topology(img, sensitivity=self.sensitivity)
        return annotated_img

# -------------------------------------------------------------------
# STREAMLIT UI LAYOUT
# -------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ SMART INSPECT")
    page = st.radio("Navigation View", ["Visual Workspace", "3D Surface Analytics", "Inspection Logs"])
    st.markdown("---")
    sensitivity = st.slider("Crack Sensitivity", 10, 100, 50)

st.markdown(f"""
<div class="top-header">
    <div><strong style="color:#58a6ff;">📍 Cloud Engine Active</strong></div>
    <div>🕒 {datetime.now().strftime('%H:%M:%S')}</div>
</div>
""", unsafe_allow_html=True)

if page == "Visual Workspace":
    st.title("Visual Inspection Workspace")
    input_mode = st.radio("Select Source", ["Live WebRTC Stream", "Static Image Upload"], horizontal=True)

    if input_mode == "Live WebRTC Stream":
        webrtc_ctx = webrtc_streamer(
            key="pcb-inspection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=PCBVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.sensitivity = sensitivity

    else:
        uploaded_file = st.file_uploader("Upload Component Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_frame = cv2.imdecode(bytes_data, 1)
            annotated_img, defects, total_area, severity, gray_img = analyze_surface_topology(raw_frame, sensitivity)
            st.image(annotated_img, channels="BGR", use_container_width=True)
            st.write(f"**Total Defects Detected:** {len(defects)}")

elif page == "Inspection Logs":
    st.title("Inspection Logs")
    st.info("Log dynamic storage ready.")
