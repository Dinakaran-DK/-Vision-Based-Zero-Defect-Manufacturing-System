import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# -------------------------------------------------------------------
# PAGE CONFIG & INDUSTRIAL LIGHT THEME
# -------------------------------------------------------------------
st.set_page_config(
    page_title="SMART INSPECT | Industrial Light AI Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Industrial Light Theme Setup */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .top-header {
        background-color: #ffffff;
        padding: 15px 25px;
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        color: #0f172a;
    }
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.03);
    }
    .metric-title {
        color: #64748b;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .metric-num {
        color: #0f172a;
        font-size: 32px;
        font-weight: 800;
        margin: 5px 0;
    }
    .badge-fail {
        background-color: #dc2626;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
    .badge-pass {
        background-color: #16a34a;
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 16px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization for dynamic history
if "history_log" not in st.session_state:
    st.session_state.history_log = []

# -------------------------------------------------------------------
# COMPREHENSIVE MULTI-SCALE PCB INSPECTION ENGINE
# -------------------------------------------------------------------
def analyze_surface_topology(image, sensitivity=50):
    """
    Detects small cracks, surface scratches, and large burnt holes/voids.
    Returns annotated frame, defect metadata, total area, severity, and grayscale map.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    annotated = image.copy()
    
    defects = []
    total_defect_area = 0.0
    occupied_mask = np.zeros((h, w), dtype=np.uint8)

    # --- PHASE 1: LARGE BURNT HOLE & VOID EXTRACTION (HSV + OTSU) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Isolate dark charred/burnt regions
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 75])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Clean morphological noise
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cleaned_dark = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_large)
    
    contours_large, _ = cv2.findContours(cleaned_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_large:
        area = cv2.contourArea(cnt)
        if area > 450:  # Large burnt hole or blowout region
            x, y, bw, bh = cv2.boundingRect(cnt)
            total_defect_area += area
            
            cv2.drawContours(occupied_mask, [cnt], -1, 255, -1)
            
            defects.append({
                "id": f"DEF-{len(defects)+1:02d}",
                "type": "Burnt Hole / Major Void",
                "area": int(area),
                "box": (x, y, bw, bh)
            })
            
            # Thick Red Box for Major Defects
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 220), 3)
            
            # Background Box for Readable Labeling
            label = f"BURNT HOLE [{int(area)}px]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x, max(0, y - 25)), (x + tw + 10, max(25, y)), (0, 0, 220), -1)
            cv2.putText(annotated, label, (x + 5, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # --- PHASE 2: MICRO-CRACKS & SCRATCHES (EXCLUDING HOLE AREA) ---
    k_size = max(3, int(sensitivity / 10) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blur = cv2.GaussianBlur(tophat, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    # Mask out regions already covered by large burnt holes
    thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(occupied_mask))
    
    contours_small, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_small:
        area = cv2.contourArea(cnt)
        if 80 < area < (h * w * 0.05):  # Filter out trivial noise pads
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / bh if bh > 0 else 0
            
            defect_type = "Surface Scratch" if (aspect_ratio > 3.0 or aspect_ratio < 0.33) else "Micro-Crack"
            
            total_defect_area += area
            defects.append({
                "id": f"DEF-{len(defects)+1:02d}",
                "type": defect_type,
                "area": int(area),
                "box": (x, y, bw, bh)
            })
            
            # Draw Yellow Box for minor anomalies
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 180, 255), 2)
            
            # Readable Label
            lbl = f"{defect_type}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (x, max(0, y - 18)), (x + tw + 6, max(18, y)), (0, 180, 255), -1)
            cv2.putText(annotated, lbl, (x + 3, max(14, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Compute Overall Severity Score (0 to 100 Scale)
    severity_score = min(100.0, (total_defect_area / (h * w * 0.005)) * 100)
    
    return annotated, defects, total_defect_area, severity_score, gray

# -------------------------------------------------------------------
# 3D TOPOGRAPHY GENERATOR
# -------------------------------------------------------------------
def generate_3d_surface_plot(gray_img):
    small_img = cv2.resize(gray_img, (80, 80))
    fig = go.Figure(data=[go.Surface(z=small_img, colorscale='Viridis')])
    fig.update_layout(
        title="3D Component Depth Mesh",
        autosize=True,
        height=400,
        margin=dict(l=10, r=10, b=10, t=40),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title="Surface Depth")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a")
    )
    return fig

# -------------------------------------------------------------------
# SIDEBAR CONTROL PANEL
# -------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ SMART INSPECT")
    st.caption("Light-Theme Quality Engine")
    st.markdown("---")
    
    page = st.radio("Navigation View", ["Visual Workspace", "3D Surface Analytics", "Inspection Logs"])
    
    st.markdown("---")
    st.subheader("⚙️ Control Settings")
    plant_line = st.selectbox("Line ID", ["Line 04 - Stamping & Milling", "Line 01 - SMT Assembly"])
    operator_id = st.text_input("Operator Name", "Op. Dinakar")
    reflow_temp = st.slider("Reflow Temperature (°C)", 40.0, 110.0, 84.0)
    sensitivity = st.slider("Crack Sensitivity", 10, 100, 50)

# -------------------------------------------------------------------
# TOP STATUS BANNER
# -------------------------------------------------------------------
st.markdown(f"""
<div class="top-header">
    <div><strong style="color:#2563eb;">📍 Plant Location:</strong> Detroit-01 &nbsp;|&nbsp; <span style="color:#64748b;">{plant_line}</span></div>
    <div>🕒 {datetime.now().strftime('%H:%M:%S')} &nbsp;|&nbsp; 👤 <strong>{operator_id}</strong></div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# PAGE 1: VISUAL INSPECTION WORKSPACE
# -------------------------------------------------------------------
if page == "Visual Workspace":
    st.title("Visual Inspection Workspace")
    st.caption("Multi-scale vision pipeline: Detects micro-cracks, scratches, and large burnt voids.")
    
    input_mode = st.radio("Select Source", ["Static Image Upload", "Live Camera Feed"], horizontal=True)
    raw_frame = None

    if input_mode == "Static Image Upload":
        uploaded_file = st.file_uploader("Upload Manufactured Part Image", type=["jpg", "png", "jpeg", "webp"])
        if uploaded_file:
            bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_frame = cv2.imdecode(bytes_data, 1)
    else:
        run_cam = st.checkbox("Enable Live Camera Feed", value=True)
        if run_cam:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                raw_frame = frame
            cap.release()

    if raw_frame is not None:
        annotated_img, defects, total_area, severity, gray_img = analyze_surface_topology(raw_frame, sensitivity)
        
        col1, col2 = st.columns([1.2, 0.8])
        
        with col1:
            st.image(annotated_img, channels="BGR", use_container_width=True, caption="Multi-Scale Defect Detection Overlay")
            
        with col2:
            st.markdown("### 📊 Diagnostic Results")
            has_major_hole = any(d["type"] == "Burnt Hole / Major Void" for d in defects)
            is_reject = len(defects) > 0 or reflow_temp > 78.0
            
            if is_reject:
                st.markdown("<span class='badge-fail'>❌ STATUS: REJECT (Defect Detected)</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge-pass'>✅ STATUS: PASS (Defect Free)</span>", unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>Total Defects</div><div class='metric-num'>{len(defects)}</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-box'><div class='metric-title'>Severity Score</div><div class='metric-num'>{severity:.1f}</div></div>", unsafe_allow_html=True)
                
            if len(defects) > 0:
                st.markdown("#### 📑 Itemized Defects")
                df_defects = pd.DataFrame(defects)[["id", "type", "area"]]
                df_defects.columns = ["ID", "Classification", "Area (px)"]
                st.dataframe(df_defects, use_container_width=True)
                
                if st.button("Save Log Record"):
                    st.session_state.history_log.append({
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Defects Count": len(defects),
                        "Primary Defect": defects[0]["type"],
                        "Severity": f"{severity:.1f}",
                        "Status": "REJECT" if is_reject else "PASS"
                    })
                    st.success("Log record saved.")

            if has_major_hole:
                st.error("🚨 **Critical Warning:** Burnt hole / severe electrical blowout detected on surface.")
            elif reflow_temp > 78.0:
                st.warning("⚠️ **Thermal Caution:** Reflow temperature elevated. Reduce zone heater temperature by 5°C.")

# -------------------------------------------------------------------
# PAGE 2: 3D TOPOGRAPHY
# -------------------------------------------------------------------
elif page == "3D Surface Analytics":
    st.title("3D Topographical Surface View")
    st.caption("3D mesh depth visualization for structural damage analysis.")
    
    uploaded_file = st.file_uploader("Upload Image for 3D Surface Processing", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)
        _, _, _, _, gray_img = analyze_surface_topology(img, sensitivity)
        
        fig = generate_3d_surface_plot(gray_img)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# PAGE 3: LOGS
# -------------------------------------------------------------------
else:
    st.title("Inspection Logs")
    if len(st.session_state.history_log) > 0:
        st.dataframe(pd.DataFrame(st.session_state.history_log), use_container_width=True)
    else:
        st.info("No records logged yet. Complete an inspection in the Visual Workspace and click 'Save Log Record'.")
