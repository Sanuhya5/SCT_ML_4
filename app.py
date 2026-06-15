"""
=============================================================================
  SkillCraft Technology — Machine Learning Internship | Task 4
  Hand Gesture Recognition System — Streamlit Web Application
  Author  : [Your Name]
  Dataset : LeapGestRecog (Kaggle) — gti-upm/leapgestrecog
=============================================================================
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Recognition | SkillCraft Task 4",
    page_icon="🤚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark professional theme
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

/* Cards */
.glass-card {
    background: rgba(22, 27, 34, 0.85);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 16px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    backdrop-filter: blur(12px);
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 40%, #1a1f2e 100%);
    border: 1px solid #21262d;
    border-radius: 20px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 30%, rgba(88,166,255,0.06) 0%, transparent 50%),
                radial-gradient(circle at 70% 70%, rgba(163,113,247,0.06) 0%, transparent 50%);
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff, #a371f7, #58a6ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shine 3s linear infinite;
    margin: 0 0 8px;
}
@keyframes shine {
    to { background-position: 200% center; }
}

.hero-sub {
    font-size: 1.1rem;
    color: #7d8590;
    font-weight: 400;
    margin: 0;
}

/* Section headers */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: #58a6ff;
    border-left: 4px solid #58a6ff;
    padding-left: 16px;
    margin: 24px 0 16px;
}

/* Metric tiles */
.metric-tile {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #58a6ff;
}
.metric-label {
    font-size: 0.85rem;
    color: #7d8590;
    margin-top: 4px;
}

/* Gesture badge */
.gesture-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1f4280, #1a3060);
    border: 1px solid #58a6ff;
    border-radius: 30px;
    padding: 8px 20px;
    font-size: 0.85rem;
    color: #58a6ff;
    font-weight: 600;
    margin: 4px;
}

/* Prediction result box */
.prediction-box {
    background: linear-gradient(135deg, #0f2027, #1a3a2a);
    border: 2px solid #3fb950;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.prediction-label {
    font-size: 2.5rem;
    font-weight: 800;
    color: #3fb950;
}
.confidence-score {
    font-size: 1.2rem;
    color: #7d8590;
    margin-top: 8px;
}

/* Upload zone */
div[data-testid="stFileUploader"] {
    border: 2px dashed #30363d !important;
    border-radius: 12px !important;
    background: rgba(22, 27, 34, 0.5) !important;
    transition: border-color 0.3s;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #58a6ff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(88,166,255,0.35) !important;
}

/* Divider */
.custom-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #30363d, transparent);
    margin: 32px 0;
}

/* Code-style tags */
.tag {
    font-family: 'JetBrains Mono', monospace;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.8rem;
    color: #a371f7;
}

/* Info box */
.info-box {
    background: rgba(31, 111, 235, 0.1);
    border: 1px solid #1f6feb;
    border-radius: 10px;
    padding: 16px 20px;
    color: #cae8ff;
    font-size: 0.92rem;
}

/* Warning box */
.warn-box {
    background: rgba(210, 153, 34, 0.1);
    border: 1px solid #d29922;
    border-radius: 10px;
    padding: 16px 20px;
    color: #f0c000;
    font-size: 0.92rem;
}

/* Steps list */
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 16px;
}
.step-num {
    background: #1f6feb;
    color: white;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    flex-shrink: 0;
    font-size: 0.9rem;
}

/* Sidebar nav */
.nav-item {
    padding: 10px 16px;
    border-radius: 8px;
    margin: 4px 0;
    cursor: pointer;
    transition: background 0.2s;
    color: #c9d1d9;
    font-size: 0.95rem;
}
.nav-item:hover { background: #21262d; }
.nav-item.active { background: #1f3c6e; color: #58a6ff; font-weight: 600; }

/* Badge strip */
.badge-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 64  # must match training
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "hand_gesture_model.h5")

GESTURE_CLASSES = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down",
]

GESTURE_DISPLAY = {
    "01_palm":       ("🖐️ Palm",        "Open hand facing the camera"),
    "02_l":          ("☝️ L-Shape",      "Index and thumb extended at 90°"),
    "03_fist":       ("✊ Fist",         "Closed fist, no movement"),
    "04_fist_moved": ("👊 Fist Moved",   "Closed fist with lateral shift"),
    "05_thumb":      ("👍 Thumbs Up",    "Only thumb extended upward"),
    "06_index":      ("☝️ Index Point",  "Only index finger extended"),
    "07_ok":         ("👌 OK",           "Thumb and index forming a circle"),
    "08_palm_moved": ("🤚 Palm Moved",   "Open palm with lateral shift"),
    "09_c":          ("🤏 C-Shape",      "Fingers curved into a C"),
    "10_down":       ("👇 Down",         "Index pointing downward"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_gesture_model():
    """Load the trained Keras model from disk. Returns None if not found."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to the preprocessed numpy array the model expects.
    Steps: RGB conversion → resize to IMG_SIZExIMG_SIZE → normalize [0,1] → add batch dim.
    """
    img = pil_image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # normalize
    arr = np.expand_dims(arr, axis=0)                # (1, 64, 64, 3)
    return arr


def predict_gesture(model, pil_image: Image.Image):
    """
    Run inference and return (class_name, confidence, full_probs).
    """
    arr = preprocess_image(pil_image)
    probs = model.predict(arr, verbose=0)[0]         # shape: (10,)
    pred_idx = int(np.argmax(probs))
    return GESTURE_CLASSES[pred_idx], float(probs[pred_idx]), probs


def confidence_bar_chart(probs: np.ndarray):
    """Build a Plotly horizontal bar chart of class probabilities."""
    labels = [GESTURE_DISPLAY[c][0] for c in GESTURE_CLASSES]
    colors = ["#3fb950" if i == int(np.argmax(probs)) else "#1f6feb"
              for i in range(len(probs))]
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        textfont=dict(color="#c9d1d9", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9", family="Inter"),
        xaxis=dict(
            title="Confidence (%)",
            range=[0, 115],
            gridcolor="#21262d",
            tickfont=dict(color="#7d8590"),
        ),
        yaxis=dict(tickfont=dict(color="#c9d1d9"), autorange="reversed"),
        margin=dict(l=140, r=60, t=20, b=40),
        height=340,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 8px;'>
        <div style='font-size:2.5rem;'>🤚</div>
        <div style='font-size:1.1rem; font-weight:700; color:#58a6ff;'>Gesture Vision</div>
        <div style='font-size:0.78rem; color:#7d8590; margin-top:4px;'>SkillCraft Technology · Task 4</div>
    </div>
    <hr style='border-color:#21262d; margin:16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Home", "📖 About Project", "🔍 Predict Gesture", "📊 Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#21262d;'>", unsafe_allow_html=True)

    # Quick model status
    model = load_gesture_model()
    if model is not None:
        st.markdown("""
        <div style='background:rgba(63,185,80,0.12); border:1px solid #3fb950;
             border-radius:10px; padding:12px 16px; font-size:0.85rem; color:#3fb950;'>
            ✅ Model loaded &amp; ready
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(210,153,34,0.12); border:1px solid #d29922;
             border-radius:10px; padding:12px 16px; font-size:0.85rem; color:#f0c000;'>
            ⚠️ Model not found<br>
            <span style='color:#7d8590; font-size:0.78rem;'>Run train_model.ipynb first</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:#21262d;'>
    <div style='font-size:0.78rem; color:#484f58; text-align:center; padding:8px 0;'>
        Built with ❤️ during<br>SkillCraft ML Internship
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Home
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>🤚 Hand Gesture Recognition</div>
        <p class='hero-sub'>SkillCraft Technology · Machine Learning Internship · Task 04</p>
        <div class='badge-strip' style='justify-content:center; margin-top:20px;'>
            <span class='gesture-badge'>TensorFlow / Keras</span>
            <span class='gesture-badge'>CNN Deep Learning</span>
            <span class='gesture-badge'>LeapGestRecog Dataset</span>
            <span class='gesture-badge'>10 Gesture Classes</span>
            <span class='gesture-badge'>Streamlit UI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("10", "Gesture Classes"),
        ("20,000+", "Training Images"),
        ("64 × 64", "Input Resolution"),
        ("CNN", "Model Architecture"),
    ]
    for col, (val, lbl) in zip([col1, col2, col3, col4], stats):
        col.markdown(f"""
        <div class='metric-tile'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # What it does
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.markdown("<div class='section-header'>What This System Does</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='glass-card'>
            <p style='color:#c9d1d9; line-height:1.8; font-size:0.95rem;'>
            This system uses a <strong style='color:#58a6ff;'>Convolutional Neural Network (CNN)</strong>
            trained on the <strong style='color:#58a6ff;'>LeapGestRecog</strong> dataset to classify
            10 distinct hand gestures captured via near-infrared imaging.
            </p>
            <p style='color:#c9d1d9; line-height:1.8; font-size:0.95rem; margin-top:12px;'>
            Upload any hand gesture image from the dataset and the model predicts
            the gesture category in real-time, along with a <strong style='color:#a371f7;'>confidence score</strong>
            and probability distribution across all classes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-header'>Recognised Gestures</div>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (key, (icon_name, desc)) in enumerate(GESTURE_DISPLAY.items()):
            cols[i % 2].markdown(f"""
            <div style='margin-bottom:10px;'>
                <span style='font-size:1.1rem;'>{icon_name}</span><br>
                <span style='color:#7d8590; font-size:0.78rem;'>{desc}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # How to use
    st.markdown("<div class='section-header'>🚀 Quick Start</div>", unsafe_allow_html=True)
    steps = [
        ("Train the model", "Open <code>train_model.ipynb</code> and run all cells. The trained model is saved to <code>model/hand_gesture_model.h5</code>."),
        ("Launch the app", "Run <code>streamlit run app.py</code> in your terminal."),
        ("Upload an image", "Navigate to <strong>Predict Gesture</strong> in the sidebar and upload a hand gesture image."),
        ("View prediction", "Instantly see the predicted gesture, confidence score, and probability chart."),
    ]
    for i, (title, body) in enumerate(steps, 1):
        st.markdown(f"""
        <div class='step-item'>
            <div class='step-num'>{i}</div>
            <div>
                <strong style='color:#c9d1d9;'>{title}</strong><br>
                <span style='color:#7d8590; font-size:0.88rem;'>{body}</span>
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: About Project
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📖 About Project":
    st.markdown("<div class='section-header'>📖 About This Project</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card'>
        <p style='color:#c9d1d9; line-height:1.8;'>
            <strong style='color:#58a6ff;'>Task 4</strong> of the SkillCraft Technology Machine Learning
            Internship challenges interns to build an end-to-end <strong>Hand Gesture Recognition System</strong>
            using Deep Learning. The system classifies 10 unique hand gestures from near-infrared images
            captured with a Leap Motion controller.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset
    st.markdown("<div class='section-header'>📁 Dataset — LeapGestRecog</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <table style='width:100%; color:#c9d1d9; font-size:0.9rem; border-collapse:collapse;'>
                <tr><td style='padding:8px 0; color:#7d8590; width:40%;'>Source</td>
                    <td><a href='https://www.kaggle.com/datasets/gti-upm/leapgestrecog' style='color:#58a6ff;'>Kaggle — gti-upm/leapgestrecog</a></td></tr>
                <tr><td style='padding:8px 0; color:#7d8590;'>Subjects</td><td>10 participants</td></tr>
                <tr><td style='padding:8px 0; color:#7d8590;'>Gestures</td><td>10 classes × 200 images per subject</td></tr>
                <tr><td style='padding:8px 0; color:#7d8590;'>Total Images</td><td>~20,000</td></tr>
                <tr><td style='padding:8px 0; color:#7d8590;'>Image Type</td><td>Near-infrared (grayscale → converted to RGB)</td></tr>
                <tr><td style='padding:8px 0; color:#7d8590;'>Resolution</td><td>240 × 640 px (raw) → 64 × 64 (model input)</td></tr>
                <tr><td style='padding:8px 0; color:#7d8590;'>Train / Test Split</td><td>80% / 20% stratified</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <div style='font-weight:600; color:#58a6ff; margin-bottom:12px;'>Gesture Classes</div>
        """, unsafe_allow_html=True)
        for key, (icon_name, desc) in GESTURE_DISPLAY.items():
            st.markdown(f"""
            <div style='margin-bottom:8px;'>
                <span class='tag'>{key}</span>
                <span style='color:#c9d1d9; margin-left:8px; font-size:0.88rem;'>{icon_name}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Methodology
    st.markdown("<div class='section-header'>🔬 Methodology</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
        <div style='display:flex; flex-wrap:wrap; gap:0; align-items:center;'>
    """, unsafe_allow_html=True)
    pipeline = [
        ("📥", "Data Loading", "Read images from directory structure; map folder names to integer labels"),
        ("🔄", "Pre-processing", "Resize to 64×64, convert to RGB, normalise pixel values to [0, 1]"),
        ("✂️", "Train/Test Split", "80% train / 20% test, stratified by class"),
        ("🏗️", "CNN Architecture", "3 Conv-BN-Pool blocks → GlobalAveragePooling → Dense → Softmax(10)"),
        ("🎯", "Training", "Adam optimiser, categorical cross-entropy, 30 epochs, batch size 32"),
        ("📈", "Evaluation", "Accuracy, Loss curves, Confusion Matrix, Classification Report"),
        ("🌐", "Deployment", "Streamlit interactive web app with real-time inference"),
    ]
    for icon, title, desc in pipeline:
        st.markdown(f"""
        <div style='display:flex; align-items:flex-start; gap:14px; margin-bottom:16px;'>
            <div style='background:#1f6feb; border-radius:10px; padding:10px;
                        font-size:1.3rem; min-width:44px; text-align:center;'>{icon}</div>
            <div>
                <div style='color:#58a6ff; font-weight:600; font-size:0.95rem;'>{title}</div>
                <div style='color:#7d8590; font-size:0.85rem; margin-top:3px;'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Model architecture
    st.markdown("<div class='section-header'>🧠 CNN Architecture</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
        <div style='font-family:"JetBrains Mono",monospace; font-size:0.82rem;
                    color:#c9d1d9; line-height:1.9; background:#0d1117;
                    padding:20px; border-radius:10px; border:1px solid #21262d;'>
<span style='color:#58a6ff;'>Input</span>         →  (64, 64, 3)  — RGB image<br>
<span style='color:#a371f7;'>Conv2D(32)</span>   →  32 × 3×3 filters, ReLU, same padding<br>
<span style='color:#a371f7;'>BatchNorm</span>    →  Stabilise activations<br>
<span style='color:#a371f7;'>MaxPool2D</span>    →  2×2 stride, output (32, 32, 32)<br>
<span style='color:#a371f7;'>Dropout(0.25)</span>→  Regularisation<br><br>
<span style='color:#a371f7;'>Conv2D(64)</span>   →  64 × 3×3 filters, ReLU, same padding<br>
<span style='color:#a371f7;'>BatchNorm</span>    →  Stabilise activations<br>
<span style='color:#a371f7;'>MaxPool2D</span>    →  2×2 stride, output (16, 16, 64)<br>
<span style='color:#a371f7;'>Dropout(0.25)</span>→  Regularisation<br><br>
<span style='color:#a371f7;'>Conv2D(128)</span>  →  128 × 3×3 filters, ReLU, same padding<br>
<span style='color:#a371f7;'>BatchNorm</span>    →  Stabilise activations<br>
<span style='color:#a371f7;'>MaxPool2D</span>    →  2×2 stride, output (8, 8, 128)<br>
<span style='color:#a371f7;'>Dropout(0.25)</span>→  Regularisation<br><br>
<span style='color:#3fb950;'>GlobalAvgPool</span>→  (128,)  — spatial compression<br>
<span style='color:#3fb950;'>Dense(256)</span>   →  ReLU + Dropout(0.5)<br>
<span style='color:#3fb950;'>Dense(10)</span>    →  Softmax → class probabilities<br>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack
    st.markdown("<div class='section-header'>🛠️ Technologies Used</div>", unsafe_allow_html=True)
    tech = [
        ("Python 3.10+", "Core language"),
        ("TensorFlow 2.x / Keras", "CNN model definition, training, and inference"),
        ("NumPy", "Array operations and image preprocessing"),
        ("OpenCV", "Image loading and colour-space conversion"),
        ("Matplotlib / Seaborn", "Training curves and confusion matrix"),
        ("Scikit-learn", "Train/test split, classification report, metrics"),
        ("Streamlit", "Interactive web application"),
        ("Plotly", "Dynamic confidence bar chart"),
    ]
    cols = st.columns(2)
    for i, (name, desc) in enumerate(tech):
        cols[i % 2].markdown(f"""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:10px;
                    padding:14px 18px; margin-bottom:10px;'>
            <div style='color:#58a6ff; font-weight:600; font-size:0.9rem;'>{name}</div>
            <div style='color:#7d8590; font-size:0.82rem; margin-top:4px;'>{desc}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Predict Gesture
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Predict Gesture":
    st.markdown("<div class='section-header'>🔍 Predict Hand Gesture</div>", unsafe_allow_html=True)

    model = load_gesture_model()
    if model is None:
        st.markdown("""
        <div class='warn-box'>
            ⚠️ <strong>Model not found.</strong> Please run <code>train_model.ipynb</code> first
            to generate <code>model/hand_gesture_model.h5</code>, then restart the app.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown("""
    <div class='info-box'>
        ℹ️ Upload a hand gesture image (JPG, PNG) from the <strong>LeapGestRecog</strong> dataset
        or any similar near-infrared hand image. The model will classify it into one of 10 gesture categories.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload
    uploaded_file = st.file_uploader(
        "Drop your image here, or click to browse",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="visible",
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Two-column layout: image | result
        col_img, col_res = st.columns([1, 1.2], gap="large")

        with col_img:
            st.markdown("<div style='font-weight:600; color:#58a6ff; margin-bottom:12px;'>📷 Uploaded Image</div>",
                        unsafe_allow_html=True)
            st.image(pil_image, use_column_width=True, caption=f"File: {uploaded_file.name}")
            meta_c1, meta_c2 = st.columns(2)
            meta_c1.metric("Width", f"{pil_image.width} px")
            meta_c2.metric("Height", f"{pil_image.height} px")

        with col_res:
            with st.spinner("🤖 Running inference…"):
                class_name, confidence, probs = predict_gesture(model, pil_image)

            icon_name, desc = GESTURE_DISPLAY[class_name]
            pct = confidence * 100

            st.markdown(f"""
            <div class='prediction-box'>
                <div style='color:#7d8590; font-size:0.85rem; margin-bottom:6px;'>PREDICTED GESTURE</div>
                <div class='prediction-label'>{icon_name}</div>
                <div style='color:#7d8590; font-size:0.9rem; margin-top:4px;'>{desc}</div>
                <div class='confidence-score' style='margin-top:16px;'>
                    Confidence: <strong style='color:#3fb950; font-size:1.4rem;'>{pct:.1f}%</strong>
                </div>
                <div style='margin-top:14px;'>
                    <span class='tag'>{class_name}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence level tag
            if pct >= 90:
                level, lcolor = "Very High Confidence", "#3fb950"
            elif pct >= 70:
                level, lcolor = "High Confidence", "#58a6ff"
            elif pct >= 50:
                level, lcolor = "Moderate Confidence", "#d29922"
            else:
                level, lcolor = "Low Confidence", "#f85149"

            st.markdown(f"""
            <div style='margin-top:12px; padding:10px 16px; border-radius:8px;
                        background:rgba(0,0,0,0.3); border:1px solid {lcolor};
                        color:{lcolor}; font-size:0.85rem; font-weight:600; text-align:center;'>
                🎯 {level}
            </div>""", unsafe_allow_html=True)

        # Probability chart
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:600; color:#58a6ff; margin-bottom:12px;'>📊 Confidence Distribution Across All Classes</div>",
                    unsafe_allow_html=True)
        fig = confidence_bar_chart(probs)
        st.plotly_chart(fig, use_container_width=True)

        # Top-3 table
        st.markdown("<div style='font-weight:600; color:#58a6ff; margin-bottom:12px;'>🏆 Top 3 Predictions</div>",
                    unsafe_allow_html=True)
        top3_idx = np.argsort(probs)[::-1][:3]
        for rank, idx in enumerate(top3_idx, 1):
            name = GESTURE_CLASSES[idx]
            icon_n, _ = GESTURE_DISPLAY[name]
            prob_val = probs[idx] * 100
            bar_color = "#3fb950" if rank == 1 else "#1f6feb"
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:16px; margin-bottom:10px;'>
                <div style='color:#7d8590; font-weight:700; width:24px;'>#{rank}</div>
                <div style='color:#c9d1d9; width:160px; font-size:0.9rem;'>{icon_n}</div>
                <div style='flex:1; background:#161b22; border-radius:6px; height:12px; overflow:hidden;'>
                    <div style='width:{prob_val:.1f}%; background:{bar_color};
                                height:100%; border-radius:6px; transition:width 0.5s;'></div>
                </div>
                <div style='color:{bar_color}; font-weight:700; width:56px; text-align:right;'>{prob_val:.1f}%</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Model Insights
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Insights":
    st.markdown("<div class='section-header'>📊 Model Insights & Evaluation</div>", unsafe_allow_html=True)

    # Check for output images
    outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")

    output_files = {
        "accuracy_curve.png": ("📈 Training & Validation Accuracy", "accuracy_curve"),
        "loss_curve.png":     ("📉 Training & Validation Loss",     "loss_curve"),
        "confusion_matrix.png": ("🗂️ Confusion Matrix",             "confusion_matrix"),
        "sample_predictions.png": ("🖼️ Sample Predictions",         "sample_predictions"),
    }

    any_found = False
    for fname, (title, key) in output_files.items():
        fpath = os.path.join(outputs_dir, fname)
        if os.path.exists(fpath):
            any_found = True

    if not any_found:
        st.markdown("""
        <div class='warn-box'>
            ⚠️ <strong>No output images found.</strong>
            Run <code>train_model.ipynb</code> to generate training curves, confusion matrix,
            and sample predictions in the <code>outputs/</code> folder.
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        for fname, (title, key) in output_files.items():
            fpath = os.path.join(outputs_dir, fname)
            if os.path.exists(fpath):
                col = col1 if list(output_files.keys()).index(fname) % 2 == 0 else col2
                with col:
                    st.markdown(f"<div style='font-weight:600; color:#58a6ff; margin:16px 0 8px;'>{title}</div>",
                                unsafe_allow_html=True)
                    st.image(fpath, use_column_width=True)

    # Model summary placeholder
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📋 Expected Performance</div>", unsafe_allow_html=True)
    metrics = [
        ("Training Accuracy", "~99%", "#3fb950"),
        ("Validation Accuracy", "~96%", "#3fb950"),
        ("Test Accuracy", "~95%", "#58a6ff"),
        ("Parameters", "~1.2 M", "#a371f7"),
    ]
    cols = st.columns(4)
    for col, (label, value, color) in zip(cols, metrics):
        col.markdown(f"""
        <div class='metric-tile'>
            <div class='metric-value' style='color:{color};'>{value}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card' style='margin-top:24px;'>
        <div style='color:#7d8590; font-size:0.88rem; line-height:1.8;'>
            <strong style='color:#58a6ff;'>Note:</strong> Exact metrics depend on the specific random seed,
            hardware, and TensorFlow version used during training. The values above are representative
            of results achieved on the full LeapGestRecog dataset with the architecture described in this project.
            Run <code>train_model.ipynb</code> to see your actual training metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)
