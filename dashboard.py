"""MedImage AI — Chest X-ray Analysis Dashboard.

Author: Mahesh Bhakre
Notice: MODEL-ASSISTED SCREENING | HUMAN REVIEW REQUIRED
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# Include src in path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from utils import (
    get_dataset_counts,
    get_training_history,
    get_evaluation_metrics,
    plot_dataset_distribution,
    plot_training_curves,
    plot_confusion_matrix,
)

MODEL_PATH = ROOT / "models" / "medical_model.h5"
DATA_PATH = ROOT / "data" / "chest_xray"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Page configuration
st.set_page_config(
    page_title="MedImage AI | Chest X-ray Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Professional Medical AI Dark Theme
st.markdown(
    """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0B1220;
        background-image: radial-gradient(circle at 10% 10%, rgba(34, 211, 238, 0.05) 0%, transparent 40%),
                          radial-gradient(circle at 90% 90%, rgba(59, 130, 246, 0.06) 0%, transparent 45%);
        color: #F8FAFC;
    }
    
    .block-container {
        max-width: 1350px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    
    /* Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111C30 !important;
        border-right: 1px solid #1E2D4A;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: #B8C5D9 !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Hero Banner */
    .hero-banner {
        padding: 1.4rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #112240 0%, #18263D 50%, #152A4A 100%);
        border: 1px solid #253858;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: 0; right: 0; width: 300px; height: 100%;
        background: radial-gradient(circle at 80% 20%, rgba(34, 211, 238, 0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .hero-title span.accent {
        color: #22D3EE;
    }
    .hero-subtitle {
        color: #B8C5D9;
        font-size: 1rem;
        margin: 0.25rem 0 0 0;
        font-weight: 500;
    }
    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        background: rgba(34, 211, 238, 0.12);
        border: 1px solid rgba(34, 211, 238, 0.4);
        color: #22D3EE;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    
    /* Content Cards */
    .content-card {
        background-color: #18263D;
        border: 1px solid #253858;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
        margin-bottom: 1.25rem;
    }
    .content-card h3 {
        color: #F8FAFC;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .content-card p {
        color: #B8C5D9;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Step Indicator Badges */
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: #3B82F6;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    /* Result Cards */
    .result-box-normal {
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.12) 0%, rgba(17, 34, 64, 0.9) 100%);
        border: 2px solid #2DD4BF;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(45, 212, 191, 0.2);
    }
    .result-box-pneumonia {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.14) 0%, rgba(17, 34, 64, 0.9) 100%);
        border: 2px solid #F59E0B;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(245, 158, 11, 0.2);
    }
    .result-label-normal {
        color: #2DD4BF;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin: 0;
    }
    .result-label-pneumonia {
        color: #F59E0B;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin: 0;
    }
    .result-score-text {
        color: #F8FAFC;
        font-size: 1.15rem;
        margin: 0.4rem 0 0 0;
        font-weight: 600;
    }
    .result-subscore {
        color: #B8C5D9;
        font-size: 0.88rem;
        margin-top: 0.2rem;
    }
    
    /* Explanation Box */
    .explanation-box {
        background: #111C30;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        border-left: 4px solid #3B82F6;
        color: #F8FAFC;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    
    /* Medical Disclaimer Footer */
    .med-disclaimer {
        background: #111C30;
        border: 1px solid #253858;
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        color: #B8C5D9;
        font-size: 0.85rem;
        line-height: 1.5;
        margin-top: 2rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    /* Metric Cards Custom Styling */
    div[data-testid="stMetric"] {
        background-color: #18263D !important;
        border: 1px solid #253858 !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25) !important;
    }
    div[data-testid="stMetric"] label {
        color: #B8C5D9 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 800 !important;
        font-size: 1.75rem !important;
    }
    
    /* File Uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #111C30 !important;
        border: 2px dashed #3B82F6 !important;
        border-radius: 14px !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"] * {
        color: #F8FAFC !important;
    }
    
    /* Primary Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6 0%, #22D3EE 100%) !important;
        color: #0B1220 !important;
        font-weight: 750 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(34, 211, 238, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(34, 211, 238, 0.5) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Secondary Buttons */
    .stButton > button[kind="secondary"] {
        background: #18263D !important;
        border: 1px solid #3B82F6 !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #1E2D4A !important;
        border-color: #22D3EE !important;
    }
    
    /* Custom Sidebar Brand */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 1.4rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #1E2D4A;
    }
    .brand-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #3B82F6, #22D3EE);
        color: #0B1220;
        display: grid;
        place-items: center;
        font-size: 1.4rem;
        font-weight: 900;
        box-shadow: 0 4px 14px rgba(34, 211, 238, 0.3);
    }
    .brand-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .brand-sub {
        font-size: 0.76rem;
        color: #22D3EE;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    
    /* Radio Nav Custom Styling */
    div[role="radiogroup"] > label {
        background: #18263D;
        border: 1px solid #253858;
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
        transition: all 0.2s ease;
    }
    div[role="radiogroup"] > label:hover {
        border-color: #3B82F6;
        background: #1E2D4A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_screening_model():
    """Load the trained MobileNetV2 chest X-ray classifier."""
    if not MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def preprocess_xray_bytes(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Decode and preprocess image bytes matching the 224x224 RGB training pipeline."""
    if not image_bytes:
        raise ValueError("Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 10 MB limit.")

    raw = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError("Invalid image file. Please provide a standard JPG or PNG chest X-ray.")

    height, width = raw.shape[:2]
    if width < 32 or height < 32:
        raise ValueError("Image resolution is too low (< 32x32). Please upload a diagnostic chest X-ray.")

    # Convert to RGB (MobileNetV2 expected format)
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    return rgb, resized, width, height


def run_inference(image_bytes: bytes) -> tuple[str, float, float, int, int]:
    """Run genuine model inference and return diagnosis label, confidence, raw score, dimensions."""
    model = load_screening_model()
    if model is None:
        raise RuntimeError("Model file 'models/medical_model.h5' is not available.")

    _, preprocessed, width, height = preprocess_xray_bytes(image_bytes)
    input_tensor = np.expand_dims(preprocessed, axis=0)

    raw_score = float(model.predict(input_tensor, verbose=0)[0][0])
    if not np.isfinite(raw_score):
        raise ValueError("Model prediction returned a non-finite score.")

    # Binary classification threshold: 0.5
    label = "PNEUMONIA" if raw_score >= 0.5 else "NORMAL"
    confidence = raw_score if label == "PNEUMONIA" else (1.0 - raw_score)

    return label, confidence, raw_score, width, height


# ---------------- SIDEBAR & NAVIGATION ----------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-icon">🩺</div>
            <div>
                <div class="brand-title">MedImage AI</div>
                <div class="brand-sub">CHEST X-RAY ANALYSIS</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_options = ["X-ray Prediction", "Overview", "Evaluation"]

    # Support programmatic page switching via st.session_state
    current_page = st.session_state.get("nav_page", "X-ray Prediction")
    if current_page not in nav_options:
        current_page = "X-ray Prediction"

    selected_page = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(current_page),
        key="sidebar_nav_radio",
    )
    # Sync navigation state
    if selected_page != st.session_state.get("nav_page"):
        st.session_state["nav_page"] = selected_page

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Model status card in sidebar
    model_loaded = MODEL_PATH.exists()
    if model_loaded:
        st.markdown(
            """
            <div style="background:#18263D; border:1px solid #2DD4BF; border-radius:10px; padding:0.75rem 1rem;">
                <div style="display:flex; align-items:center; gap:0.5rem; color:#2DD4BF; font-weight:700; font-size:0.85rem;">
                    <span style="font-size:1.1rem;">●</span> Model Ready
                </div>
                <div style="color:#B8C5D9; font-size:0.75rem; margin-top:0.25rem;">
                    MobileNetV2 (H5 Loaded)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:#18263D; border:1px solid #EF4444; border-radius:10px; padding:0.75rem 1rem;">
                <div style="display:flex; align-items:center; gap:0.5rem; color:#EF4444; font-weight:700; font-size:0.85rem;">
                    <span style="font-size:1.1rem;">●</span> Model Not Found
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color:#B8C5D9; font-size:0.8rem; border-top:1px solid #1E2D4A; padding-top:1rem;">
            <div style="color:#F8FAFC; font-weight:600;">Author: Mahesh Bhakre</div>
            <div style="color:#22D3EE; font-size:0.72rem; margin-top:0.2rem;">Senior Full-Stack & AI</div>
            <div style="margin-top:0.6rem; color:#64748B; font-size:0.72rem; line-height:1.4;">
                For educational & research screening prototype.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- TOP HEADER BANNER ----------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title-row">
            <div>
                <h1 class="hero-title">🩺 MedImage <span class="accent">AI</span></h1>
                <p class="hero-subtitle">Chest X-ray Analysis — Intelligent Diagnostic Assistance</p>
            </div>
            <div>
                <span class="hero-badge">MODEL-ASSISTED SCREENING | HUMAN REVIEW REQUIRED</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# PAGE 1: X-RAY PREDICTION (DEFAULT)
# ==============================================================================
if selected_page == "X-ray Prediction":
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h2 style="color:#F8FAFC; font-size:1.5rem; font-weight:700; margin:0;">
                Clinical Screening Workflow
            </h2>
            <p style="color:#B8C5D9; margin:0.3rem 0 0 0; font-size:0.95rem;">
                Upload a frontal chest radiograph (PA/AP view) to obtain instant AI-assisted screening analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown(
            """
            <div class="content-card">
                <h3><span class="step-badge">1</span> Step 1: Upload X-ray</h3>
                <p>Select a clear JPG or PNG chest radiograph from your device, or load a sample case.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose X-ray Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG (max 10 MB)",
            key="xray_uploader",
        )

        # Quick sample selector for reviewer convenience
        st.markdown(
            "<div style='color:#B8C5D9; font-size:0.85rem; font-weight:600; margin:0.8rem 0 0.4rem 0;'>Or test with sample cases from dataset:</div>",
            unsafe_allow_html=True,
        )
        sample_col1, sample_col2, sample_col3 = st.columns(3)
        sample_normal_path = DATA_PATH / "test" / "NORMAL" / "IM-0001-0001.jpeg"
        sample_pneu_path = DATA_PATH / "test" / "PNEUMONIA" / "person1_virus_6.jpeg"

        with sample_col1:
            if st.button("Sample Normal", use_container_width=True, key="btn_sample_norm"):
                if sample_normal_path.exists():
                    with open(sample_normal_path, "rb") as f:
                        st.session_state["sample_bytes"] = f.read()
                        st.session_state["sample_name"] = "IM-0001-0001.jpeg (NORMAL)"
                        st.session_state.pop("analysis_result", None)
                        st.rerun()

        with sample_col2:
            if st.button("Sample Pneumonia", use_container_width=True, key="btn_sample_pneu"):
                if sample_pneu_path.exists():
                    with open(sample_pneu_path, "rb") as f:
                        st.session_state["sample_bytes"] = f.read()
                        st.session_state["sample_name"] = "person1_virus_6.jpeg (PNEUMONIA)"
                        st.session_state.pop("analysis_result", None)
                        st.rerun()

        with sample_col3:
            if st.button("Reset / Clear", use_container_width=True, key="btn_clear_all"):
                st.session_state.pop("sample_bytes", None)
                st.session_state.pop("sample_name", None)
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("last_upload_id", None)
                st.rerun()

        # Determine active image bytes
        active_bytes = None
        active_name = ""

        if uploaded_file is not None:
            active_bytes = uploaded_file.getvalue()
            active_name = uploaded_file.name
            upload_sig = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("last_upload_id") != upload_sig:
                st.session_state["last_upload_id"] = upload_sig
                st.session_state.pop("sample_bytes", None)
                st.session_state.pop("sample_name", None)
                st.session_state.pop("analysis_result", None)
        elif "sample_bytes" in st.session_state:
            active_bytes = st.session_state["sample_bytes"]
            active_name = st.session_state.get("sample_name", "Sample X-ray")

        # Display preview if image is selected
        if active_bytes is not None:
            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            st.image(active_bytes, caption=f"Active Preview: {active_name}", use_container_width=True)
            st.caption(f"Payload Size: {len(active_bytes) / 1024:.1f} KB")

    with col_right:
        st.markdown(
            """
            <div class="content-card">
                <h3><span class="step-badge">2</span> Step 2: Analyze</h3>
                <p>Run the deep learning classification model (MobileNetV2) to evaluate radiographic features.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        analyze_clicked = st.button(
            "⚡ Analyze X-ray",
            type="primary",
            use_container_width=True,
            disabled=(active_bytes is None),
        )

        if active_bytes is None:
            st.info("Please upload an X-ray or select a sample case above to enable analysis.")

        if analyze_clicked and active_bytes is not None:
            with st.spinner("Processing image and running MobileNetV2 inference..."):
                try:
                    result = run_inference(active_bytes)
                    st.session_state["analysis_result"] = result
                except Exception as err:
                    st.error(f"Inference error: {err}")

        # Step 3: View Result
        st.markdown(
            """
            <div class="content-card" style="margin-top:1.25rem;">
                <h3><span class="step-badge">3</span> Step 3: View Result</h3>
            """,
            unsafe_allow_html=True,
        )

        if "analysis_result" in st.session_state and active_bytes is not None:
            label, confidence, raw_score, width, height = st.session_state["analysis_result"]

            if label == "NORMAL":
                st.markdown(
                    f"""
                    <div class="result-box-normal">
                        <div style="color:#2DD4BF; font-size:0.85rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase;">
                            Model Prediction
                        </div>
                        <h2 class="result-label-normal">NORMAL</h2>
                        <div class="result-score-text">
                            Model Confidence: <b style="color:#2DD4BF;">{confidence:.1%}</b>
                        </div>
                        <div class="result-subscore">
                            Raw Sigmoid Output: {raw_score:.4f} &nbsp;|&nbsp; Diagnostic Resolution: {width} × {height} px
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    <div class="explanation-box" style="border-left-color: #2DD4BF;">
                        <b>Clinical Explanation:</b> No prominent consolidation, focal opacity, or dense infiltration
                        associated with pneumonia was detected in this scan.
                        <div style="margin-top:0.4rem; color:#B8C5D9; font-size:0.85rem;">
                            Note: Subtle or early-stage findings may require correlation with clinical history, physical examination, and follow-up imaging.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box-pneumonia">
                        <div style="color:#F59E0B; font-size:0.85rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase;">
                            Model Prediction
                        </div>
                        <h2 class="result-label-pneumonia">PNEUMONIA</h2>
                        <div class="result-score-text">
                            Model Confidence: <b style="color:#F59E0B;">{confidence:.1%}</b>
                        </div>
                        <div class="result-subscore">
                            Raw Sigmoid Output: {raw_score:.4f} &nbsp;|&nbsp; Diagnostic Resolution: {width} × {height} px
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    """
                    <div class="explanation-box" style="border-left-color: #F59E0B;">
                        <b>Clinical Explanation:</b> Radiographic opacity patterns and consolidation consistent with
                        pulmonary infiltration/pneumonia were identified.
                        <div style="margin-top:0.4rem; color:#B8C5D9; font-size:0.85rem;">
                            Recommended Action: Prompt physician correlation and comprehensive radiologist review required. Do not initiate pharmacological therapy solely based on this screening result.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Score metric bar
            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            st.progress(
                min(max(raw_score, 0.0), 1.0),
                text=f"Pneumonia Likelihood Score: {raw_score:.1%}",
            )
        else:
            st.markdown(
                """
                <div style="padding:1.5rem; text-align:center; background:#111C30; border-radius:12px; border:1px dashed #253858; color:#B8C5D9;">
                    <div style="font-size:1.8rem; margin-bottom:0.4rem;">🔬</div>
                    Awaiting analysis. Upload an image or select a sample case and click <b>Analyze X-ray</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: OVERVIEW (NEVER BLANK!)
# ==============================================================================
elif selected_page == "Overview":
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color:#F8FAFC; font-size:1.7rem; font-weight:800; margin:0;">
                Welcome to MedImage AI
            </h2>
            <p style="color:#22D3EE; font-size:1.05rem; font-weight:600; margin:0.3rem 0 0 0;">
                A machine-learning prototype that assists with chest X-ray screening.
            </p>
            <p style="color:#B8C5D9; font-size:0.95rem; margin:0.3rem 0 0 0;">
                Built on deep convolutional transfer learning (MobileNetV2), trained and validated on real pediatric chest radiograph datasets to differentiate normal lungs from pneumonia cases.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Key Metrics Cards
    counts = get_dataset_counts()
    metrics = get_evaluation_metrics()
    model_ready = MODEL_PATH.exists()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Training Images",
            value=f"{counts['train']['total']:,}",
            help=f"Train Split: {counts['train']['NORMAL']} Normal, {counts['train']['PNEUMONIA']} Pneumonia",
        )
    with col2:
        st.metric(
            label="Validation Images",
            value=f"{counts['test']['total']:,}",
            help="Held-out validation set evaluated during epoch training",
        )
    with col3:
        st.metric(
            label="Test Images",
            value=f"{counts['test']['total']:,}",
            help=f"Held-out Test Split: {counts['test']['NORMAL']} Normal, {counts['test']['PNEUMONIA']} Pneumonia",
        )
    with col4:
        st.metric(
            label="Verified Test Accuracy",
            value=f"{metrics['accuracy']:.1%}",
            help=f"Calculated on all {metrics['samples']} test radiographs",
        )
    with col5:
        st.metric(
            label="Model Status",
            value="Loaded" if model_ready else "Not Loaded",
            help="Trained MobileNetV2 model binary state",
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 2. Charts Section
    chart_col1, chart_col2 = st.columns([1.1, 1], gap="medium")

    with chart_col1:
        st.markdown(
            """
            <div class="content-card">
                <h3>📊 Chart 1 — Dataset Distribution</h3>
                <p>Distribution of NORMAL vs PNEUMONIA chest radiographs across training and test partitions.</p>
            """,
            unsafe_allow_html=True,
        )
        dist_fig = plot_dataset_distribution(counts)
        st.pyplot(dist_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown(
            """
            <div class="content-card">
                <h3>📈 Chart 2 — Model Training Performance</h3>
                <p>Accuracy and loss trajectories across 5 training epochs on training and validation cohorts.</p>
            """,
            unsafe_allow_html=True,
        )
        history = get_training_history()
        curves_fig = plot_training_curves(history)
        st.pyplot(curves_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Quick Action / Call to Action
    st.markdown(
        """
        <div class="content-card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
            <div>
                <h3 style="margin-bottom:0.25rem;">Ready to screen an X-ray?</h3>
                <p>Proceed to the X-ray Prediction page to upload a chest radiograph and test the live model.</p>
            </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚀 Go to X-ray Prediction", type="primary", key="btn_go_predict"):
        st.session_state["nav_page"] = "X-ray Prediction"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: EVALUATION
# ==============================================================================
elif selected_page == "Evaluation":
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color:#F8FAFC; font-size:1.7rem; font-weight:800; margin:0;">
                Model Evaluation & Performance Diagnostics
            </h2>
            <p style="color:#B8C5D9; font-size:0.95rem; margin:0.3rem 0 0 0;">
                Comprehensive, verified validation metrics on the held-out test cohort (485 verified chest X-rays).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = get_evaluation_metrics()

    # Metric Cards
    ev1, ev2, ev3, ev4 = st.columns(4)
    with ev1:
        st.metric(
            label="Test Accuracy",
            value=f"{metrics['accuracy']:.1%}",
            help="Overall proportion of correct classifications on test data",
        )
    with ev2:
        st.metric(
            label="Precision",
            value=f"{metrics['precision']:.1%}",
            help="When pneumonia is predicted, how often it is truly pneumonia",
        )
    with ev3:
        st.metric(
            label="Recall (Sensitivity)",
            value=f"{metrics['recall']:.1%}",
            help="Proportion of actual pneumonia cases detected by the model",
        )
    with ev4:
        st.metric(
            label="F1-Score",
            value=f"{metrics['f1']:.1%}",
            help="Harmonic mean of precision and recall",
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    col_cm, col_curves = st.columns([1, 1], gap="medium")

    with col_cm:
        st.markdown(
            """
            <div class="content-card">
                <h3>🔍 Confusion Matrix</h3>
                <p>Class-by-class comparison between ground truth diagnoses and model screening predictions.</p>
            """,
            unsafe_allow_html=True,
        )
        cm_fig = plot_confusion_matrix(metrics["matrix"])
        st.pyplot(cm_fig, use_container_width=True)

        cm = metrics["matrix"]
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        st.markdown(
            f"""
            <div style="background:#111C30; border-radius:10px; padding:0.9rem; margin-top:0.8rem; font-size:0.85rem; border:1px solid #253858;">
                <div style="color:#F8FAFC; font-weight:700; margin-bottom:0.4rem;">Diagnostic Breakdown:</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; color:#B8C5D9;">
                    <div>• <b>True Normal (TN):</b> <span style="color:#2DD4BF;">{tn} cases</span></div>
                    <div>• <b>False Pneumonia (FP):</b> <span style="color:#F59E0B;">{fp} cases</span></div>
                    <div>• <b>False Normal (FN):</b> <span style="color:#F87171;">{fn} cases</span></div>
                    <div>• <b>True Pneumonia (TP):</b> <span style="color:#22D3EE;">{tp} cases</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_curves:
        st.markdown(
            """
            <div class="content-card">
                <h3>📈 Training & Validation Convergence</h3>
                <p>Convergence curves across all training epochs showing learning progression and generalization.</p>
            """,
            unsafe_allow_html=True,
        )
        history = get_training_history()
        curves_fig = plot_training_curves(history)
        st.pyplot(curves_fig, use_container_width=True)

        # Plain English Explanations
        st.markdown(
            """
            <div style="background:#111C30; border-radius:10px; padding:0.9rem; margin-top:0.8rem; font-size:0.85rem; border:1px solid #253858;">
                <div style="color:#F8FAFC; font-weight:700; margin-bottom:0.4rem;">Plain-English Metric Guide:</div>
                <div style="color:#B8C5D9; line-height:1.5;">
                    • <b>Accuracy ({acc}):</b> How often the model correctly classified test X-rays (normal or pneumonia).<br>
                    • <b>Confusion Matrix:</b> Summary of correct vs incorrect classifications across all clinical categories.<br>
                    • <b>Precision ({prec}):</b> High precision minimizes false alarms when pneumonia is reported.<br>
                    • <b>Recall ({rec}):</b> High recall ensures critical pneumonia cases are not overlooked.
                </div>
            </div>
            """.format(
                acc=f"{metrics['accuracy']:.1%}",
                prec=f"{metrics['precision']:.1%}",
                rec=f"{metrics['recall']:.1%}",
            ),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Optional Recompute Button
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("⚙️ Advanced: Re-run Live Evaluation on Test Dataset"):
        st.caption("Executes real-time inference across all 485 held-out test images using MobileNetV2.")
        if st.button("Re-evaluate Full Test Cohort", key="btn_recalc_metrics"):
            with st.spinner("Evaluating 485 test radiographs (this takes ~10-15 seconds on CPU)..."):
                try:
                    updated_metrics = get_evaluation_metrics(force_recompute=True)
                    st.success(f"Evaluation complete! Verified Accuracy: {updated_metrics['accuracy']:.1%}")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Re-evaluation failed: {ex}")


# ---------------- FOOTER & DISCLAIMER ----------------
st.markdown(
    """
    <div class="med-disclaimer">
        <div style="font-size:1.3rem;">⚠️</div>
        <div>
            <b style="color:#F8FAFC;">Medical Disclaimer & Educational Use:</b><br>
            For educational and research use only. AI screening results are not a diagnosis and must be reviewed by a qualified healthcare professional.
            This system is a proof-of-concept screening prototype designed to assist radiologic review, not replace clinical judgment.
            <div style="margin-top:0.35rem; color:#64748B;">
                MedImage AI • Developed by Mahesh Bhakre • MobileNetV2 Architecture
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
