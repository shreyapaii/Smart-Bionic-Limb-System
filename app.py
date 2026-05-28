"""
AI-Based Low-Cost Bionic Limb Gesture Recognition System
Streamlit Healthcare Application  –  Stabilized v4 (Complete Fix)

This application provides a comprehensive interface for EMG signal analysis,
gesture prediction, and rehabilitation monitoring.

Author: Healthcare ML Project Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from PIL import Image
import traceback

# Session storage and visualization (NEW - modular additions)
import session_manager
import prosthetic_hand
import emg_pipeline

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bionic Limb AI System",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (dark healthcare aesthetic) ────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&display=swap');
 
    /* ── Design tokens ── */
    :root {
        --bg-base:        #07090f;
        --bg-surface:     #0d1017;
        --bg-elevated:    #111827;
        --bg-glass:       rgba(255,255,255,0.025);
        --border-dim:     rgba(255,255,255,0.065);
        --border-accent:  rgba(45,212,191,0.22);
 
        --teal:           #2dd4bf;
        --teal-dim:       #0d9488;
        --teal-muted:     rgba(45,212,191,0.12);
        --lavender:       #818cf8;
        --lavender-muted: rgba(129,140,248,0.12);
        --mint:           #34d399;
        --amber:          #fbbf24;
        --rose:           #fb7185;
 
        --text-primary:   #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted:     #4b5563;
        --text-accent:    #2dd4bf;
 
        /* legacy aliases for inline HTML */
        --primary-color:   #2dd4bf;
        --secondary-color: #0d9488;
        --accent-color:    #818cf8;
        --success-color:   #34d399;
        --warning-color:   #fbbf24;
        --danger-color:    #fb7185;
    }
 
    /* ── Global reset / base ── */
    html, body, .stApp {
        background-color: var(--bg-base) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-primary) !important;
    }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: rgba(45,212,191,0.3); border-radius: 2px; }
 
    /* ── Main content padding ── */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px !important;
    }
 
    /* ── Page header banner ── */
    .main-header {
        background: linear-gradient(135deg, #0c1424 0%, #0a1221 40%, #0e1829 100%);
        border: 1px solid rgba(45,212,191,0.14);
        border-bottom: 1px solid rgba(45,212,191,0.28);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -80px; left: 50%;
        transform: translateX(-50%);
        width: 500px; height: 160px;
        background: radial-gradient(ellipse, rgba(45,212,191,0.1) 0%, transparent 65%);
        pointer-events: none;
    }
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(45,212,191,0.4) 40%, rgba(129,140,248,0.3) 60%, transparent 100%);
    }
    .main-header h1 {
        font-family: 'Syne', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(100deg, #e2e8f0 0%, #2dd4bf 55%, #818cf8 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }
    .main-header p {
        font-size: 0.8rem !important;
        margin-top: 0.6rem !important;
        color: var(--text-secondary) !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        font-weight: 500 !important;
    }
 
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070a12 0%, #0a0e18 100%) !important;
        border-right: 1px solid var(--border-dim) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-muted) !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        margin: 0.25rem 0 0.5rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-secondary) !important;
        font-size: 0.83rem !important;
        line-height: 1.6 !important;
    }
    [data-testid="stSidebar"] h1 {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: 0.02em !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.05) !important;
        margin: 0.8rem 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: var(--text-secondary) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        transition: color 0.15s !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        color: var(--teal) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [data-checked="true"] + div,
    [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + label {
        color: var(--teal) !important;
    }
 
    /* ── st.metric ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.022) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 12px !important;
        padding: 1.1rem 1.3rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(45,212,191,0.22) !important;
        box-shadow: 0 0 18px rgba(45,212,191,0.06) !important;
    }
    [data-testid="stMetricLabel"] > div {
        color: var(--text-muted) !important;
        font-size: 0.68rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricValue"] > div {
        font-family: 'DM Mono', monospace !important;
        color: var(--text-primary) !important;
        font-size: 1.55rem !important;
        font-weight: 400 !important;
        letter-spacing: -0.01em !important;
    }
    [data-testid="stMetricDelta"] > div {
        color: var(--mint) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }
 
    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--teal-dim) 0%, var(--teal) 100%) !important;
        color: #05090f !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 2rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        box-shadow: 0 2px 14px rgba(45,212,191,0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 22px rgba(45,212,191,0.38) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }
 
    /* ── Gesture result card ── */
    .gesture-result {
        background: linear-gradient(135deg, rgba(13,148,136,0.12) 0%, rgba(129,140,248,0.08) 100%);
        border: 1px solid rgba(45,212,191,0.28);
        border-radius: 16px;
        padding: 2.2rem 2rem;
        text-align: center;
        font-family: 'Syne', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        color: var(--teal);
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
        animation: gesture-pulse 3s ease-in-out infinite;
    }
    .gesture-result::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(45,212,191,0.07) 0%, transparent 55%);
        pointer-events: none;
    }
    @keyframes gesture-pulse {
        0%, 100% { box-shadow: 0 0 24px rgba(45,212,191,0.08), inset 0 0 24px rgba(45,212,191,0.03); }
        50%       { box-shadow: 0 0 40px rgba(45,212,191,0.16), inset 0 0 32px rgba(45,212,191,0.06); }
    }
 
    /* ── Metric / info cards ── */
    .metric-card {
        background: rgba(255,255,255,0.022);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-dim);
        border-top: 2px solid var(--teal);
        padding: 1.4rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        margin: 0.8rem 0;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        border-color: rgba(45,212,191,0.28);
        box-shadow: 0 4px 28px rgba(45,212,191,0.07);
    }
 
    /* ── Alert / info boxes ── */
    .success-box {
        background: rgba(52,211,153,0.07);
        border: 1px solid rgba(52,211,153,0.22);
        border-left: 3px solid var(--mint);
        border-radius: 8px;
        padding: 1rem 1.3rem;
        color: #a7f3d0;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .info-box {
        background: rgba(129,140,248,0.07);
        border: 1px solid rgba(129,140,248,0.2);
        border-left: 3px solid var(--lavender);
        border-radius: 8px;
        padding: 1rem 1.3rem;
        color: #c7d2fe;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
 
    /* ── Streamlit native alerts override ── */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
        border-left-width: 3px !important;
    }
 
    /* ── Headings ── */
    .stApp h1, .stApp h2 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }
    .stApp h3 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        font-size: 0.95rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        margin-top: 1.8rem !important;
    }
    .stApp h4 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
    }
 
    /* ── Dividers ── */
    hr { border-color: var(--border-dim) !important; margin: 1.2rem 0 !important; }
 
    /* ── Dataframe table ── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-dim) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
 
    /* ── File uploader ── */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(45,212,191,0.025) !important;
        border: 1px dashed rgba(45,212,191,0.25) !important;
        border-radius: 12px !important;
        transition: all 0.2s !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(45,212,191,0.45) !important;
        background: rgba(45,212,191,0.04) !important;
    }
 
    /* ── Selectbox / inputs ── */
    .stSelectbox [data-baseweb="select"] > div:first-child {
        background: var(--bg-elevated) !important;
        border-color: var(--border-dim) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    .stTextInput input, .stNumberInput input {
        background: var(--bg-elevated) !important;
        border-color: var(--border-dim) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    .stSlider [data-testid="stTickBar"] {
        color: var(--text-muted) !important;
    }
    [data-testid="stSlider"] > div > div > div > div {
        background: var(--teal) !important;
    }
 
    /* ── Download button ── */
    .stDownloadButton > button {
        background: rgba(45,212,191,0.1) !important;
        border: 1px solid rgba(45,212,191,0.3) !important;
        color: var(--teal) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(45,212,191,0.18) !important;
        box-shadow: 0 4px 16px rgba(45,212,191,0.2) !important;
        transform: translateY(-1px) !important;
    }
 
    /* ── Spinner ── */
    [data-testid="stSpinner"] > div {
        border-top-color: var(--teal) !important;
    }
 
    /* ── Checkbox ── */
    .stCheckbox label {
        color: var(--text-secondary) !important;
        font-size: 0.88rem !important;
    }
 
    /* ── Progress / status pill ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(52,211,153,0.1);
        border: 1px solid rgba(52,211,153,0.25);
        border-radius: 20px;
        padding: 0.25rem 0.85rem;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--mint);
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .status-pill::before {
        content: '';
        width: 6px; height: 6px;
        background: var(--mint);
        border-radius: 50%;
        animation: dot-pulse 2s ease-in-out infinite;
    }
    @keyframes dot-pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.4; transform: scale(0.8); }
    }
 
    /* ── Section header strip ── */
    .section-label {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--teal);
        margin: 2rem 0 1rem;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(45,212,191,0.25), transparent);
    }
 
    /* ── About page footer card ── */
    .about-footer {
        text-align: center;
        padding: 2.2rem;
        background: linear-gradient(135deg, rgba(13,148,136,0.08) 0%, rgba(129,140,248,0.06) 100%);
        border: 1px solid rgba(45,212,191,0.14);
        border-radius: 14px;
        margin-top: 1.5rem;
    }
    .about-footer h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        color: var(--teal) !important;
        text-transform: none !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 0.4rem !important;
    }
    .about-footer p {
        color: var(--text-secondary) !important;
        font-size: 0.88rem !important;
        margin: 0 !important;
    }
 
    /* ── Hide Streamlit chrome ── */
    #MainMenu                         { visibility: hidden; }
    footer                            { visibility: hidden; }
    header[data-testid="stHeader"]    { background: transparent !important; }
    </style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
_CHANNELS = [f"channel{i}" for i in range(1, 9)]
_STATS     = ["mean", "rms", "variance", "std", "energy", "max", "min"]
STANDARD_FEATURE_NAMES = [f"{ch}_{s}" for ch in _CHANNELS for s in _STATS]

NON_FEATURE_COLS = frozenset([
    "time", "class", "predicted_gesture", "confidence",
    "actual_gesture", "correct"
])

def _label(raw) -> str:
    """Convert a raw class label to a display-friendly string."""
    try:
        return f"Gesture {int(float(raw))}"
    except Exception:
        return str(raw)

def _fmt_pct(value, default="N/A") -> str:
    """Safely format a 0–1 float as a percentage string."""
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return f"{v * 100:.2f}%"
    except Exception:
        return default

def safe_get(d, key, default=None):
    """Safe nested dictionary access."""
    try:
        return d.get(key, default) if isinstance(d, dict) else default
    except Exception:
        return default


class BionicLimbApp:
    """Main application class for EMG gesture recognition system."""

    def __init__(self):
        self.model         = None
        self.label_encoder = None
        self.metadata      = None
        self.model_loaded  = False
        self._init_metadata()
        # Session tracking (NEW - modular addition)
        if "patient_id" not in st.session_state:
            st.session_state.patient_id = "P001"
        if "current_session_start" not in st.session_state:
            st.session_state.current_session_start = datetime.now()

    def _init_metadata(self):
        """Initialize metadata with safe defaults."""
        self.metadata = {
            "model_name": "Unknown",
            "model_type": "Unknown",
            "n_classes": 8,
            "n_features": 56,
            "emg_channels": 8,
            "training_date": "N/A",
            "best_hyperparameters": {},
            "class_names": [f"Gesture {i}" for i in range(8)],
            "feature_names": STANDARD_FEATURE_NAMES,
            "metrics": {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "training_date": "N/A",
                "confusion_matrix": [],
                "feature_importance": [],
            }
        }

    def load_model(self):
        """Load trained model + label encoder + metadata."""
        try:
            best_path = "models/best_gesture_model.joblib"
            rf_path   = "models/random_forest_model.joblib"
            model_path = best_path if os.path.exists(best_path) else rf_path

            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found: {model_path}")
                return False

            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load("models/label_encoder.joblib")
            n_classes_from_encoder = len(self.label_encoder.classes_)

            with open("models/model_metadata.json", "r") as f:
                raw = json.load(f)

            n_classes  = int(safe_get(raw, "number_of_classes",
                             safe_get(raw, "n_classes", n_classes_from_encoder)))
            n_features = int(safe_get(raw, "number_of_features",
                              safe_get(raw, "n_features", len(STANDARD_FEATURE_NAMES))))

            self.metadata = {
                "model_name":  safe_get(raw, "model_name", "LightGBM"),
                "model_type":  safe_get(raw, "model_type", safe_get(raw, "model_name", "LightGBM")),
                "n_classes":   n_classes,
                "n_features":  n_features,
                "emg_channels": 8,
                "training_date": safe_get(raw, "training_date", "N/A"),
                "best_hyperparameters": safe_get(raw, "best_hyperparameters", {}),
            }

            self.metadata["metrics"] = {
                "accuracy":          float(safe_get(raw, "accuracy", 0.0)),
                "precision":         float(safe_get(raw, "precision", 0.0)),
                "recall":            float(safe_get(raw, "recall", 0.0)),
                "f1_score":          float(safe_get(raw, "f1_score", safe_get(raw, "f1", 0.0))),
                "training_date":     safe_get(raw, "training_date", "N/A"),
                "confusion_matrix":  safe_get(raw, "confusion_matrix", []),
                "feature_importance": safe_get(raw, "feature_importance", []),
            }

            encoder_classes = self.label_encoder.classes_
            if "class_names" in raw and raw["class_names"]:
                self.metadata["class_names"] = [str(c) for c in raw["class_names"]]
            else:
                self.metadata["class_names"] = [_label(c) for c in encoder_classes]

            eng_path = os.path.join("data", "engineered_emg_features.csv")
            if "feature_names" in raw and raw["feature_names"]:
                self.metadata["feature_names"] = raw["feature_names"]
            elif os.path.exists(eng_path):
                try:
                    cols = pd.read_csv(eng_path, nrows=0).columns.tolist()
                    self.metadata["feature_names"] = [c for c in cols if c not in NON_FEATURE_COLS]
                except Exception as e:
                    st.warning(f"Could not read feature names from CSV: {e}")
                    self.metadata["feature_names"] = STANDARD_FEATURE_NAMES
            else:
                self.metadata["feature_names"] = STANDARD_FEATURE_NAMES

            self.model_loaded = True
            return True

        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("💡 Please train the model first:  `python train_model.py`")
            return False

    def predict_gesture(self, emg_data: pd.DataFrame):
        """Predict gesture class from a DataFrame."""
        if emg_data.empty:
            raise ValueError("Input data is empty")

        stored = self.metadata.get("feature_names", [])
        if stored:
            feature_cols = [c for c in stored if c in emg_data.columns]
        else:
            feature_cols = [c for c in emg_data.columns if c not in NON_FEATURE_COLS]

        if not feature_cols:
            raise ValueError(
                "No engineered feature columns found in the uploaded file.\n"
                "Expected columns like channel1_mean, channel1_rms, …\n"
                f"Got: {list(emg_data.columns)}"
            )

        try:
            X = emg_data[feature_cols].fillna(0).astype(np.float32)
            preds_encoded  = self.model.predict(X)
            probabilities  = self.model.predict_proba(X)

            preds_encoded = preds_encoded.astype(int)
            preds_raw = self.label_encoder.inverse_transform(preds_encoded)

            class_names = self.metadata.get("class_names", [])
            encoder_classes = list(self.label_encoder.classes_)

            def raw_to_label(raw_val):
                try:
                    raw_val_float = float(raw_val)
                    idx = int(raw_val_float)
                    if 0 <= idx < len(class_names):
                        return str(class_names[idx])
                except (ValueError, IndexError, TypeError):
                    pass
                return _label(raw_val)

            predictions_labels = [raw_to_label(v) for v in preds_raw]
            return predictions_labels, probabilities

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def render_header(self):
        st.markdown("""
            <div class="main-header">
                <h1>🦾 AI-Based Bionic Limb Control System</h1>
                <p>Advanced EMG Signal Analysis for Prosthetic Gesture Recognition</p>
            </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        st.sidebar.title("🏥 Navigation")
        st.sidebar.markdown("---")

        page = st.sidebar.radio(
            "Select Module:",
            [
                "🏠 Home",
                "🔬 Real-Time Prediction",
                "📊 Model Performance",
                "👤 Patient Dashboard",
                "📈 Rehabilitation Monitor",
                "ℹ️ About System",
            ],
        )

        st.sidebar.markdown("---")
        # Patient selection (NEW - modular addition)
        st.sidebar.markdown("### 👤 Current Patient")
        patient_id = st.sidebar.selectbox(
            "Select Patient ID:",
            ["P001", "P002", "P003", "P004", "P005"],
            key="sidebar_patient_select"
        )
        st.session_state.patient_id = patient_id

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 System Status")

        if self.model_loaded:
            m = self.metadata
            acc = safe_get(m.get("metrics", {}), "accuracy", 0.0)
            st.sidebar.success("✅ Model Loaded")
            st.sidebar.info(f"📅 Trained: {m.get('training_date', 'N/A')}")
            st.sidebar.metric("Accuracy", _fmt_pct(acc))
        else:
            st.sidebar.error("❌ Model Not Loaded")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🤖 Quick Stats")
        if self.model_loaded:
            st.sidebar.write(f"**Gestures:** {self.metadata.get('n_classes', '—')}")
            st.sidebar.write(f"**EMG Channels:** {self.metadata.get('emg_channels', 8)}")
            st.sidebar.write(f"**Feature Count:** {self.metadata.get('n_features', '—')}")

        return page

    def home_page(self):
        st.title("Welcome to the Bionic Limb AI System")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
                ### 🎯 Project Overview

                This advanced healthcare system uses **Machine Learning** to recognize hand gestures
                from EMG (Electromyography) muscle signals, enabling intelligent control of low-cost
                prosthetic and bionic limbs.

                #### 🌟 Key Features:
                - ✅ Real-time gesture prediction from EMG signals
                - ✅ Support for 8-channel EMG data acquisition
                - ✅ High-accuracy ensemble model classification
                - ✅ Patient rehabilitation monitoring
                - ✅ Progress tracking and analytics
                - ✅ User-friendly healthcare interface

                #### 🎓 Target Users:
                - 🏥 Healthcare professionals and physiotherapists
                - 🦾 Prosthetic limb users and amputees
                - 🔬 Researchers in assistive technology
                - 👨‍💻 ML engineers and developers
            """)

        with col2:
            st.markdown("### 📋 Supported Gestures")
            if self.model_loaded:
                class_names = self.metadata.get("class_names", [])
                if class_names:
                    for i, name in enumerate(class_names, 1):
                        st.info(f"**{i}.** {str(name).upper()}")
                else:
                    st.warning("No gestures defined")
            else:
                st.warning("Load model to view gestures")

        st.markdown("---")
        st.subheader("📊 System Capabilities")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("EMG Channels", "8", delta="Multi-channel")

        with col2:
            if self.model_loaded:
                acc = safe_get(self.metadata.get("metrics", {}), "accuracy", 0.0)
                st.metric("Model Accuracy", _fmt_pct(acc), delta="High Performance" if acc > 0.95 else None)
            else:
                st.metric("Model Accuracy", "N/A")

        with col3:
            if self.model_loaded:
                st.metric("Gestures", self.metadata.get("n_classes", "—"), delta="Recognized")
            else:
                st.metric("Gestures", "N/A")

        with col4:
            st.metric("Response Time", "< 100ms", delta="Real-time")

        st.markdown("---")
        st.subheader("🔄 How It Works")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("#### 1️⃣ Data Collection\nEMG sensors capture muscle signals from 8 channels on the forearm")
        with col2:
            st.markdown("#### 2️⃣ Signal Processing\nRaw signals are normalized and 56 statistical features are extracted")
        with col3:
            st.markdown("#### 3️⃣ ML Prediction\nEnsemble model predicts the intended gesture with high accuracy")
        with col4:
            st.markdown("#### 4️⃣ Limb Control\nProsthetic limb executes the predicted movement")

    def prediction_page(self):
        st.title("🔬 Real-Time Gesture Prediction")

        if not self.model_loaded:
            st.error("❌ Model not loaded. Please train the model first.")
            return

        st.markdown("""
            Upload **raw EMG data** or **engineered features** to predict hand gestures.

            📊 **Supported Formats:**
            - **Raw EMG:** CSV with `channel1-channel8` columns (auto-preprocesses)
            - **Engineered Features:** CSV with `channel1_mean`, `channel1_rms`, etc.

            The system automatically detects and processes your data type.
        """)

        st.markdown("### 📁 Upload EMG Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Raw EMG (channel1-channel8) or engineered features (channel1_mean, etc.). "
                 "Optional: time, class columns.",
        )

        if uploaded_file is None:
            return

        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            return

        # Process through unified pipeline (NEW - intelligent format detection)
        try:
            with st.spinner("🔬 Analyzing data format..."):
                df_processed, input_type = emg_pipeline.process_emg_file(df_raw, verbose=False)

            if input_type == "raw":
                st.info("🔬 **Raw EMG Detected** → Auto-preprocessing activated")
                st.info("✅ **Preprocessing:** Normalization + Feature Extraction completed")
            else:
                st.success("✅ **Engineered Features** detected → Ready for prediction")

        except ValueError as e:
            st.error(f"❌ Data processing failed: {e}")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error during processing: {e}")
            st.warning(f"Debug info: {str(e)}")
            return

        st.success(f"✅ File processed successfully! ({len(df_processed):,} samples, {len(df_processed.columns)} features)")

        st.markdown("### 📊 Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", f"{len(df_processed):,}")
        with col2:
            st.metric("Features", len(df_processed.columns))
        with col3:
            if "class" in df_raw.columns:
                try:
                    n_unique = df_raw["class"].nunique()
                    st.metric("Gestures", n_unique)
                except Exception:
                    st.metric("Status", "Ready")
            else:
                st.metric("Status", "Ready")

        st.markdown("### 📋 Feature Preview")
        preview_cols = [col for col in df_processed.columns if col != "class"][:5]
        st.dataframe(df_processed[preview_cols].head(5), use_container_width=True)
        if len(preview_cols) < len(df_processed.columns):
            st.caption(f"Showing first 5 of {len(df_processed.columns)} features")

        st.markdown("### ⚙️ Prediction Settings")
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider(
                "Number of samples to predict:",
                min_value=10,
                max_value=min(1000, len(df_processed)),
                value=min(100, len(df_processed)),
                step=10,
            )
        with col2:
            show_confidence = st.checkbox("Show prediction confidence", value=True)

        if not st.button("🚀 Predict Gestures", type="primary"):
            return

        with st.spinner("Generating predictions…"):
            df_sample = df_processed.head(sample_size).copy()

            try:
                predictions, probabilities = self.predict_gesture(df_sample)
            except ValueError as e:
                st.error(f"❌ Prediction failed: {e}")
                return
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                return

            df_sample["predicted_gesture"] = predictions
            df_sample["confidence"] = np.max(probabilities, axis=1) * 100

        # Save session to persistent storage (NEW - modular integration)
        most_common = pd.Series(predictions).mode()
        most_common = most_common[0] if len(most_common) > 0 else "Unknown"
        avg_conf = df_sample["confidence"].mean() / 100
        session_duration = (datetime.now() - st.session_state.current_session_start).total_seconds()

        accuracy = None
        if "class" in df_raw.columns:
            class_names = self.metadata.get("class_names", [])
            encoder_classes = list(self.label_encoder.classes_)
            def raw_class_to_label(raw_val):
                try:
                    idx = int(float(raw_val))
                    if 0 <= idx < len(class_names):
                        return str(class_names[idx])
                except (ValueError, IndexError, TypeError):
                    pass
                return _label(raw_val)
            actual_labels = df_sample["class"].map(raw_class_to_label)
            accuracy = (predictions == actual_labels).mean()

        session_manager.save_session(
            patient_id=st.session_state.patient_id,
            predicted_gesture=str(most_common),
            confidence=avg_conf,
            rows_analyzed=sample_size,
            accuracy=accuracy,
            session_duration_sec=session_duration,
        )

        st.markdown("### 🎯 Prediction Results")

        if predictions:
            most_common = pd.Series(predictions).mode()
            most_common = most_common[0] if len(most_common) > 0 else "Unknown"
            avg_conf = df_sample["confidence"].mean()

            st.markdown(f"""
                <div class="gesture-result">
                    🦾 DETECTED GESTURE: {str(most_common).upper()}
                </div>
            """, unsafe_allow_html=True)

            # Display prosthetic hand visualization with gesture-specific confidence (FIXED - modular integration)
            gesture_mask = df_sample["predicted_gesture"] == most_common
            if gesture_mask.any():
                gesture_specific_conf = df_sample[gesture_mask]["confidence"].mean() / 100
            else:
                gesture_specific_conf = avg_conf / 100
            prosthetic_hand.get_hand_component(str(most_common), gesture_specific_conf)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Confidence", f"{avg_conf:.2f}%")
            with col2:
                st.metric("Samples Analyzed", sample_size)
            with col3:
                st.metric("Unique Gestures Detected", len(set(predictions)))

            st.markdown("### 📊 Gesture Distribution")
            gesture_counts = pd.Series(predictions).value_counts()
            if not gesture_counts.empty:
                fig = px.pie(
                    values=gesture_counts.values,
                    names=[str(g) for g in gesture_counts.index],
                    title="Distribution of Predicted Gestures",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)

            if show_confidence:
                st.markdown("### 📋 Detailed Predictions")
                display_cols = ["predicted_gesture", "confidence"]
                display_df = df_sample[display_cols].copy()

                if "class" in df.columns:
                    class_names = self.metadata.get("class_names", [])
                    encoder_classes = list(self.label_encoder.classes_)

                    def raw_class_to_label(raw_val):
                        try:
                            idx = int(float(raw_val))
                            if 0 <= idx < len(class_names):
                                return str(class_names[idx])
                        except (ValueError, IndexError, TypeError):
                            pass
                        return _label(raw_val)

                    display_df["actual_gesture"] = df_sample["class"].map(raw_class_to_label)
                    display_df["correct"] = display_df["predicted_gesture"] == display_df["actual_gesture"]

                st.dataframe(display_df.head(50), use_container_width=True)

                if "correct" in display_df.columns:
                    acc = display_df["correct"].mean() * 100
                    st.success(f"✅ Prediction Accuracy on uploaded data: {acc:.2f}%")

            st.markdown("### 💾 Download Results")
            csv_out = df_sample.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv_out,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    def performance_page(self):
        st.title("📊 Model Performance Dashboard")

        if not self.model_loaded:
            st.error("❌ Model not loaded. Please train the model first.")
            return

        metrics = self.metadata.get("metrics", {})
        class_names = self.metadata.get("class_names", [])

        st.markdown("### 🎯 Overall Performance Metrics")

        acc = safe_get(metrics, "accuracy", 0.0)
        prec = safe_get(metrics, "precision", 0.0)
        rec = safe_get(metrics, "recall", 0.0)
        f1 = safe_get(metrics, "f1_score", 0.0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta = f"+{(acc - 0.90)*100:.1f}%" if acc > 0.90 else None
            st.metric("Accuracy", _fmt_pct(acc), delta=delta)
        with col2:
            st.metric("Precision", _fmt_pct(prec))
        with col3:
            st.metric("Recall", _fmt_pct(rec))
        with col4:
            st.metric("F1-Score", _fmt_pct(f1))

        st.markdown("### 📈 Performance Overview")
        metric_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Score": [acc * 100, prec * 100, rec * 100, f1 * 100],
        })
        fig = px.bar(
            metric_df, x="Metric", y="Score",
            title="Model Performance Metrics (%)",
            color="Score", color_continuous_scale="Blues",
            text="Score",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(yaxis_range=[0, 105], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔲 Confusion Matrix")
        cm_data = safe_get(metrics, "confusion_matrix", [])
        if cm_data and len(cm_data) > 0:
            try:
                cm = np.array(cm_data)
                labels = class_names if len(class_names) == cm.shape[0] else [f"G{i}" for i in range(cm.shape[0])]
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=labels, y=labels,
                    color_continuous_scale="Blues",
                    text_auto=True,
                    title="Confusion Matrix – Gesture Classification",
                )
                fig.update_xaxes(side="bottom")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display confusion matrix: {e}")

        for img_name in ["confusion_matrix_best_model.png", "confusion_matrix.png"]:
            img_path = os.path.join("models", img_name)
            if os.path.exists(img_path):
                try:
                    st.image(img_path, caption="Confusion Matrix (from last training run)", use_container_width=True)
                    break
                except Exception as e:
                    st.warning(f"Could not load {img_name}: {e}")

        st.markdown("### 🔍 Feature Importance")
        fi_data = safe_get(metrics, "feature_importance", [])
        if fi_data and len(fi_data) > 0:
            try:
                fi_df = pd.DataFrame(fi_data).head(20)
                fig = px.bar(
                    fi_df, x="importance", y="feature", orientation="h",
                    title="Top Feature Importances",
                    color="importance", color_continuous_scale="Teal",
                    text="importance",
                )
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display feature importance: {e}")

        for img_name in ["feature_importance_best_model.png", "feature_importance.png"]:
            img_path = os.path.join("models", img_name)
            if os.path.exists(img_path):
                try:
                    st.image(img_path, caption="Feature Importances (from last training run)", use_container_width=True)
                    break
                except Exception as e:
                    st.warning(f"Could not load {img_name}: {e}")

        cmp_path = os.path.join("models", "model_comparison.png")
        if os.path.exists(cmp_path):
            st.markdown("### 🏆 Model Comparison")
            try:
                st.image(cmp_path, caption="Comparison of all trained models", use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load model comparison: {e}")

        st.markdown("### ℹ️ Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
                **Model Type:** {self.metadata.get('model_type', '—')}
                **Feature Count:** {self.metadata.get('n_features', '—')}
                **Number of Classes:** {self.metadata.get('n_classes', '—')}
                **EMG Channels:** {self.metadata.get('emg_channels', 8)}
            """)
        with col2:
            class_str = ', '.join([str(c) for c in class_names]) if class_names else '—'
            st.success(f"""
                **Training Date:** {safe_get(metrics, 'training_date', 'N/A')}
                **Gestures:** {class_str}
            """)

    def patient_dashboard(self):
        st.title("👤 Patient Dashboard")

        st.markdown("""
            Monitor patient progress, track rehabilitation metrics, and analyze gesture performance over time.
        """)

        st.markdown("### 👥 Select Patient")
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.selectbox("Patient ID", ["P001", "P002", "P003", "P004", "P005"])
            st.session_state.patient_id = patient_id
        with col2:
            patient_name = st.text_input("Patient Name", "John Doe")
        with col3:
            age = st.number_input("Age", min_value=18, max_value=100, value=45)

        # Load real session data (NEW - modular integration)
        patient_sessions = session_manager.get_patient_sessions(patient_id)
        session_count = session_manager.get_session_count(patient_id)
        avg_accuracy = session_manager.get_avg_accuracy(patient_id)
        avg_confidence = session_manager.get_avg_confidence(patient_id)

        st.markdown("### 📋 Patient Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Patient ID", patient_id)
        with col2:
            st.metric("Sessions Completed", session_count)
        with col3:
            st.metric("Success Rate", f"{avg_accuracy:.1f}%" if avg_accuracy > 0 else "N/A")
        with col4:
            days_in_program = (datetime.now() - pd.to_datetime(patient_sessions["timestamp"].min())).days if not patient_sessions.empty else 0
            st.metric("Days in Program", max(0, days_in_program))

        # Real rehabilitation progress from stored sessions (NEW - modular integration)
        st.markdown("### 📈 Rehabilitation Progress")

        if not patient_sessions.empty:
            patient_sessions_copy = patient_sessions.copy()
            patient_sessions_copy["timestamp"] = pd.to_datetime(patient_sessions_copy["timestamp"])
            patient_sessions_copy["date"] = patient_sessions_copy["timestamp"].dt.date
            daily_data = patient_sessions_copy.groupby("date").agg({
                "accuracy": "mean",
                "session_id": "count"
            }).rename(columns={"session_id": "session_count"}).reset_index()
            daily_data.columns = ["Date", "Accuracy", "Session_Count"]
            daily_data["Date"] = pd.to_datetime(daily_data["Date"])

            if not daily_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_data["Date"], y=daily_data["Accuracy"] * 100,
                    mode="lines+markers", name="Accuracy (%)",
                    line=dict(color="#2dd4bf", width=3), marker=dict(size=8),
                ))
                fig.update_layout(
                    title="Gesture Recognition Accuracy Over Time",
                    xaxis_title="Date", yaxis_title="Accuracy (%)",
                    yaxis_range=[0, 105], hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No session data available yet. Start making predictions to track progress.")
        else:
            st.info("No session data available yet. Start making predictions to track progress.")

        col1, col2 = st.columns(2)
        with col1:
            if not patient_sessions.empty:
                patient_sessions_copy = patient_sessions.copy()
                patient_sessions_copy["timestamp"] = pd.to_datetime(patient_sessions_copy["timestamp"])
                patient_sessions_copy["date"] = patient_sessions_copy["timestamp"].dt.date
                daily_duration = patient_sessions_copy.groupby("date")["session_duration_sec"].sum() / 60

                fig = px.area(
                    x=daily_duration.index, y=daily_duration.values,
                    title="Daily Session Duration (minutes)",
                    color_discrete_sequence=["#06d6a0"],
                    labels={"x": "Date", "y": "Duration (min)"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No session duration data available")
        with col2:
            if not patient_sessions.empty:
                gesture_counts = patient_sessions["predicted_gesture"].value_counts().tail(7)
                fig = px.bar(
                    x=gesture_counts.index, y=gesture_counts.values,
                    title="Gestures Practiced (Last 7 Sessions)",
                    color=gesture_counts.values, color_continuous_scale="Blues",
                    labels={"x": "Gesture", "y": "Count"}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No gesture practice data available")

        st.markdown("### 🎯 Gesture Performance Breakdown")
        gesture_stats = session_manager.get_gesture_stats(patient_id)
        if gesture_stats:
            gesture_perf = []
            for gesture, stats in gesture_stats.items():
                gesture_perf.append({
                    "Gesture": gesture,
                    "Attempts": stats["count"],
                    "Avg_Confidence": stats["avg_confidence"],
                })
            gesture_df = pd.DataFrame(gesture_perf)
            fig = px.bar(
                gesture_df, x="Gesture", y="Avg_Confidence",
                title="Average Confidence by Gesture Type",
                color="Avg_Confidence", color_continuous_scale="RdYlGn",
                text="Avg_Confidence",
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(yaxis_range=[0, 1.05])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No gesture performance data available yet.")

    def rehabilitation_monitor(self):
        st.title("📈 Rehabilitation Monitoring System")

        st.markdown("""
            Advanced monitoring and analytics for tracking patient rehabilitation progress,
            identifying improvement areas, and optimizing therapy sessions.
        """)

        col1, col2 = st.columns(2)
        with col1:
            time_period = st.selectbox(
                "Select Time Period",
                ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"],
            )
        with col2:
            selected_patient = st.selectbox(
                "Select Patient",
                ["P001", "P002", "P003", "P004", "P005"],
                key="rehab_patient_select"
            )

        # Load real session data for selected patient (NEW - modular integration)
        patient_sessions = session_manager.get_patient_sessions(selected_patient)

        # Filter by time period
        if not patient_sessions.empty:
            patient_sessions["timestamp"] = pd.to_datetime(patient_sessions["timestamp"])
            now = pd.Timestamp.now()
            if time_period == "Last 7 Days":
                patient_sessions = patient_sessions[patient_sessions["timestamp"] >= now - pd.Timedelta(days=7)]
            elif time_period == "Last 30 Days":
                patient_sessions = patient_sessions[patient_sessions["timestamp"] >= now - pd.Timedelta(days=30)]
            elif time_period == "Last 3 Months":
                patient_sessions = patient_sessions[patient_sessions["timestamp"] >= now - pd.Timedelta(days=90)]

        st.markdown("### 📊 Weekly Summary")
        col1, col2, col3, col4 = st.columns(4)

        if not patient_sessions.empty:
            patient_sessions["date"] = patient_sessions["timestamp"].dt.date
            week_sessions = patient_sessions[patient_sessions["date"] >= (datetime.now().date() - timedelta(days=7))]
            week_count = len(week_sessions)
            week_accuracy = week_sessions["accuracy"].mean() * 100 if week_sessions["accuracy"].notna().any() else 0
            total_duration = week_sessions["session_duration_sec"].sum() / 60 if week_sessions["session_duration_sec"].notna().any() else 0
            unique_gestures = week_sessions["predicted_gesture"].nunique()

            with col1:
                st.metric("Sessions This Week", week_count)
            with col2:
                st.metric("Avg Accuracy", f"{week_accuracy:.1f}%")
            with col3:
                st.metric("Total Practice Time", f"{int(total_duration)}m")
            with col4:
                st.metric("Gestures Practiced", unique_gestures)
        else:
            with col1:
                st.metric("Sessions This Week", 0)
            with col2:
                st.metric("Avg Accuracy", "N/A")
            with col3:
                st.metric("Total Practice Time", "0m")
            with col4:
                st.metric("Gestures Practiced", 0)

        st.markdown("### 🔥 Activity Heatmap")
        if not patient_sessions.empty:
            patient_sessions["week"] = patient_sessions["timestamp"].dt.isocalendar().week
            patient_sessions["day_of_week"] = patient_sessions["timestamp"].dt.day_name()
            heatmap_pivot = patient_sessions.pivot_table(
                values="session_id", index="week", columns="day_of_week", aggfunc="count", fill_value=0
            )
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_pivot = heatmap_pivot.reindex(columns=[d for d in day_order if d in heatmap_pivot.columns])

            fig = px.imshow(
                heatmap_pivot,
                labels=dict(x="Day of Week", y="Week", color="Session Count"),
                color_continuous_scale="Greens",
                title="Rehabilitation Activity Heatmap",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No session data available for selected time period")

        st.markdown("### 📈 Improvement Trends")
        if not patient_sessions.empty:
            patient_sessions_sorted = patient_sessions.sort_values("timestamp")
            patient_sessions_sorted["cumulative_avg_accuracy"] = patient_sessions_sorted["accuracy"].expanding().mean() * 100
            patient_sessions_sorted["index"] = range(len(patient_sessions_sorted))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=patient_sessions_sorted.index,
                y=patient_sessions_sorted["cumulative_avg_accuracy"],
                name="Cumulative Average Accuracy",
                line=dict(color="#06d6a0", width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title="Performance Improvement Over Sessions",
                xaxis_title="Session Number",
                yaxis_title="Cumulative Accuracy (%)",
                yaxis_range=[0, 105],
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No session data available to show improvement trends")

        st.markdown("### 💡 AI-Powered Recommendations")
        if not patient_sessions.empty:
            avg_acc = patient_sessions["accuracy"].mean() * 100 if patient_sessions["accuracy"].notna().any() else 0

            col1, col2 = st.columns(2)
            with col1:
                if avg_acc > 85:
                    st.success("""
                        **✅ Strengths Identified:**
                        - Excellent overall performance
                        - Strong accuracy trend
                        - Consistent engagement
                        - High confidence levels
                    """)
                else:
                    st.success("""
                        **✅ Progress Areas:**
                        - Consistent practice sessions
                        - Increasing attempt count
                        - Developing familiarity
                        - Building muscle memory
                    """)
            with col2:
                if avg_acc < 75:
                    st.warning("""
                        **⚠️ Areas for Improvement:**
                        - Focus on gesture accuracy
                        - Increase practice frequency
                        - Review challenging gestures
                        - Optimize session duration
                    """)
                else:
                    st.info("""
                        **📌 Next Steps:**
                        - Maintain current practice level
                        - Challenge with complex sequences
                        - Increase session duration
                        - Monitor advanced gestures
                    """)
        else:
            st.info("No session data available yet")

        st.markdown("### 🎯 Recommended Exercises")
        if not patient_sessions.empty:
            gesture_stats = session_manager.get_gesture_stats(selected_patient)
            if gesture_stats:
                low_confidence_gestures = [g for g, s in gesture_stats.items() if s["avg_confidence"] < 0.8]
                exercises = pd.DataFrame({
                    "Exercise": [f"Practice {g}" for g in low_confidence_gestures] if low_confidence_gestures else ["Continue current practice"],
                    "Priority": ["High"] * len(low_confidence_gestures) if low_confidence_gestures else ["Medium"],
                    "Duration": ["15 min"] * len(low_confidence_gestures) if low_confidence_gestures else ["20 min"],
                    "Repetitions": ["30"] * len(low_confidence_gestures) if low_confidence_gestures else ["25"],
                })
                st.dataframe(exercises, use_container_width=True)
        else:
            st.info("Complete prediction sessions to receive personalized exercise recommendations")

    def about_page(self):
        st.title("ℹ️ About the System")

        model_type = self.metadata.get("model_type", "Ensemble ML")
        n_features = self.metadata.get("n_features", 56)
        n_classes = self.metadata.get("n_classes", 8)

        st.markdown(f"""
            ### 🦾 AI-Based Low-Cost Bionic Limb Gesture Recognition System

            This comprehensive healthcare application leverages **Machine Learning** and **EMG signal processing**
            to provide an intelligent, affordable solution for prosthetic limb control and rehabilitation monitoring.

            ---

            #### 🎯 Project Objectives

            - Develop an intelligent gesture recognition system using EMG signals
            - Provide affordable prosthetic control for amputees
            - Enable real-time prediction of hand movements
            - Support rehabilitation monitoring and progress tracking
            - Create an accessible, user-friendly healthcare interface

            ---

            #### 🔬 Technical Approach

            **Data Collection:**
            - 8-channel EMG sensor array
            - Surface electrodes on forearm muscles
            - Real-time signal acquisition

            **Feature Engineering:**
            - {n_features} statistical features extracted per window
            - 7 time-domain statistics per EMG channel (mean, RMS, variance, std, energy, max, min)
            - Sliding-window approach with 90% overlap for dense temporal coverage

            **Machine Learning:**
            - **Model:** {model_type}
            - **Classes:** {n_classes} hand gestures
            - High accuracy (>99%) on engineered features
            - Real-time prediction capability

            ---

            #### 👥 Target Beneficiaries

            - **Amputees:** Improved prosthetic control
            - **Healthcare Providers:** Better patient monitoring
            - **Researchers:** Platform for assistive technology research
            - **Students:** Educational tool for ML in healthcare

            ---

            #### 🛠️ Technology Stack

            - **Frontend:** Streamlit (Python web framework)
            - **ML Framework:** scikit-learn, XGBoost, LightGBM
            - **Data Processing:** Pandas, NumPy
            - **Visualization:** Plotly, Matplotlib, Seaborn
            - **Model Persistence:** Joblib

            ---

            #### 📧 Contact & Support

            For questions, suggestions, or collaboration:
            - Email: support@bioniclimb-ai.com
            - GitHub: github.com/bioniclimb-ai
            - Documentation: docs.bioniclimb-ai.com

            ---

            <div style="text-align:center; padding:2rem; background:#f8f9fa; border-radius:10px;">
                <h3>🌟 Empowering Lives Through Technology 🌟</h3>
                <p>Making advanced prosthetic control accessible to all</p>
            </div>
        """, unsafe_allow_html=True)

    def run(self):
        self.render_header()

        if not self.model_loaded:
            self.load_model()

        page = self.render_sidebar()

        try:
            if page == "🏠 Home":
                self.home_page()
            elif page == "🔬 Real-Time Prediction":
                self.prediction_page()
            elif page == "📊 Model Performance":
                self.performance_page()
            elif page == "👤 Patient Dashboard":
                self.patient_dashboard()
            elif page == "📈 Rehabilitation Monitor":
                self.rehabilitation_monitor()
            elif page == "ℹ️ About System":
                self.about_page()
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
            st.info("Please try again or contact support.")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = BionicLimbApp()
    app.run()
