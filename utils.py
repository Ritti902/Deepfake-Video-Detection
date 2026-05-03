"""
app.py — Deepfake Video Detection System
Inspired by: Electronics Store dark-tech aesthetic (Wix template #3745)
Layout: dark background, cyan/electric accent, 6×6 frame grid preview
"""

import os
import tempfile
import base64
import cv2
import numpy as np
import streamlit as st

from pipeline import get_device, list_available_models, load_model, run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepScan · Deepfake Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Electronics Store dark-tech aesthetic
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Barlow:wght@300;400;500;600&display=swap');

/* ── Reset & base ──────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #080c10 !important;
    color: #c8d8e8 !important;
    font-family: 'Barlow', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #080c10 !important;
    padding: 0 !important;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #00e5ff44; border-radius: 2px; }

/* ── Top nav bar ───────────────────────────────────────────────────────── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 3rem;
    height: 64px;
    background: #0a0f15;
    border-bottom: 1px solid #00e5ff22;
    position: sticky; top: 0; z-index: 100;
}
.nav-logo {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 3px;
    color: #00e5ff;
    text-transform: uppercase;
}
.nav-logo span { color: #ffffff; }
.nav-links {
    display: flex; gap: 2.5rem;
    font-size: .78rem; font-weight: 500;
    letter-spacing: 2px; text-transform: uppercase;
    color: #607080;
}
.nav-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: .7rem;
    padding: .3em .9em;
    border: 1px solid #00e5ff55;
    color: #00e5ff;
    letter-spacing: 2px;
}

/* ── Hero band ─────────────────────────────────────────────────────────── */
.hero {
    position: relative;
    padding: 4.5rem 3rem 3.5rem;
    overflow: hidden;
    background: linear-gradient(135deg, #080c10 60%, #001a22 100%);
    border-bottom: 1px solid #00e5ff18;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        repeating-linear-gradient(0deg,   transparent, transparent 39px, #00e5ff08 40px),
        repeating-linear-gradient(90deg,  transparent, transparent 39px, #00e5ff08 40px);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Share Tech Mono', monospace;
    font-size: .7rem; letter-spacing: 4px;
    color: #00e5ff; text-transform: uppercase;
    margin-bottom: .8rem;
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4.2rem);
    font-weight: 700;
    line-height: 1.05;
    color: #ffffff;
    letter-spacing: -1px;
}
.hero-title em {
    font-style: normal;
    color: #00e5ff;
    text-shadow: 0 0 30px #00e5ff66;
}
.hero-sub {
    margin-top: 1rem;
    max-width: 540px;
    font-size: .95rem; line-height: 1.7;
    color: #607080; font-weight: 300;
}
.hero-stats {
    display: flex; gap: 3rem;
    margin-top: 2.5rem;
    flex-wrap: wrap;
}
.stat-item { display: flex; flex-direction: column; }
.stat-num {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.8rem; font-weight: 700;
    color: #00e5ff; line-height: 1;
}
.stat-lbl {
    font-size: .68rem; letter-spacing: 2px;
    color: #405060; text-transform: uppercase;
    margin-top: .2rem;
}
.hero-corner {
    position: absolute; top: 2rem; right: 3rem;
    width: 200px; height: 200px;
    border: 1px solid #00e5ff15;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
}
.hero-corner::before {
    content: '';
    position: absolute;
    width: 140px; height: 140px;
    border: 1px solid #00e5ff25; border-radius: 50%;
}
.hero-corner::after {
    content: '⚡';
    font-size: 2.5rem;
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: .4; transform: scale(1); }
    50%       { opacity: 1;  transform: scale(1.1); }
}

/* ── Main layout ───────────────────────────────────────────────────────── */
.main-grid {
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 0;
    min-height: calc(100vh - 64px - 240px);
}

/* ── Left panel ────────────────────────────────────────────────────────── */
.left-panel {
    background: #0a0f15;
    border-right: 1px solid #00e5ff15;
    padding: 2rem 1.8rem;
    display: flex; flex-direction: column; gap: 1.4rem;
}

/* ── Section card ──────────────────────────────────────────────────────── */
.card {
    background: #0d1520;
    border: 1px solid #1a2535;
    padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden;
}
.card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00e5ff, #0044ff44);
}
.card-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: .65rem; letter-spacing: 3px;
    color: #00e5ff88; text-transform: uppercase;
    margin-bottom: .5rem;
}
.card-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem; font-weight: 600;
    color: #c8d8e8; letter-spacing: 1px;
}

/* ── Streamlit widget overrides ────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: #111820 !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 0 !important;
    color: #c8d8e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .82rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 1px #00e5ff44 !important;
}
[data-testid="stSelectbox"] svg { fill: #00e5ff !important; }

[data-testid="stFileUploader"] {
    background: #0d1520 !important;
    border: 1px dashed #1e2d3d !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00e5ff55 !important;
}
[data-testid="stFileUploader"] label {
    color: #607080 !important;
    font-size: .8rem !important;
    font-family: 'Share Tech Mono', monospace !important;
}

[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #00e5ff15, #0044ff15) !important;
    border: 1px solid #00e5ff55 !important;
    border-radius: 0 !important;
    color: #00e5ff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: .9rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    padding: .8rem !important;
    text-transform: uppercase !important;
    transition: all .2s !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #00e5ff25, #0044ff25) !important;
    border-color: #00e5ff !important;
    box-shadow: 0 0 20px #00e5ff22 !important;
}

/* ── Right panel ───────────────────────────────────────────────────────── */
.right-panel {
    padding: 2rem 2.5rem;
    display: flex; flex-direction: column; gap: 1.5rem;
}

/* ── Empty state ───────────────────────────────────────────────────────── */
.empty-state {
    flex: 1;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 400px;
    border: 1px dashed #1a2535;
    gap: 1rem;
}
.empty-icon { font-size: 3rem; opacity: .3; }
.empty-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: .75rem; letter-spacing: 3px;
    color: #2a3a4a; text-transform: uppercase;
}

/* ── Frame grid ─────────────────────────────────────────────────────────── */
.grid-header {
    display: flex; align-items: baseline; gap: 1rem;
    margin-bottom: .6rem;
}
.grid-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem; font-weight: 600; color: #c8d8e8;
}
.grid-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: .62rem; color: #405060; letter-spacing: 2px;
}
.frame-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 3px;
    background: #0a0f15;
    padding: 3px;
    border: 1px solid #1a2535;
}
.frame-cell {
    aspect-ratio: 1;
    overflow: hidden;
    position: relative;
}
.frame-cell img {
    width: 100%; height: 100%;
    object-fit: cover; display: block;
}
.frame-idx {
    position: absolute; bottom: 2px; right: 3px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .45rem; color: #00e5ff88;
    background: #00000088;
    padding: 1px 3px;
}

/* ── Result card ───────────────────────────────────────────────────────── */
.result-outer {
    border: 1px solid #1a2535;
    padding: 1.8rem;
    position: relative; overflow: hidden;
}
.result-outer.real { border-color: #00e5ff44; background: #001a2244; }
.result-outer.fake { border-color: #ff003344; background: #1a000044; }

.result-row { display: flex; align-items: center; gap: 2rem; }
.result-verdict {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem; font-weight: 700;
    line-height: 1; white-space: nowrap;
    flex-shrink: 0;
}
.result-verdict.real { color: #00e5ff; text-shadow: 0 0 40px #00e5ff66; }
.result-verdict.fake { color: #ff3355; text-shadow: 0 0 40px #ff335566; }

.result-meta { flex: 1; }
.result-conf {
    font-family: 'Share Tech Mono', monospace;
    font-size: .7rem; color: #607080; letter-spacing: 2px;
    margin-bottom: .4rem;
}
.conf-bar-wrap {
    height: 4px; background: #1a2535;
    margin-bottom: .7rem; overflow: hidden;
}
.conf-bar { height: 100%; transition: width .6s ease; }
.conf-bar.real { background: linear-gradient(90deg, #00e5ff, #0088ff); }
.conf-bar.fake { background: linear-gradient(90deg, #ff3355, #ff6600); }

.prob-chips { display: flex; gap: .6rem; flex-wrap: wrap; }
.prob-chip {
    font-family: 'Share Tech Mono', monospace;
    font-size: .68rem; padding: .25em .7em;
    border: 1px solid; letter-spacing: 1px;
}
.prob-chip.real { color: #00e5ff; border-color: #00e5ff44; }
.prob-chip.fake { color: #ff3355; border-color: #ff335544; }

.info-box {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: .5rem .8rem;
    margin-top: 1.2rem;
    padding-top: 1.2rem;
    border-top: 1px solid #1a2535;
}
.info-item { display: flex; flex-direction: column; gap: .15rem; }
.info-key {
    font-family: 'Share Tech Mono', monospace;
    font-size: .58rem; color: #405060; letter-spacing: 2px;
    text-transform: uppercase;
}
.info-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: .88rem; font-weight: 600; color: #c8d8e8;
}

/* ── Invert notice ─────────────────────────────────────────────────────── */
.invert-notice {
    font-family: 'Share Tech Mono', monospace;
    font-size: .68rem; letter-spacing: 2px;
    color: #ff9900;
    border: 1px solid #ff990033;
    background: #ff990011;
    padding: .5rem 1rem;
    margin-bottom: .5rem;
}

/* ── Footer ────────────────────────────────────────────────────────────── */
.footer {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1rem 3rem;
    border-top: 1px solid #0d1520;
    background: #0a0f15;
    margin-top: auto;
}
.footer-left {
    font-family: 'Share Tech Mono', monospace;
    font-size: .6rem; color: #253545; letter-spacing: 2px;
}
.footer-right {
    display: flex; gap: 2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: .6rem; color: #253545; letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_frame_grid_html(frames: list, grid_size: int = 36) -> str:
    step = max(1, len(frames) // grid_size)
    selected = frames[::step][:grid_size]
    cells = ""
    for i, frm in enumerate(selected):
        _, buf = cv2.imencode(".jpg", frm, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf).decode()
        cells += (
            f'<div class="frame-cell">'
            f'<img src="data:image/jpeg;base64,{b64}" loading="lazy"/>'
            f'<div class="frame-idx">{i+1:02d}</div>'
            f'</div>'
        )
    return f'<div class="frame-grid">{cells}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Device & model discovery
# ─────────────────────────────────────────────────────────────────────────────

device       = get_device()
device_label = "CUDA GPU" if device.type == "cuda" else "CPU"
model_files  = list_available_models()

# ─────────────────────────────────────────────────────────────────────────────
# NAV BAR
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">DEEP<span>SCAN</span></div>
    <div class="nav-links">
        <span>Detection</span>
        <span>Analytics</span>
        <span>Documentation</span>
    </div>
    <div class="nav-badge">v2.0 · FORENSICS</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">// Neural Forensics Engine</div>
    <div class="hero-title">Detect <em>Deepfakes</em><br>With Precision</div>
    <div class="hero-sub">
        ResNeXt50 + LSTM architecture trained on FaceForensics++.
        Frame-level face extraction, temporal modelling, softmax classification.
    </div>
    <div class="hero-stats">
        <div class="stat-item">
            <div class="stat-num">150</div>
            <div class="stat-lbl">Frames Analyzed</div>
        </div>
        <div class="stat-item">
            <div class="stat-num">112px</div>
            <div class="stat-lbl">Face Resolution</div>
        </div>
        <div class="stat-item">
            <div class="stat-num">2048</div>
            <div class="stat-lbl">LSTM Hidden Units</div>
        </div>
        <div class="stat-item">
            <div class="stat-num">Real/Fake</div>
            <div class="stat-lbl">Binary Classification</div>
        </div>
    </div>
    <div class="hero-corner"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN GRID
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-grid">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# LEFT PANEL
# ════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="left-panel">', unsafe_allow_html=True)

# ── Step 01: Model selector ───────────────────────────────────────────────────
st.markdown("""
<div class="card">
    <div class="card-label">// Step 01</div>
    <div class="card-title">Select Detection Model</div>
</div>
""", unsafe_allow_html=True)

if not model_files:
    st.error("❌ No model files found. Place `.pt` files in the `models/` directory.")
    st.stop()

model_names   = [os.path.basename(p) for p in model_files]
selected_name = st.selectbox("", model_names, label_visibility="collapsed")
selected_path = model_files[model_names.index(selected_name)]

# ── Detect if this is the first model (model_84_acc_10_frames_final_data.pt)
# This model has inverted labels — REAL output means FAKE, FAKE output means REAL.
# We flip the result at display time to correct it.
is_first_model = (selected_name == "model_84_acc_10_frames_final_data.pt")

st.markdown(f"""
<div class="info-box" style="margin-top:.6rem;">
    <div class="info-item">
        <span class="info-key">Architecture</span>
        <span class="info-val">ResNeXt50_32x4d</span>
    </div>
    <div class="info-item">
        <span class="info-key">LSTM Hidden</span>
        <span class="info-val">2048 units</span>
    </div>
    <div class="info-item">
        <span class="info-key">Dropout</span>
        <span class="info-val">0.4</span>
    </div>
    <div class="info-item">
        <span class="info-key">Label Mode</span>
        <span class="info-val" style="color:{'#ff9900' if is_first_model else '#00e5ff'};">
            {'⚠ INVERTED' if is_first_model else '✓ NORMAL'}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Step 02: Upload ───────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
    <div class="card-label">// Step 02</div>
    <div class="card-title">Upload Video File</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["mp4", "avi", "mov"], label_visibility="collapsed")

# ── Step 03: Analyze button ───────────────────────────────────────────────────
st.markdown('<div style="margin-top:.4rem;">', unsafe_allow_html=True)
run_btn = st.button("⚡  ANALYZE VIDEO", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# ── Pipeline legend ───────────────────────────────────────────────────────────
steps = [
    ("01", "Extract 150 frames (uniform)"),
    ("02", "dlib frontal face detection"),
    ("03", "Crop → 112×112 → normalize"),
    ("04", "ResNeXt50 feature extraction"),
    ("05", "LSTM temporal modelling"),
    ("06", "Softmax → REAL / FAKE + score"),
]
steps_html = "".join(
    f'<div style="display:flex;align-items:center;gap:.6rem;font-size:.7rem;'
    f'color:#607080;font-family:\'Share Tech Mono\',monospace;">'
    f'<span style="color:#00e5ff;">{n}</span>{lbl}</div>'
    for n, lbl in steps
)
st.markdown(f"""
<div class="card">
    <div class="card-label">// Inference Pipeline</div>
    <div style="display:flex;flex-direction:column;gap:.5rem;margin-top:.4rem;">
        {steps_html}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /left-panel

# ════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL
# ════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="right-panel">', unsafe_allow_html=True)

if not uploaded:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🎬</div>
        <div class="empty-text">Upload a video to begin analysis</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:.6rem;color:#1e2d3d;letter-spacing:2px;margin-top:.3rem;">
            SUPPORTED FORMATS · MP4 &nbsp;·&nbsp; AVI &nbsp;·&nbsp; MOV
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Save to temp
    suffix = os.path.splitext(uploaded.name)[-1]
    os.makedirs("temp", exist_ok=True)
    with tempfile.NamedTemporaryFile(dir="temp", suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # ── Compact video preview ──────────────────────────────────────────────
    with st.expander("▶  VIDEO PREVIEW", expanded=False):
        st.video(tmp_path)

    # ── 6×6 Frame grid ────────────────────────────────────────────────────
    try:
        with st.spinner("Sampling frames for grid …"):
            # Inline frame extraction — avoids re-importing utils
            _cap   = cv2.VideoCapture(tmp_path)
            _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if _total < 150:
                _cap.release()
                raise ValueError(f"Video only has {_total} frames — need at least 150.")
            _idx_list = np.linspace(0, _total - 1, 150, dtype=int)
            preview_frames = []
            for _i in _idx_list:
                _cap.set(cv2.CAP_PROP_POS_FRAMES, int(_i))
                _ret, _frm = _cap.read()
                if _ret:
                    preview_frames.append(_frm)
            _cap.release()

        st.markdown("""
        <div class="grid-header">
            <div class="grid-title">Frame Sample Grid</div>
            <div class="grid-meta">6 × 6 &nbsp;·&nbsp; 36 UNIFORM SAMPLES FROM 150 FRAMES</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(build_frame_grid_html(preview_frames), unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"❌ {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        st.stop()

    # ── Run analysis on button press ───────────────────────────────────────
    if run_btn:
        if is_first_model:
            st.markdown(f"""
            <div class="invert-notice">
                ⚠ &nbsp; LABEL INVERSION ACTIVE — {selected_name} predicts opposite class
            </div>
            """, unsafe_allow_html=True)

        result   = None
        err_msg  = None

        try:
            with st.spinner("Loading model weights …"):
                model = load_model(selected_path, device)

            with st.spinner("Running deep learning inference …"):
                result = run_pipeline(
                    tmp_path, model, device,
                    model_name=selected_name,
                    model_names=model_files,
                )
        except ValueError as e:
            err_msg = f"Processing Error: {e}"
        except RuntimeError as e:
            err_msg = f"Model Error: {e}"
        except Exception as e:
            err_msg = f"Unexpected Error: {e}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if err_msg:
            st.error(f"❌ {err_msg}")
        elif result:
            label      = result["label"]
            confidence = result["confidence"]
            real_pct   = result["probs"][0]
            fake_pct   = result["probs"][1]

            # ── ✅ FIX: Flip label and probabilities for model_84_acc_10_frames_final_data.pt
            # This model was trained with inverted labels — its REAL output is actually FAKE
            # and its FAKE output is actually REAL. We correct this here at display time only.
            # All other models are unaffected.
            if is_first_model:
                label      = "FAKE" if label.upper() == "REAL" else "REAL"
                real_pct   = result["probs"][1]   # swap: old fake% becomes new real%
                fake_pct   = result["probs"][0]   # swap: old real% becomes new fake%
                # confidence stays the same — it's just the max probability, direction doesn't matter

            lc = label.lower()

            st.markdown(f"""
            <div class="result-outer {lc}">
                <div class="result-row">
                    <div class="result-verdict {lc}">
                        {'✓ &nbsp;REAL' if lc == 'real' else '✗ &nbsp;FAKE'}
                    </div>
                    <div class="result-meta">
                        <div class="result-conf">CONFIDENCE SCORE &nbsp;·&nbsp; {confidence}%</div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar {lc}" style="width:{confidence}%;"></div>
                        </div>
                        <div class="prob-chips">
                            <span class="prob-chip real">✓ REAL &nbsp; {real_pct}%</span>
                            <span class="prob-chip fake">✗ FAKE &nbsp; {fake_pct}%</span>
                        </div>
                    </div>
                </div>
                <div class="info-box">
                    <div class="info-item">
                        <span class="info-key">Model</span>
                        <span class="info-val">{selected_name}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-key">Device</span>
                        <span class="info-val">{device_label}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-key">Frames</span>
                        <span class="info-val">150 processed</span>
                    </div>
                    <div class="info-item">
                        <span class="info-key">Face Size</span>
                        <span class="info-val">112 × 112 px</span>
                    </div>
                    <div class="info-item">
                        <span class="info-key">Input Tensor</span>
                        <span class="info-val">(1,150,3,112,112)</span>
                    </div>
                    <div class="info-item">
                        <span class="info-key">Label Mode</span>
                        <span class="info-val" style="color:{'#ff9900' if is_first_model else '#00e5ff'}">
                            {'INVERTED' if is_first_model else 'NORMAL'}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # No analysis yet — clean up temp
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

st.markdown('</div>', unsafe_allow_html=True)  # /right-panel
st.markdown('</div>', unsafe_allow_html=True)  # /main-grid

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <div class="footer-left">© DEEPSCAN · DEEPFAKE FORENSICS ENGINE · v2.0</div>
    <div class="footer-right">
        <span>ResNeXt50_32x4d</span>
        <span>LSTM · 2048</span>
        <span>dlib Face Detector</span>
        <span>PyTorch</span>
        <span>Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)