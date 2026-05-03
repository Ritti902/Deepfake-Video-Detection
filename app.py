"""
app.py — DeepScan · Deepfake Detection System
Dark-tech aesthetic. All widgets use native Streamlit layout.
6×6 frame grid rendered as inline HTML after upload.
"""

import os
import tempfile
import base64
import cv2
import numpy as np
import streamlit as st

from pipeline import get_device, list_available_models, load_model, run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepScan · Deepfake Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Fonts + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&family=Barlow:wght@300;400;500&display=swap');

/* ── Global ── */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
    background: #07090d !important;
    font-family: 'Barlow', sans-serif !important;
    color: #b8ccd8 !important;
}
[data-testid="stAppViewContainer"] > .main { background: #07090d !important; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1300px !important; }
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #00e5ff30; }

/* ── Typography helpers ── */
.mono { font-family: 'Share Tech Mono', monospace !important; }
.rj   { font-family: 'Rajdhani', sans-serif !important; }

/* ── Top logo bar ── */
.logo-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 1.4rem;
    border-bottom: 1px solid #ffffff0d;
    margin-bottom: 2rem;
}
.logo-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem; font-weight: 700;
    letter-spacing: 3px; color: #fff;
    text-transform: uppercase;
}
.logo-text span { color: #00e5ff; }
.logo-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: .68rem; letter-spacing: 2px;
    color: #00e5ff; padding: .25em .8em;
    border: 1px solid #00e5ff40;
}

/* ── Section label ── */
.sec-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: .62rem; letter-spacing: 3px;
    color: #00e5ff80; text-transform: uppercase;
    margin-bottom: .5rem;
}
.sec-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem; font-weight: 600;
    color: #c8dde8; letter-spacing: 1px;
    margin-bottom: 1rem;
}

/* ── Panel card ── */
.panel {
    background: #0c1018;
    border: 1px solid #ffffff0a;
    border-left: 2px solid #00e5ff30;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] label { display: none !important; }
[data-testid="stSelectbox"] > div > div {
    background: #10161e !important;
    border: 1px solid #1e2d3d !important;
    border-radius: 0 !important;
    color: #c8dde8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .82rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 1px #00e5ff30 !important;
}
[data-testid="stSelectbox"] svg { fill: #00e5ff !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploaderDropzone"] {
    background: #10161e !important;
    border: 1px dashed #00e5ff25 !important;
    border-radius: 0 !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #00e5ff70 !important;
    background: #0d1520 !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #506070 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .75rem !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid #00e5ff !important;
    color: #00e5ff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: .95rem !important; font-weight: 700 !important;
    letter-spacing: 4px !important; text-transform: uppercase !important;
    padding: .8rem !important; border-radius: 0 !important;
    transition: all .2s !important;
}
[data-testid="stButton"] > button:hover {
    background: #00e5ff12 !important;
    box-shadow: 0 0 18px #00e5ff25 !important;
}

/* ── Video player ── */
[data-testid="stVideo"] video {
    border: 1px solid #ffffff0a !important;
    border-radius: 0 !important;
    max-height: 220px !important;
}

/* ── Frame grid ── */
.fgrid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 3px;
    background: #07090d;
    border: 1px solid #ffffff0a;
    padding: 3px;
}
.fgrid-cell {
    aspect-ratio: 1;
    overflow: hidden;
    position: relative;
    background: #0c1018;
}
.fgrid-cell img {
    width: 100%; height: 100%;
    object-fit: cover; display: block;
    filter: brightness(.85) saturate(1.1);
    transition: filter .2s;
}
.fgrid-cell:hover img { filter: brightness(1.1) saturate(1.4); }
.fgrid-cell .fidx {
    position: absolute; bottom: 2px; right: 3px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .44rem; color: #00e5ff90;
    text-shadow: 0 0 4px #000;
}
.grid-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: .6rem;
}
.grid-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: .95rem; font-weight: 600;
    color: #c8dde8; letter-spacing: 2px; text-transform: uppercase;
}
.grid-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: .62rem; color: #405060;
}

/* ── Result block ── */
.result-wrap {
    border: 1px solid #ffffff0a;
    background: #0c1018;
    padding: 1.6rem 2rem;
    margin-top: 1.4rem;
    position: relative;
}
.result-wrap.real { border-top: 2px solid #00e5ff; }
.result-wrap.fake { border-top: 2px solid #ff3060; }

.verdict {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem; font-weight: 700;
    letter-spacing: 4px; line-height: 1;
}
.verdict.real { color: #00e5ff; text-shadow: 0 0 30px #00e5ff50; }
.verdict.fake { color: #ff3060; text-shadow: 0 0 30px #ff306050; }

.conf-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: .7rem; color: #506070;
    margin: .8rem 0 .35rem; letter-spacing: 1px;
}
.bar-track {
    height: 4px; background: #10161e;
    border: 1px solid #1e2d3d; margin-bottom: .9rem;
}
.bar-fill { height: 100%; }
.bar-fill.real { background: linear-gradient(90deg,#00e5ff,#00ff88); }
.bar-fill.fake { background: linear-gradient(90deg,#ff3060,#ff8800); }

.chips { display: flex; gap: .8rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.chip {
    font-family: 'Share Tech Mono', monospace;
    font-size: .7rem; padding: .28em .75em;
    border: 1px solid; letter-spacing: 1px;
}
.chip.real { border-color: #00e5ff35; color: #00e5ff; background: #00e5ff08; }
.chip.fake { border-color: #ff306035; color: #ff3060; background: #ff306008; }

.meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px,1fr));
    gap: .5rem;
    padding: 1rem;
    background: #07090d;
    border: 1px solid #ffffff07;
}
.meta-item { display: flex; flex-direction: column; gap: .15rem; }
.meta-key {
    font-family: 'Share Tech Mono', monospace;
    font-size: .55rem; color: #304050;
    letter-spacing: 2px; text-transform: uppercase;
}
.meta-val {
    font-family: 'Share Tech Mono', monospace;
    font-size: .72rem; color: #00e5ff;
}

/* ── Invert notice ── */
.inv-notice {
    font-family: 'Share Tech Mono', monospace;
    font-size: .68rem; color: #ff990090;
    background: #110d00; border: 1px solid #ff990025;
    padding: .5rem .9rem; margin-bottom: 1rem; letter-spacing: 1px;
}

/* ── Pipeline steps ── */
.pipe-step {
    display: flex; align-items: center; gap: .6rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: .68rem; color: #506070;
    padding: .3rem 0;
    border-bottom: 1px solid #ffffff05;
}
.pipe-step:last-child { border-bottom: none; }
.pipe-num { color: #00e5ff; min-width: 1.2rem; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: #00e5ff !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: #0c1018 !important; border-radius: 0 !important;
    font-family: 'Share Tech Mono', monospace !important; font-size: .78rem !important;
    border-left-color: #ff3060 !important;
}
[data-testid="stAlert"][data-baseweb="notification"] {
    border-left-color: #00e5ff !important;
}

/* ── Divider ── */
hr { border-color: #ffffff0a !important; margin: 1.5rem 0 !important; }

/* ── Responsive ── */
@media (max-width: 768px) {
    .block-container { padding: 1.2rem !important; }
    .verdict { font-size: 2rem; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def frames_from_video(path: str, n: int = 150):
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n:
        cap.release()
        raise ValueError(f"Video has only {total} frames — need at least {n}.")
    idxs   = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frm = cap.read()
        if ret:
            frames.append(frm)
    cap.release()
    return frames


def frame_b64(bgr, size=160):
    h, w   = bgr.shape[:2]
    scale  = size / max(h, w)
    small  = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode()


def render_grid(frames):
    idxs  = np.linspace(0, len(frames) - 1, 36, dtype=int)
    cells = ""
    for pos, i in enumerate(idxs):
        b64    = frame_b64(frames[int(i)])
        cells += (f'<div class="fgrid-cell">'
                  f'<img src="data:image/jpeg;base64,{b64}"/>'
                  f'<span class="fidx">F{int(i):03d}</span>'
                  f'</div>')
    return f'<div class="fgrid">{cells}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

device      = get_device()
is_gpu      = str(device) == "cuda"
device_lbl  = "CUDA · GPU" if is_gpu else "CPU"
model_files = list_available_models("models")

# ─────────────────────────────────────────────────────────────────────────────
# Logo bar
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="logo-bar">
    <div class="logo-text">Deep<span>Scan</span></div>
    <div class="logo-badge">{'⚡ ' if is_gpu else ''}{device_lbl}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Guard: no models
# ─────────────────────────────────────────────────────────────────────────────

if not model_files:
    st.error("No `.pt` files found in `models/` — run `python download_models.py` first.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Layout: left controls | right workspace
# ─────────────────────────────────────────────────────────────────────────────

left, right = st.columns([1, 2.4], gap="large")

# ══════════════════════════════════════════════════════════════════
# LEFT — controls
# ══════════════════════════════════════════════════════════════════
with left:

    # ── Model select ─────────────────────────────────────────────
    st.markdown('<div class="sec-label">// 01 · Model</div>', unsafe_allow_html=True)
    selected = st.selectbox("model", model_files, label_visibility="collapsed")
    is_first = (selected == "model_84_acc_10_frames_final_data.pt")

    st.markdown(f"""
    <div class="panel" style="margin-top:.6rem;">
        <div style="display:flex;flex-direction:column;gap:.45rem;">
            <div class="meta-item">
                <span class="meta-key">File</span>
                <span class="meta-val" style="font-size:.65rem;word-break:break-all;">{selected}</span>
            </div>
            <div class="meta-item">
                <span class="meta-key">Backbone</span>
                <span class="meta-val">ResNeXt50_32x4d</span>
            </div>
            <div class="meta-item">
                <span class="meta-key">LSTM</span>
                <span class="meta-val">2048 hidden units</span>
            </div>
            <div class="meta-item">
                <span class="meta-key">Labels</span>
                <span class="meta-val" style="color:#00e5ff;">
                    ✓ NORMAL
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:.6rem;"></div>', unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────
    st.markdown('<div class="sec-label">// 02 · Upload Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("video", type=["mp4", "avi", "mov"],
                                label_visibility="collapsed")

    st.markdown('<div style="height:.6rem;"></div>', unsafe_allow_html=True)

    # ── Analyze button ───────────────────────────────────────────
    run_btn = st.button("⚡  ANALYZE VIDEO")

    # ── Pipeline legend ──────────────────────────────────────────
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    steps = [
        ("01", "Extract 150 frames"),
        ("02", "dlib face detection"),
        ("03", "Crop · resize 112²"),
        ("04", "Normalize (ImageNet)"),
        ("05", "ResNeXt50 features"),
        ("06", "LSTM · Softmax"),
    ]
    steps_html = "".join(
        f'<div class="pipe-step"><span class="pipe-num">{n}</span>{t}</div>'
        for n, t in steps
    )
    st.markdown(f"""
    <div class="panel">
        <div class="sec-label" style="margin-bottom:.7rem;">// Pipeline</div>
        {steps_html}
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# RIGHT — workspace
# ══════════════════════════════════════════════════════════════════
with right:

    if not uploaded:
        # ── Empty state ───────────────────────────────────────────
        st.markdown("""
        <div style="
            display:flex; flex-direction:column;
            align-items:center; justify-content:center;
            min-height:460px; gap:1rem;
        ">
            <div style="font-size:3rem;opacity:.15;">🎬</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:.72rem;
                        letter-spacing:3px;color:#1e2d3d;text-transform:uppercase;">
                Upload a video to begin
            </div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:.58rem;
                        color:#182028;letter-spacing:2px;">
                MP4 &nbsp;·&nbsp; AVI &nbsp;·&nbsp; MOV
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Save temp file ────────────────────────────────────────
        os.makedirs("temp", exist_ok=True)
        suffix = os.path.splitext(uploaded.name)[-1]
        tmp = tempfile.NamedTemporaryFile(dir="temp", suffix=suffix, delete=False)
        tmp.write(uploaded.read())
        tmp.close()
        tmp_path = tmp.name

        # ── Video preview (collapsed) ─────────────────────────────
        with st.expander("▶  VIDEO PREVIEW", expanded=False):
            st.video(tmp_path)

        # ── 6×6 Frame grid ────────────────────────────────────────
        try:
            with st.spinner("Sampling frames…"):
                frames = frames_from_video(tmp_path, 150)

            st.markdown("""
            <div class="grid-header">
                <div class="grid-title">Frame Grid</div>
                <div class="grid-meta">6 × 6 &nbsp;·&nbsp; 36 of 150 frames</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(render_grid(frames), unsafe_allow_html=True)

        except ValueError as e:
            st.error(str(e))
            os.remove(tmp_path)
            st.stop()

        # ── Run inference ─────────────────────────────────────────
        if run_btn:
            result  = None
            err_msg = None

            try:
                with st.spinner("Loading model…"):
                    model = load_model(os.path.join("models", selected), device)
                with st.spinner("Running inference…"):
                    result = run_pipeline(
                        tmp_path, model, device,
                        invert=is_first,
                    )
            except Exception as e:
                err_msg = str(e)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

            if err_msg:
                st.error(f"❌ {err_msg}")

            elif result:
                lbl       = result["label"].lower()
                conf      = result["confidence"]
                rp        = result["probs"][0]
                fp        = result["probs"][1]

                # For model_84: flip the display label and swap probabilities
                if is_first:
                    lbl       = "fake" if lbl == "real" else "real"
                    rp, fp    = fp, rp

                icon      = "&#10003;" if lbl == "real" else "&#10007;"
                lbl_color = "#00e5ff"
                lbl_mode  = "NORMAL"

                html = (
                    f'<div class="result-wrap {lbl}">'
                    f'<div class="verdict {lbl}">{icon}&nbsp;{lbl.upper()}</div>'
                    f'<div class="conf-label">CONFIDENCE &nbsp; {conf}%</div>'
                    f'<div class="bar-track">'
                    f'<div class="bar-fill {lbl}" style="width:{conf}%;"></div>'
                    f'</div>'
                    f'<div class="chips">'
                    f'<span class="chip real">REAL &nbsp; {rp}%</span>'
                    f'<span class="chip fake">FAKE &nbsp; {fp}%</span>'
                    f'</div>'
                    f'<div class="meta-grid">'
                    f'<div class="meta-item"><span class="meta-key">Model</span>'
                    f'<span class="meta-val" style="font-size:.62rem;word-break:break-all;">{selected}</span></div>'
                    f'<div class="meta-item"><span class="meta-key">Device</span>'
                    f'<span class="meta-val">{device_lbl}</span></div>'
                    f'<div class="meta-item"><span class="meta-key">Frames</span>'
                    f'<span class="meta-val">150</span></div>'
                    f'<div class="meta-item"><span class="meta-key">Face size</span>'
                    f'<span class="meta-val">112 x 112</span></div>'
                    f'<div class="meta-item"><span class="meta-key">Tensor</span>'
                    f'<span class="meta-val">(1,150,3,112,112)</span></div>'
                    f'<div class="meta-item"><span class="meta-key">Labels</span>'
                    f'<span class="meta-val" style="color:{lbl_color};">{lbl_mode}</span></div>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(html, unsafe_allow_html=True)

        else:
            # Not yet run — clean up temp
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:.5rem;
     font-family:'Share Tech Mono',monospace;font-size:.6rem;color:#253545;letter-spacing:1px;">
    <span>© DEEPSCAN · v2.0</span>
    <span>ResNeXt50_32x4d &nbsp;·&nbsp; LSTM:2048 &nbsp;·&nbsp; dlib &nbsp;·&nbsp; PyTorch · Streamlit</span>
</div>
""", unsafe_allow_html=True)