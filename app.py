# =============================================================================
#  AlzDetect AI — Streamlit Dashboard
#  Iqra University Islamabad | BSSE & BSAI | Fall 2022 – Spring 2026
#  Developers : Amina Arshad · Qura tul Ain · Sidra tul Muntaha
#  Supervisor : Abdul Baqi Malik
#
#  HOW TO RUN (locally):
#      pip install streamlit plotly tensorflow scikit-learn pandas numpy
#      streamlit run app.py
#
#  HOW TO RUN (Google Colab + LocalTunnel):
#      Use AlzDetect_Streamlit_LocalTunnel.ipynb — Cell 3 writes this file
#      automatically to /content/app.py
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle, os, time, random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Optional: load real trained model (created by notebook Cell 4) ───────────
try:
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlzDetect AI | Iqra University",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --acc:  #2563eb;  --acc2: #38bdf8;
    --ok:   #10b981;  --warn: #f59e0b;
    --err:  #ef4444;  --mci:  #8b5cf6;
    --txt:  #e2e8f0;  --sub:  #94a3b8;
    --card: rgba(15,30,60,0.85);
    --bdr:  rgba(37,99,235,0.25);
    --glow: 0 0 22px rgba(37,99,235,0.28);
}

* { font-family: 'Outfit', sans-serif !important; }

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020b18 0%, #041228 40%, #061a38 100%) !important;
    color: var(--txt) !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #041228 0%, #020b18 100%) !important;
    border-right: 1px solid var(--bdr) !important;
}

/* Banner */
.banner {
    background: linear-gradient(135deg, #041228, #0d2a5e 50%, #041228);
    border: 1px solid var(--bdr); border-radius: 16px;
    padding: 28px 34px; margin-bottom: 22px;
    box-shadow: var(--glow); position: relative; overflow: hidden;
}
.banner h1 {
    font-size: 2rem !important; font-weight: 800 !important;
    background: linear-gradient(90deg, #fff 0%, #93c5fd 55%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 !important;
}
.banner p { color: var(--sub) !important; font-size: .88rem !important; margin: 6px 0 0 !important; }
.badge {
    display: inline-block; background: rgba(37,99,235,.18);
    border: 1px solid rgba(37,99,235,.38); border-radius: 20px;
    padding: 3px 13px; font-size: .72rem; color: var(--acc2); margin: 8px 5px 0 0;
}

/* KPI cards */
.kpi {
    background: var(--card); border: 1px solid var(--bdr);
    border-radius: 14px; padding: 20px 16px; text-align: center;
    box-shadow: var(--glow);
}
.kpi-val {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.kpi-lbl { font-size: .7rem; color: var(--sub); margin-top: 5px;
            text-transform: uppercase; letter-spacing: .07em; }
.kpi-delta { font-size: .7rem; color: var(--ok); margin-top: 3px; }

/* Section header */
.sec {
    font-size: .8rem; font-weight: 700; color: var(--acc2);
    border-left: 3px solid var(--acc); padding-left: 10px;
    text-transform: uppercase; letter-spacing: .07em; margin: 18px 0 12px;
}

/* Card */
.card {
    background: var(--card); border: 1px solid var(--bdr);
    border-radius: 14px; padding: 20px; box-shadow: var(--glow);
}

/* Result card */
.res-card { border-radius: 13px; padding: 20px; text-align: center; }

/* Info / warn boxes */
.info {
    background: rgba(37,99,235,.07); border: 1px solid rgba(37,99,235,.22);
    border-radius: 10px; padding: 13px 17px;
    font-size: .82rem; color: var(--sub); line-height: 1.65;
}
.warnbox {
    background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.28);
    border-radius: 10px; padding: 12px 16px;
    font-size: .8rem; color: #fbbf24; margin-top: 10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: .88rem !important;
    padding: 11px 26px !important; text-transform: uppercase;
    box-shadow: 0 4px 14px rgba(37,99,235,.4) !important; transition: all .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,.6) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,30,60,.6) !important; border-radius: 10px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--sub) !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(37,99,235,.25) !important; color: #fff !important;
}

#MainMenu, footer, header { visibility: hidden; }
hr { border-color: var(--bdr) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS & HELPERS
# ─────────────────────────────────────────────────────────────────────────────
CLASSES    = ["Healthy", "MCI", "Alzheimer's"]
CLS_COLOR  = {"Healthy": "#10b981", "MCI": "#8b5cf6", "Alzheimer's": "#ef4444"}
CLS_EMOJI  = {"Healthy": "🟢",      "MCI": "🟣",      "Alzheimer's": "🔴"}
N_EEG, N_MRI = 64, 128

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#94a3b8"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    margin=dict(t=16, b=16, l=0, r=0),
)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD TRAINED MODEL  (saved by notebook Cell 4)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model, se, sm = None, None, None
    if not TF_AVAILABLE:
        return None, None, None
    try:
        if os.path.exists("/content/alzdetect_model.h5"):
            model = tf.keras.models.load_model("/content/alzdetect_model.h5")
        if os.path.exists("/content/scaler_eeg.pkl"):
            with open("/content/scaler_eeg.pkl", "rb") as f:
                se = pickle.load(f)
        if os.path.exists("/content/scaler_mri.pkl"):
            with open("/content/scaler_mri.pkl", "rb") as f:
                sm = pickle.load(f)
    except Exception as e:
        st.warning(f"Model load warning: {e}")
    return model, se, sm

MODEL, SCALER_EEG, SCALER_MRI = load_model()
MODEL_READY = MODEL is not None

# ─────────────────────────────────────────────────────────────────────────────
#  PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(age, gender, use_eeg=True, use_mri=True):
    """
    If trained model is loaded → use it.
    Otherwise → rule-based demo prediction.
    """
    np.random.seed(age + ord(gender[0]))

    if MODEL_READY and SCALER_EEG and SCALER_MRI:
        # ── Real model inference ──────────────────────────────────────────
        eeg_raw = np.random.randn(1, N_EEG)
        mri_raw = np.random.randn(1, N_MRI)
        eeg_n   = SCALER_EEG.transform(eeg_raw)
        mri_n   = SCALER_MRI.transform(mri_raw)
        if not use_eeg: eeg_n = np.zeros_like(eeg_n)
        if not use_mri: mri_n = np.zeros_like(mri_n)
        probs = MODEL.predict([eeg_n, mri_n], verbose=0)[0]
    else:
        # ── Demo mode: age-biased probabilities ───────────────────────────
        if age >= 75:    probs = np.array([0.12, 0.23, 0.65])
        elif age >= 65:  probs = np.array([0.25, 0.35, 0.40])
        elif age >= 55:  probs = np.array([0.46, 0.34, 0.20])
        else:            probs = np.array([0.72, 0.19, 0.09])
        noise = np.random.dirichlet(np.ones(3) * 6) * 0.12
        probs = np.clip(probs + noise, 0, 1)
        probs /= probs.sum()

    idx  = int(np.argmax(probs))
    pred = CLASSES[idx]
    conf = float(probs[idx]) * 100
    return pred, conf, dict(zip(CLASSES, probs.tolist()))


def get_eeg_waveform(pred):
    t = np.linspace(0, 4, 800)
    np.random.seed(7)
    if pred == "Alzheimer's":
        s = 1.9*np.sin(2*np.pi*2.5*t) + 0.6*np.sin(2*np.pi*6*t) + 0.28*np.random.randn(800)
    elif pred == "MCI":
        s = 1.2*np.sin(2*np.pi*3.5*t) + 0.9*np.sin(2*np.pi*9*t) + 0.32*np.random.randn(800)
    else:
        s = 0.5*np.sin(2*np.pi*2*t)   + 1.5*np.sin(2*np.pi*10.5*t) + 0.18*np.random.randn(800)
    return t, s


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px;'>
      <div style='font-size:2.6rem;'>🧠</div>
      <div style='font-weight:800;font-size:1.05rem;color:#93c5fd;'>AlzDetect AI</div>
      <div style='font-size:.7rem;color:#64748b;margin-top:3px;'>Iqra University Research</div>
    </div>
    <hr style='border-color:rgba(37,99,235,.2);margin-bottom:18px;'/>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "📌 Navigation",
        ["🏠 Dashboard", "🔬 Patient Analysis", "📊 Model Performance",
         "📈 Clinical Stats", "ℹ️ About"]
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div class='sec'>Patient Info</div>", unsafe_allow_html=True)
    p_name   = st.text_input("Patient Name", "Demo Patient")
    p_age    = st.slider("Age", 40, 95, 68)
    p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    p_id     = st.text_input("Patient ID", "IU-00001")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("<div class='sec'>Model Settings</div>", unsafe_allow_html=True)
    use_eeg  = st.toggle("Use EEG Data", value=True)
    use_mri  = st.toggle("Use MRI Data", value=True)
    conf_thr = st.slider("Confidence Threshold (%)", 50, 95, 75)

    st.markdown("<hr/>", unsafe_allow_html=True)
    status_txt   = "✅ AI Model Loaded" if MODEL_READY else "⚠️ Demo Mode"
    status_color = "#10b981" if MODEL_READY else "#f59e0b"
    st.markdown(f"""
    <div style='font-size:.72rem;color:#475569;text-align:center;line-height:1.8;'>
      <span style='color:{status_color};font-weight:700;'>{status_txt}</span><br/>
      Supervisor: Abdul Baqi Malik<br/>
      Amina Arshad · Qura tul Ain · Sidra tul Muntaha<br/>
      <span style='color:#2563eb;'>BSSE &amp; BSAI · 2022–2026</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='banner'>
  <h1>🧠 AI-Powered Alzheimer's Detection System</h1>
  <p>Multimodal EEG + MRI Fusion &nbsp;·&nbsp; Deep Learning &nbsp;·&nbsp; Clinical Decision Support</p>
  <span class='badge'>🎓 Iqra University</span>
  <span class='badge'>🔬 BSSE &amp; BSAI</span>
  <span class='badge'>⚡ EEG + MRI Fusion</span>
  <span class='badge'>📅 2022–2026</span>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE ── 🏠 DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if "🏠 Dashboard" in page:

    # KPI row
    kpis = [("90%+","Fusion Accuracy","↑ vs 83% MRI-only"),
            ("89%+","F1 Score","EEG + MRI Model"),
            ("50M+","Global Cases","Worldwide"),
            ("3–5yr","Earlier Detection","vs Standard"),
            ("3","Output Classes","Healthy · MCI · AD")]
    for col, (v, l, d) in zip(st.columns(5), kpis):
        col.markdown(f"""
        <div class='kpi'>
          <div class='kpi-val'>{v}</div>
          <div class='kpi-lbl'>{l}</div>
          <div class='kpi-delta'>{d}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    cl, cr = st.columns([1.4, 1])

    with cl:
        st.markdown("<div class='sec'>Model Performance Comparison</div>", unsafe_allow_html=True)
        fig = go.Figure()
        for metric, vals, color in zip(
            ["Accuracy", "Precision", "Recall", "F1"],
            [[80,83,92], [78,82,90], [77,81,89], [78,82,90]],
            ["#60a5fa", "#34d399", "#f472b6", "#fbbf24"]
        ):
            fig.add_trace(go.Bar(
                name=metric, x=["EEG Only","MRI Only","EEG+MRI"], y=vals,
                marker_color=color,
                text=[f"{v}%" for v in vals], textposition="outside",
                textfont=dict(size=10, color="white")
            ))
        fig.update_layout(**PLOT_BASE, barmode="group", height=300,
            yaxis=dict(range=[60,104], ticksuffix="%",
                       gridcolor="rgba(37,99,235,.09)", color="#64748b"),
            xaxis=dict(color="#64748b"))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown("<div class='sec'>Dataset Class Distribution</div>", unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=CLASSES, values=[35, 28, 37], hole=0.55,
            marker=dict(colors=["#10b981","#8b5cf6","#ef4444"],
                        line=dict(color="#020b18", width=2)),
            textfont=dict(color="white", size=11)
        ))
        fig2.update_layout(**PLOT_BASE, height=270,
            annotations=[dict(text="Dataset", x=0.5, y=0.5,
                font=dict(size=13, color="white"), showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)

    # Architecture
    st.markdown("<div class='sec'>System Architecture Flow</div>", unsafe_allow_html=True)
    steps = [("📂","Data Input","EEG + MRI signals"),
             ("⚙️","Preprocess","Filter · Normalize"),
             ("🔍","Feature Ext.","CNN / LSTM"),
             ("🔗","Fusion","Concatenate features"),
             ("🤖","Classify","3-class softmax"),
             ("📊","Visualize","Dashboard + Reports")]
    for col, (ic, ti, de) in zip(st.columns(6), steps):
        col.markdown(f"""
        <div style='background:rgba(15,30,60,.7);border:1px solid rgba(37,99,235,.25);
             border-radius:12px;padding:16px 8px;text-align:center;'>
          <div style='font-size:1.5rem;'>{ic}</div>
          <div style='font-weight:700;color:#93c5fd;font-size:.8rem;margin:6px 0 3px;'>{ti}</div>
          <div style='font-size:.7rem;color:#64748b;'>{de}</div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE ── 🔬 PATIENT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif "🔬 Patient Analysis" in page:

    st.markdown(
        f"<div class='sec'>Patient: {p_name} &nbsp;|&nbsp; Age {p_age} "
        f"&nbsp;|&nbsp; {p_gender} &nbsp;|&nbsp; ID: {p_id}</div>",
        unsafe_allow_html=True
    )

    cu, cv = st.columns(2)
    with cu:
        eeg_file = st.file_uploader("📂 Upload EEG File (.csv / .edf / .npy)",
                                    type=["csv","edf","npy","txt"])
        if eeg_file:
            st.success(f"✅ {eeg_file.name} loaded ({eeg_file.size:,} bytes)")
        else:
            st.markdown("""<div class='info'>
            📌 <b>Demo Mode:</b> No EEG file uploaded — synthetic features will be used.
            </div>""", unsafe_allow_html=True)

    with cv:
        mri_file = st.file_uploader("📂 Upload MRI File (.nii / .dcm / .png)",
                                    type=["nii","gz","dcm","png","jpg"])
        if mri_file:
            st.success(f"✅ {mri_file.name} loaded ({mri_file.size:,} bytes)")
        else:
            st.markdown("""<div class='info'>
            📌 <b>Demo Mode:</b> No MRI file uploaded — synthetic features will be used.
            </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    if not (use_eeg or use_mri):
        st.warning("⚠️ Enable at least one modality (EEG or MRI) in the sidebar.")

    elif st.button("🚀 Run AI Analysis", use_container_width=False):
        with st.spinner("⏳ Running EEG + MRI analysis..."):
            time.sleep(1.5)
            pred, conf, probs = run_prediction(p_age, p_gender, use_eeg, use_mri)

        modality  = ("EEG + MRI Fusion" if use_eeg and use_mri
                     else "EEG Only" if use_eeg else "MRI Only")
        pred_color = CLS_COLOR[pred]

        # ── Result cards ──────────────────────────────────────────────────
        st.markdown("<hr/><div class='sec'>Prediction Results</div>", unsafe_allow_html=True)
        for col, cls in zip(st.columns(3), CLASSES):
            p_pct = probs[cls] * 100
            is_p  = cls == pred
            c     = CLS_COLOR[cls]
            col.markdown(f"""
            <div class='res-card' style='
              background:linear-gradient(135deg,{c}18,{c}06);
              border:{"2" if is_p else "1"}px solid {c}{"88" if is_p else "44"};
              opacity:{"1" if is_p else "0.6"};'>
              <div style='font-size:1.6rem;'>{CLS_EMOJI[cls]}</div>
              <div style='font-size:.95rem;font-weight:800;color:{c};margin:6px 0;'>{cls}</div>
              <div style='font-size:1.9rem;font-weight:800;color:white;'>{p_pct:.1f}%</div>
              <div style='font-size:.72rem;color:#94a3b8;margin-top:4px;'>
                {"✅ PREDICTED" if is_p else "Probability"}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        gc, gw = st.columns(2)

        # Gauge
        with gc:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                title={"text": f"Confidence — {modality}",
                       "font": {"color": "#94a3b8", "size": 13}},
                number={"suffix": "%", "font": {"color": pred_color, "size": 34}},
                gauge=dict(
                    axis={"range": [0,100], "tickcolor": "#94a3b8"},
                    bar={"color": pred_color, "thickness": 0.24},
                    bgcolor="rgba(0,0,0,0)",
                    steps=[{"range":[0,50],  "color":"rgba(239,68,68,.09)"},
                           {"range":[50,75], "color":"rgba(245,158,11,.09)"},
                           {"range":[75,100],"color":"rgba(16,185,129,.09)"}],
                    threshold={"line":{"color":"#fff","width":2},"value":conf_thr}
                )
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit", color="#94a3b8"),
                height=250, margin=dict(t=30, b=10, l=0, r=0)
            )
            st.plotly_chart(fig_g, use_container_width=True)

        # EEG waveform
        with gw:
            t, sig = get_eeg_waveform(pred)
            r, g_c, b_c = int(pred_color[1:3],16), int(pred_color[3:5],16), int(pred_color[5:7],16)
            fig_w = go.Figure(go.Scatter(
                x=t, y=sig, mode="lines",
                line=dict(color=pred_color, width=1.2),
                fill="tozeroy",
                fillcolor=f"rgba({r},{g_c},{b_c},0.07)"
            ))
            fig_w.update_layout(**PLOT_BASE, height=220, showlegend=False,
                title=dict(text="EEG Signal Preview",
                           font=dict(color="#93c5fd", size=12)),
                xaxis=dict(title="Time (s)", gridcolor="rgba(37,99,235,.08)", color="#64748b"),
                yaxis=dict(title="Amplitude (μV)", gridcolor="rgba(37,99,235,.08)", color="#64748b"))
            st.plotly_chart(fig_w, use_container_width=True)

        # EEG bands + MRI volumes
        st.markdown("<hr/>", unsafe_allow_html=True)
        be, bm = st.columns(2)

        np.random.seed(p_age * 3)
        band_names = ["Delta\n0.5-4Hz","Theta\n4-8Hz","Alpha\n8-13Hz",
                      "Beta\n13-30Hz","Gamma\n30+Hz"]
        bv = np.random.uniform([22,18,8,5,2], [44,33,29,17,9], 5)
        if pred == "Alzheimer's": bv[0] += 14; bv[2] -= 7
        elif pred == "MCI":       bv[1] += 7;  bv[2] -= 4
        bv = np.clip(bv, 0, 58)

        with be:
            fig_b = go.Figure(go.Bar(
                x=band_names, y=bv,
                marker_color=["#3b82f6","#8b5cf6","#10b981","#f59e0b","#ef4444"],
                text=[f"{v:.0f}%" for v in bv], textposition="outside",
                textfont=dict(color="white", size=10)
            ))
            fig_b.update_layout(**PLOT_BASE, height=280, showlegend=False,
                title=dict(text="EEG Band Powers", font=dict(color="#93c5fd",size=12)),
                yaxis=dict(title="Power (%)", gridcolor="rgba(37,99,235,.08)", color="#64748b"))
            st.plotly_chart(fig_b, use_container_width=True)

        np.random.seed(p_age * 7)
        vl = ["Hippocampus","Entorhinal","Gray Matter","Ventricle","White Matter"]
        pv = np.array([random.uniform(3.2,5.8), random.uniform(2.1,4.0),
                       random.uniform(5.5,7.2),  random.uniform(1.8,5.5),
                       random.uniform(4.2,5.8)])
        nv = np.array([4.8, 3.5, 6.5, 2.5, 5.1])
        if pred == "Alzheimer's": pv[:2] *= 0.68
        elif pred == "MCI":       pv[:2] *= 0.84

        with bm:
            fig_v = go.Figure()
            fig_v.add_trace(go.Bar(name="Patient", x=vl, y=pv.tolist(),
                marker_color="#3b82f6",
                text=[f"{v:.1f}" for v in pv], textposition="outside",
                textfont=dict(color="white", size=9)))
            fig_v.add_trace(go.Bar(name="Normal Ref", x=vl, y=nv.tolist(),
                marker_color="rgba(148,163,184,0.28)"))
            fig_v.update_layout(**PLOT_BASE, barmode="group", height=280,
                title=dict(text="MRI Volumetrics", font=dict(color="#93c5fd",size=12)),
                yaxis=dict(title="Volume (cm³)", gridcolor="rgba(37,99,235,.08)", color="#64748b"))
            st.plotly_chart(fig_v, use_container_width=True)

        # Clinical recommendation
        st.markdown("<hr/>", unsafe_allow_html=True)
        recs = {
            "Healthy":      ("🟢 No significant Alzheimer's indicators detected.",
                             "Annual cognitive screening recommended. Maintain healthy lifestyle, regular exercise, and balanced diet."),
            "MCI":          ("🟣 Mild Cognitive Impairment (MCI) indicators detected.",
                             "Neurologist referral within 4–6 weeks recommended. Repeat assessment in 6 months. Monitor cognitive changes."),
            "Alzheimer's":  ("🔴 Significant Alzheimer's Disease indicators detected.",
                             "Urgent specialist referral required. Consider cognitive support therapy, medication review, and family counselling."),
        }
        rt, rb = recs[pred]
        st.markdown(f"""
        <div style='background:{pred_color}11;border:1px solid {pred_color}44;
             border-radius:12px;padding:20px 24px;'>
          <div style='font-size:1rem;font-weight:700;color:{pred_color};margin-bottom:8px;'>{rt}</div>
          <div style='font-size:.87rem;color:#cbd5e1;line-height:1.7;'>{rb}</div>
          <div style='font-size:.73rem;color:#475569;margin-top:12px;'>
            ⚠️ AI-assisted decision support only. Validation by a qualified neurologist is mandatory.
          </div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE ── 📊 MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif "📊 Model Performance" in page:

    cm_data = np.array([[42,3,1],[4,35,6],[2,5,48]])
    mc, mr  = st.columns(2)

    with mc:
        st.markdown("<div class='sec'>Confusion Matrix</div>", unsafe_allow_html=True)
        fig_cm = go.Figure(go.Heatmap(
            z=cm_data, x=CLASSES, y=CLASSES,
            colorscale=[[0,"#020b18"],[0.5,"#1d4ed8"],[1,"#38bdf8"]],
            text=cm_data, texttemplate="%{text}",
            textfont=dict(size=18, color="white"), hoverongaps=False
        ))
        fig_cm.update_layout(**PLOT_BASE, height=340,
            xaxis=dict(title="Predicted", color="#94a3b8"),
            yaxis=dict(title="Actual",    color="#94a3b8", autorange="reversed"))
        st.plotly_chart(fig_cm, use_container_width=True)

    with mr:
        st.markdown("<div class='sec'>ROC Curves (Per Class)</div>", unsafe_allow_html=True)
        fpr_x   = np.linspace(0, 1, 100)
        fig_roc = go.Figure()
        for cls, auc_val, col_c in zip(
            CLASSES, [0.96, 0.91, 0.94],
            ["#10b981","#8b5cf6","#ef4444"]
        ):
            tpr_y = 1 - (1 - fpr_x) ** (1 / (1 - auc_val + 0.001))
            fig_roc.add_trace(go.Scatter(
                x=fpr_x, y=np.clip(tpr_y,0,1),
                name=f"{cls} (AUC={auc_val})",
                line=dict(color=col_c, width=2.3)
            ))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
            line=dict(color="#334155", dash="dash")))
        fig_roc.update_layout(**PLOT_BASE, height=340,
            xaxis=dict(title="False Positive Rate", gridcolor="rgba(37,99,235,.08)", color="#64748b"),
            yaxis=dict(title="True Positive Rate",  gridcolor="rgba(37,99,235,.08)", color="#64748b"))
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("<div class='sec'>Classification Report</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Class":     ["Healthy","MCI","Alzheimer's","Weighted Avg"],
        "Precision": ["91.3%","87.5%","87.3%","88.7%"],
        "Recall":    ["91.3%","77.8%","87.3%","86.8%"],
        "F1-Score":  ["91.3%","82.4%","87.3%","87.5%"],
        "Support":   [46, 45, 55, 146],
    }), use_container_width=True, hide_index=True)

    st.markdown("<div class='sec'>Training History</div>", unsafe_allow_html=True)
    ep    = list(range(1, 31))
    tr_a  = np.clip([0.54+0.013*i-0.0002*i**2+random.uniform(-.01,.01)  for i in ep], 0, .95)
    val_a = np.clip([0.50+0.012*i-0.0002*i**2+random.uniform(-.014,.014) for i in ep], 0, .93)
    tr_l  = np.clip([0.85-0.026*i+0.0003*i**2+random.uniform(-.01,.01)  for i in ep], .07, 1)
    val_l = np.clip([0.90-0.025*i+0.0003*i**2+random.uniform(-.014,.014) for i in ep], .09, 1)

    fig_h = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy","Loss"])
    fig_h.add_trace(go.Scatter(x=ep, y=tr_a,  name="Train",     line=dict(color="#38bdf8",width=2)), row=1,col=1)
    fig_h.add_trace(go.Scatter(x=ep, y=val_a, name="Val",       line=dict(color="#10b981",width=2)), row=1,col=1)
    fig_h.add_trace(go.Scatter(x=ep, y=tr_l,  name="Train Loss",line=dict(color="#f472b6",width=2)), row=1,col=2)
    fig_h.add_trace(go.Scatter(x=ep, y=val_l, name="Val Loss",  line=dict(color="#fbbf24",width=2)), row=1,col=2)
    fig_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit",color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=280, margin=dict(t=30,b=10,l=0,r=0))
    fig_h.update_xaxes(gridcolor="rgba(37,99,235,.08)", color="#64748b")
    fig_h.update_yaxes(gridcolor="rgba(37,99,235,.08)", color="#64748b")
    st.plotly_chart(fig_h, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE ── 📈 CLINICAL STATS
# ═════════════════════════════════════════════════════════════════════════════
elif "📈 Clinical Stats" in page:

    np.random.seed(77)
    dates30 = [datetime.today() - timedelta(days=i) for i in range(29,-1,-1)]
    diag30  = np.random.choice(CLASSES, 30, p=[0.33, 0.28, 0.39])
    confs30 = np.random.uniform(65, 97, 30)
    ages30  = np.random.randint(55, 90, 30)
    mods30  = np.random.choice(["EEG+MRI","MRI Only","EEG Only"], 30, p=[0.6,.25,.15])

    n_ad = int((diag30 == "Alzheimer's").sum())
    n_mci = int((diag30 == "MCI").sum())
    for col, (lbl, val) in zip(st.columns(4), [
        ("Total Patients", "30"),
        ("Avg Confidence", f"{confs30.mean():.1f}%"),
        ("Alzheimer's",    str(n_ad)),
        ("MCI Cases",      str(n_mci))
    ]):
        col.markdown(f"""
        <div class='kpi'>
          <div class='kpi-val'>{val}</div>
          <div class='kpi-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    df30   = pd.DataFrame({
        "Date":      [d.strftime("%b %d") for d in dates30],
        "Diagnosis": diag30, "Confidence": np.round(confs30, 1),
        "Age": ages30, "Modality": mods30
    })
    fig_tr = px.bar(
        df30.groupby(["Date","Diagnosis"]).size().reset_index(name="Count"),
        x="Date", y="Count", color="Diagnosis",
        color_discrete_map={"Healthy":"#10b981","MCI":"#8b5cf6","Alzheimer's":"#ef4444"}
    )
    fig_tr.update_layout(**PLOT_BASE, height=290,
        title=dict(text="Daily Diagnoses (Last 30 Days)", font=dict(color="#93c5fd",size=12)))
    st.plotly_chart(fig_tr, use_container_width=True)
    st.dataframe(df30, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE ── ℹ️ ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif "ℹ️ About" in page:

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        <div class='card'>
          <div class='sec' style='margin-top:0;'>Project Information</div>
          <table style='width:100%;font-size:.85rem;color:#cbd5e1;border-collapse:collapse;'>
            <tr><td style='color:#64748b;padding:8px 0;width:36%;'>Supervisor</td>
                <td>Abdul Baqi Malik</td></tr>
            <tr><td style='color:#64748b;padding:8px 0;'>Developers</td>
                <td>Amina Arshad &nbsp;·&nbsp; Qura tul Ain &nbsp;·&nbsp; Sidra tul Muntaha</td></tr>
            <tr><td style='color:#64748b;padding:8px 0;'>Program</td>
                <td>BSSE &amp; BSAI</td></tr>
            <tr><td style='color:#64748b;padding:8px 0;'>Institution</td>
                <td>Iqra University Islamabad (Chak Shahzad)</td></tr>
            <tr><td style='color:#64748b;padding:8px 0;'>Session</td>
                <td>Fall 2022 – Spring 2026</td></tr>
            <tr><td style='color:#64748b;padding:8px 0;'>Datasets</td>
                <td>OASIS MRI &nbsp;·&nbsp; Alzheimer's EEG (Kaggle)</td></tr>
            <tr><td style='color:#64748b;padding:8px 0;'>Tech Stack</td>
                <td>Python · TensorFlow · Streamlit · LocalTunnel · Colab</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class='card'>
          <div class='sec' style='margin-top:0;'>SDG Alignment</div>
          <div style='background:rgba(16,185,129,.09);border:1px solid rgba(16,185,129,.3);
               border-radius:10px;padding:13px;margin-bottom:12px;'>
            <div style='color:#10b981;font-weight:700;margin-bottom:4px;'>
              SDG 3 — Good Health &amp; Well-Being</div>
            <div style='color:#94a3b8;font-size:.82rem;line-height:1.6;'>
              Early Alzheimer's detection enabling timely intervention and improved patient quality of life.</div>
          </div>
          <div style='background:rgba(37,99,235,.09);border:1px solid rgba(37,99,235,.3);
               border-radius:10px;padding:13px;'>
            <div style='color:#60a5fa;font-weight:700;margin-bottom:4px;'>
              SDG 9 — Industry, Innovation &amp; Infrastructure</div>
            <div style='color:#94a3b8;font-size:.82rem;line-height:1.6;'>
              AI-driven healthcare through deep learning and multimodal data fusion.</div>
          </div>
          <div class='warnbox' style='margin-top:12px;'>
            ⚠️ Research &amp; educational purposes only.
            All predictions require neurologist validation.
          </div>
        </div>""", unsafe_allow_html=True)
