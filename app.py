import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import torch
torch.device("cpu")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(
    page_title="MockData",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Kaggle credentials da Streamlit secrets (solo in cloud)
if hasattr(st, "secrets") and "kaggle" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["kaggle"]["username"]
    os.environ["KAGGLE_KEY"]      = st.secrets["kaggle"]["key"]
    
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 720px; }

  /* ── Background: warm off-white ── */
  .stApp { background: #eef0f7; }
  .block-container { background: #eef0f7; }
  p, label, div { color: #1a1a2e; }
  .stMarkdown p { color: #374151; }

  /* ── Header: mantiene gradient scuro per contrasto logo ── */
  .md-header {
    background: linear-gradient(135deg, #3730a3 0%, #1d4ed8 60%, #0369a1 100%);
    border-radius: 16px;
    padding: 36px 40px 30px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(79,70,229,0.15);
  }
  .md-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(99,102,241,0.2);
  }
  .md-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    color: #fff; margin: 0 0 6px; letter-spacing: -1px;
    position: relative; z-index: 1;
  }
  .md-logo span { color: #a5b4fc; }
  .md-tagline {
    font-size: 0.95rem; color: rgba(255,255,255,0.65);
    margin: 0; position: relative; z-index: 1;
  }

  /* ── Section labels ── */
  .section-label {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: .1em; text-transform: uppercase;
    color: #4f46e5; margin-bottom: 10px; display: block;
  }

  /* ── Metric cards: light ── */
  .metric-row { display: flex; gap: 12px; margin: 20px 0; }
  .metric-card {
    flex: 1;
    background: #ffffff;
    border-radius: 12px; padding: 16px 18px; text-align: center;
    border: 1px solid #e0e3f0;
    box-shadow: 0 2px 8px rgba(79,70,229,0.07);
  }
  .metric-card .val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem; font-weight: 600;
    color: #1e1b4b; display: block; margin-bottom: 2px;
  }
  .metric-card .lbl {
    font-size: 0.72rem; color: #6b7280;
    text-transform: uppercase; letter-spacing: .07em;
  }
  .metric-card .delta {
    font-size: 0.75rem; color: #059669;
    margin-top: 2px; display: block;
  }

  /* ── Disclaimer ── */
  .disclaimer {
    font-size: 0.72rem; color: #9ca3af;
    text-align: center; margin-top: 8px; line-height: 1.6;
  }

  /* ── Generate button ── */
  .stButton > button {
    background: linear-gradient(135deg, #4f46e5, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important; font-weight: 600 !important;
    padding: 0.7rem 1.5rem !important; width: 100% !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  /* ── Download button ── */
  .stDownloadButton > button {
    background: transparent !important; color: #4f46e5 !important;
    border: 1px solid rgba(79,70,229,0.4) !important;
    border-radius: 10px !important; width: 100% !important;
  }

  /* ── Sliders ── */
  .stSlider label { font-size: 0.88rem !important; color: #374151 !important; font-weight: 500 !important; }
  .stSlider [data-testid="stTickBar"] { color: #6b7280 !important; }

  /* ── Dataset cards text override ── */
  .ds-card-title { font-family: 'Space Grotesk', sans-serif; font-size: 1rem;
    font-weight: 600; color: #1e1b4b; margin-bottom: 4px; }
  .ds-card-desc  { font-size: 0.8rem; color: #6b7280; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    from sdv.single_table import CTGANSynthesizer
    return CTGANSynthesizer.load("ctgan_paysim.pkl")


@st.cache_data(show_spinner=False)
def load_real_sample():
    if not os.path.exists("paysim.csv"):
        import kaggle
        with st.spinner("Downloading PaySim dataset (~470MB, first run only)..."):
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "ealaxi/paysim1",
                path=".",
                unzip=True
            )
            # Rinomina il file con il nome corretto
            import glob
            csv_files = glob.glob("PS_*.csv")
            if csv_files:
                os.rename(csv_files[0], "paysim.csv")

    df = pd.read_csv("paysim.csv")
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])
    return df.sample(n=10000, random_state=42)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 24px;">
  <p style="font-family:'Space Grotesk',sans-serif; font-size:2.4rem;
            font-weight:700; color:#1e1b4b; margin:0 0 8px; letter-spacing:-1px;">
    Mock<span style="color:#4f46e5;">Data</span>
  </p>
  <p style="font-size:1rem; color:#6b7280; margin:0;">
    Generate realistic datasets in 30 seconds — no code required
  </p>
</div>
""", unsafe_allow_html=True)

# ── Dataset selector ──────────────────────────────────────────────────────────
st.markdown('<span class="section-label">01 — Choose dataset</span>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div style="border:2px solid #4f46e5;border-radius:12px;padding:16px 18px;
                background:#eef2ff;box-shadow:0 2px 8px rgba(79,70,229,0.1);">
      <div class="ds-card-title">⚡ PaySim — Banking</div>
      <div class="ds-card-desc">Financial transactions with fraud patterns</div>
      <span style="font-size:0.7rem;background:rgba(79,70,229,0.12);color:#4f46e5;
                   border-radius:20px;padding:3px 10px;font-weight:600;">
        TRANSFER · CASH_OUT
      </span>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="border:1px solid #e5e7eb;border-radius:12px;padding:16px 18px;
                background:#f9fafb;opacity:0.55;cursor:not-allowed;">
      <div class="ds-card-title">⚽ StatsBomb — Football</div>
      <div class="ds-card-desc">Match events &amp; player performance</div>
      <span style="font-size:0.7rem;background:#f3f4f6;color:#9ca3af;
                   border-radius:20px;padding:3px 10px;">Coming soon</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

# ── Configure ─────────────────────────────────────────────────────────────────
st.markdown('<span class="section-label">02 — Configure</span>', unsafe_allow_html=True)

# FIX 2: max rows 100,000
num_rows = st.slider(
    "Number of rows",
    min_value=100,
    max_value=100000,
    value=5000,
    step=100
)

# FIX 3: label esplicita con range chiarificato
st.markdown(
    "<p style='font-size:0.88rem;font-weight:500;color:#374151;margin-bottom:4px;'>Fraud rate "
    "<span style='font-size:0.78rem;color:#6b7280;font-weight:400;'>"
    "(0% = no fraud · 10% = 1 in 10 transactions · real data ≈ 0.3%)</span></p>",
    unsafe_allow_html=True
)
fraud_pct = st.slider(
    "Fraud rate",
    min_value=0.0,
    max_value=10.0,
    value=0.3,
    step=0.1,
    format="%.1f%%",
    label_visibility="collapsed"
)

st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

# ── Generate ──────────────────────────────────────────────────────────────────
generate = st.button("⚡  Generate dataset", use_container_width=True)

if generate:
    with st.spinner("Loading model..."):
        synthesizer = load_model()
        real_sample = load_real_sample()

    n_fraud  = max(1, int(num_rows * fraud_pct / 100))
    n_normal = num_rows - n_fraud

    with st.spinner(f"Generating {num_rows:,} rows..."):
        normal = synthesizer.sample_remaining_columns(
            known_columns=pd.DataFrame({"isFraud": [0] * n_normal}),
            max_tries_per_batch=100
        )
        fraud = synthesizer.sample_remaining_columns(
            known_columns=pd.DataFrame({"isFraud": [1] * n_fraud}),
            max_tries_per_batch=100
        )
        synthetic = (
            pd.concat([normal, fraud])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        clip_val = real_sample["amount"].quantile(0.995)
        synthetic["amount"] = synthetic["amount"].clip(upper=clip_val)

    # ── Metrics ───────────────────────────────────────────────────────────────
    real_mean   = real_sample["amount"].mean()
    synth_mean  = synthetic["amount"].mean()
    delta_pct   = ((synth_mean - real_mean) / real_mean) * 100
    real_fraud  = real_sample["isFraud"].mean()
    synth_fraud = synthetic["isFraud"].mean()

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <span class="val">{num_rows:,}</span>
        <span class="lbl">Rows generated</span>
        <span class="delta">✓ synthetic</span>
      </div>
      <div class="metric-card">
        <span class="val">{synth_fraud*100:.2f}%</span>
        <span class="lbl">Fraud rate</span>
        <span class="delta">real: {real_fraud*100:.2f}%</span>
      </div>
      <div class="metric-card">
        <span class="val">{synth_mean/1000:.0f}k</span>
        <span class="lbl">Amount mean</span>
        <span class="delta">{delta_pct:+.1f}% vs real</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts — light theme ──────────────────────────────────────────────────
    st.markdown('<span class="section-label">03 — Validation</span>', unsafe_allow_html=True)

    BG    = "#f8f9fc"
    GRID  = "rgba(0,0,0,0.08)"
    FONT  = "#000000"
    BLUE  = "rgba(79,70,229,0.65)"
    ORANGE= "rgba(234,88,12,0.65)"

    clip_viz      = real_sample["amount"].quantile(0.99)
    real_clipped  = real_sample["amount"].clip(upper=clip_viz)
    synth_clipped = synthetic["amount"].clip(upper=clip_viz)

    # Plot 1 — Amount distribution
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=real_clipped,  nbinsx=60, name="Real",
        marker_color=BLUE, histnorm="probability density"))
    fig1.add_trace(go.Histogram(x=synth_clipped, nbinsx=60, name="Synthetic",
        marker_color=ORANGE, histnorm="probability density"))
    fig1.update_layout(
        title="Amount distribution — real vs synthetic", barmode="overlay",
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=FONT, family="DM Sans"),
        title_font=dict(size=14, color="#1e1b4b"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#000000")),
        margin=dict(t=50, b=40, l=40, r=20), height=300,
        xaxis=dict(gridcolor=GRID, tickformat=",.0f", tickfont=dict(color="#000000")),
        yaxis=dict(gridcolor=GRID, tickfont=dict(color="#000000"))
    )
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        type_real  = real_sample["type"].value_counts(normalize=True).round(3)
        type_synth = synthetic["type"].value_counts(normalize=True).round(3)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Real", x=type_real.index, y=type_real.values,
            marker_color=BLUE))
        fig2.add_trace(go.Bar(name="Synthetic", x=type_synth.index, y=type_synth.values,
            marker_color=ORANGE))
        fig2.update_layout(
            title="Transaction types", barmode="group",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=FONT, family="DM Sans"),
            title_font=dict(size=13, color="#1e1b4b"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#000000")),
            margin=dict(t=44, b=32, l=36, r=12), height=260,
            xaxis=dict(tickfont=dict(color="#000000")),
            yaxis=dict(gridcolor=GRID, tickformat=".0%", tickfont=dict(color="#000000"))
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=["Real", "Synthetic"],
            y=[real_fraud*100, synth_fraud*100],
            marker_color=[BLUE, ORANGE],
            text=[f"{real_fraud*100:.3f}%", f"{synth_fraud*100:.3f}%"],
            textposition="outside",
            textfont=dict(color="#1e1b4b", size=12)
        ))
        fig3.update_layout(
            title="Fraud rate",
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=FONT, family="DM Sans"),
            title_font=dict(size=13, color="#1e1b4b"),
            showlegend=False,
            margin=dict(t=44, b=32, l=36, r=12), height=260,
            xaxis=dict(tickfont=dict(color="#000000")),
            yaxis=dict(gridcolor=GRID, ticksuffix="%",
                       tickfont=dict(color="#000000"),
                       range=[0, max(real_fraud, synth_fraud)*100*1.6])
        )
        st.plotly_chart(fig3, use_container_width=True)

    # FIX 4: heatmap con abbrevazioni + margini generosi per evitare overlap
    num_cols  = ["step", "amount", "oldbalOrg", "newbalOrg", "oldbalDest", "newbalDest"]
    full_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

    corr_real  = real_sample[full_cols].corr().round(2)
    corr_synth = synthetic[full_cols].corr().round(2)
    corr_real.columns  = corr_real.index  = num_cols
    corr_synth.columns = corr_synth.index = num_cols

    fig4 = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Correlation — real", "Correlation — synthetic"],
        horizontal_spacing=0.18
    )
    fig4.add_trace(go.Heatmap(
        z=corr_real.values, x=num_cols, y=num_cols,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr_real.values.round(2), texttemplate="%{text}",
        textfont=dict(size=11), showscale=False
    ), row=1, col=1)
    fig4.add_trace(go.Heatmap(
        z=corr_synth.values, x=num_cols, y=num_cols,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr_synth.values.round(2), texttemplate="%{text}",
        textfont=dict(size=11), showscale=True,
        colorbar=dict(tickfont=dict(color=FONT, size=10), thickness=12, len=0.9)
    ), row=1, col=2)
    fig4.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=FONT, family="DM Sans", size=11),
        margin=dict(t=60, b=80, l=70, r=70),
        height=380
    )
    fig4.update_xaxes(tickangle=35, tickfont=dict(size=10, color="#1e1b4b"))
    fig4.update_yaxes(tickfont=dict(size=10, color="#1e1b4b"))
    for ann in fig4.layout.annotations:
        ann.font.color = "#1e1b4b"
        ann.font.size  = 13
    st.plotly_chart(fig4, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    csv_buffer = io.StringIO()
    synthetic.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇  Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"mockdata_paysim_{num_rows}rows.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.markdown("""
    <p class="disclaimer">
      Distributions are approximate &nbsp;·&nbsp;
      Transaction type split may vary ±15% &nbsp;·&nbsp;
      Not suitable for production compliance use
    </p>""", unsafe_allow_html=True)
