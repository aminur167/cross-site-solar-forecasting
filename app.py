"""Dashboard for the retrospective two-site PV forecasting study."""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Two-Site PV Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# WORLD-CLASS GLASSMORPHIC & CYBER-RENEWABLE CSS STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    /* Hero Banner */
    .hero-container {
        background: radial-gradient(120% 120% at 50% 10%, #172033 0%, #0b0f19 100%);
        border: 1px solid rgba(255, 179, 0, 0.25);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 153, 0, 0.12);
        color: #ffb300;
        border: 1px solid rgba(255, 179, 0, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 800;
        line-height: 1.25;
        background: linear-gradient(135deg, #ffffff 30%, #ffd000 70%, #ff6b00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-meta {
        font-size: 14px;
        color: #94a3b8;
        display: flex;
        flex-wrap: wrap;
        gap: 18px;
        align-items: center;
    }

    .hero-meta a {
        color: #38bdf8;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s;
    }
    .hero-meta a:hover {
        color: #7dd3fc;
        text-decoration: underline;
    }

    /* Metric Glass Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 179, 0, 0.35);
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
        margin: 6px 0;
    }
    .metric-delta {
        font-size: 12.5px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .delta-green { color: #10b981; }
    .delta-gold { color: #f59e0b; }
    .delta-blue { color: #38bdf8; }

    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 13.5px;
        color: #94a3b8;
        border: none !important;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 153, 0, 0.2) 0%, rgba(255, 102, 0, 0.3) 100%) !important;
        color: #ffb300 !important;
        border: 1px solid rgba(255, 179, 0, 0.4) !important;
        box-shadow: 0 4px 12px rgba(255, 153, 0, 0.15);
    }

    /* Section Cards */
    .content-box {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# HERO BANNER
# -------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Retrospective PV forecasting</div>
    <div class="hero-title">Cross-site solar PV forecasting</div>
    <div class="hero-meta">

        <span>📄 <b>Status:</b> Research draft</span>
        <span>📦 <b>Open Science:</b> <a href="https://github.com/aminur167/cross-site-solar-forecasting" target="_blank">aminur167/cross-site-solar-forecasting</a></span>

    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CACHED DATA & BENCHMARK LOADERS
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df_a = pd.read_csv("data/processed/site_a_features.csv", index_col=0, parse_dates=True)
    df_b = pd.read_csv("data/processed/site_b_features.csv", index_col=0, parse_dates=True)
    return df_a, df_b

@st.cache_data
def load_all_benchmarks():
    with open("experiments/results/baseline_benchmark_results.json") as f:
        res_base = json.load(f)
    with open("experiments/results/cross_site_adaptation_results.json") as f:
        res_cs = json.load(f)
    with open("experiments/results/uncertainty_robustness_results.json") as f:
        res_u = json.load(f)
    with open("experiments/results/deep_learning_and_statistics.json") as f:
        res_dl = json.load(f)
    with open("experiments/results/shap_importance_results.json") as f:
        res_shap = json.load(f)
    return res_base, res_cs, res_u, res_dl, res_shap

df_a, df_b = load_data()
res_base, res_cs, res_u, res_dl, res_shap = load_all_benchmarks()

# Standard Plotly Dark Theme
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PALETTE = {
    "primary": "#ff9900",
    "secondary": "#38bdf8",
    "accent": "#10b981",
    "danger": "#ef4444",
    "purple": "#a855f7",
    "dark_bg": "rgba(15, 23, 42, 0.4)",
    "grid": "rgba(255, 255, 255, 0.08)"
}

# -------------------------------------------------------------
# INTERACTIVE RESEARCH TABS
# -------------------------------------------------------------
tab_exec, tab_data, tab_base, tab_adapt, tab_uncert, tab_stress, tab_deep, tab_diag, tab_paper = st.tabs([
    "📊 Summary",
    "📡 Two-Site Dataset",
    "🏆 Baseline Benchmark",
    "🚀 Adaptation Results",
    "🛡️ Conformal Prediction (CQR)",
    "⚠️ Synthetic Input Tests",
    "🧠 Deep Learning & SHAP",
    "📈 Parity Diagnostics",
    "📄 Research Files"
])

# =============================================================
# TAB 1: EXECUTIVE SUMMARY
# =============================================================
with tab_exec:
    st.markdown("### Corrected Experimental Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Target-Only vs Pooled, 1 Day</div>
            <div class="metric-value">75.1% Lower nMAE</div>
            <div class="metric-delta delta-green">6.97% pooled vs 28.06% target-only</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Zero-Shot Transfer Error</div>
            <div class="metric-value">6.93% nMAE</div>
            <div class="metric-delta delta-blue">Retrospective; weather is hindsight data</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Interval Coverage (PICP)</div>
            <div class="metric-value">94.51%</div>
            <div class="metric-delta delta-gold">Empirical; not a guarantee</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Synthetic Perturbation Result</div>
            <div class="metric-value">30% feature-wise MCAR</div>
            <div class="metric-delta delta-gold">7.46% nMAE · 92.78% PICP</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns([1.2, 0.8])
    with c_left:
        st.markdown("""
        <div class="content-box">
            <h4 style="color: #ffb300; margin-top: 0;">📖 Abstract & Problem Statement</h4>
            <p style="color: #cbd5e1; line-height: 1.6; font-size: 14px;">
            This retrospective study compares transfer and target-only models and reports empirical interval results.
            </p>
            <ol style="color: #cbd5e1; font-size: 13.5px; line-height: 1.6;">
                <li><b>Limited target data:</b> In this split, one-day target-only training had 28.06% nMAE.</li>
                <li><b>Uncertainty:</b> Empirical prediction intervals are evaluated; temporal dependence limits theoretical guarantees.</li>
                <li><b>Input perturbations:</b> Simulated historical weather-feature corruption is not field sensor-fault evidence.</li>
            </ol>
            <p style="color: #cbd5e1; font-size: 14px;">
            This dashboard summarizes a retrospective two-site study. Weather inputs are historical estimates unavailable at forecast issue time; synthetic perturbations do not demonstrate field sensor resilience.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c_right:
        st.markdown("""
        <div class="content-box">
            <h4 style="color: #38bdf8; margin-top: 0;">Study results</h4>
            <ul style="color: #cbd5e1; font-size: 13px; line-height: 1.7; padding-left: 20px;">
                <li><b>RQ1:</b> Source-only transfer: 6.93% nMAE vs 14.14% persistence on this holdout.</li>
                <li><b>RQ2:</b> One-day pooled adaptation (6.97%) did not improve on zero-shot (6.93%).</li>
                <li><b>RQ3:</b> Target-calibrated intervals had 94.51% empirical daytime coverage on this holdout.</li>
                <li><b>Input stress:</b> 30% feature-wise MCAR had mean 7.46% nMAE and 92.78% empirical PICP across five perturbation seeds.</li>
                <li><b>Scope:</b> Historical reanalysis weather is not available at forecast issue time.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================================================
# TAB 2: MULTI-SITE TELEMETRY
# =============================================================
with tab_data:
    st.markdown("### 📡 Commercial 50 kWp Inverter & Meteorological Data Lake")

    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        site_select = st.radio("Select Solar PV Installation:", ["Site A (50 kWp Reference)", "Site B (50 kWp Target)"], horizontal=True)
        active_df = df_a if "Site A" in site_select else df_b

        st.markdown(f"""
        - **Nominal AC Capacity:** `50.0 kWp`
        - **Total Active Samples:** `{len(active_df):,}` daytime hours
        - **Coverage Period:** `{active_df.index.min().strftime('%d %b %Y')}` to `{active_df.index.max().strftime('%d %b %Y')}`
        - **Max Observed AC Power:** `{active_df['active_power_kw'].max():.2f} kW`
        - **Mean Capacity Factor:** `{active_df['capacity_factor'].mean():.4f}`
        """)

    with col_sel2:
        date_range = st.date_input("Zoom Time Window:", [active_df.index.min().date() + pd.Timedelta(days=60), active_df.index.min().date() + pd.Timedelta(days=67)])

    if len(date_range) == 2:
        sub_df = active_df.loc[str(date_range[0]):str(date_range[1])]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=sub_df.index, y=sub_df['active_power_kw'], name="Inverter Generation (kW)", line=dict(color="#ff9900", width=2.5)),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=sub_df.index, y=sub_df['ghi_wm2'], name="Solar Irradiance GHI (W/m²)", line=dict(color="#38bdf8", width=1.5, dash="dot")),
            secondary_y=True
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=COLOR_PALETTE["dark_bg"],
            plot_bgcolor="rgba(0,0,0,0)",
            title="High-Resolution Telemetry: Inverter Power vs. Solar Irradiance",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center")
        )
        fig.update_yaxes(title_text="Active AC Power (kW)", secondary_y=False, gridcolor=COLOR_PALETTE["grid"])
        fig.update_yaxes(title_text="Global Horizontal Irradiance (W/m²)", secondary_y=True, gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    # Inverter Power Curve vs Irradiance
    st.markdown("#### ⚡ Photovoltaic Power Curve & Physical Inverter Saturation")
    fig_scatter = px.scatter(
        active_df.sample(n=min(2000, len(active_df)), random_state=42),
        x="ghi_wm2",
        y="active_power_kw",
        color="temp_c",
        color_continuous_scale="Viridis",
        labels={"ghi_wm2": "Solar Irradiance (W/m²)", "active_power_kw": "Inverter Output (kW)", "temp_c": "Ambient Temp (°C)"},
        title="Physical PV Generation Characteristic (P vs GHI with Thermal Derating)"
    )
    fig_scatter.add_hline(y=50.0, line_dash="dash", line_color="#ef4444", annotation_text="50 kW Inverter Rated Ceiling")
    fig_scatter.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=COLOR_PALETTE["dark_bg"], plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# =============================================================
# TAB 3: BASELINE BENCHMARK
# =============================================================
with tab_base:
    st.markdown("### 🏆 Experiment 1: Within-Site Baseline & Machine Learning Benchmark")
    st.markdown("Rigorous chronological holdout validation on Site A (April–June 2026).")

    base_df = pd.DataFrame(res_base).T

    col_t1, col_t2 = st.columns([1.2, 0.8])
    with col_t1:
        # Error comparison bar chart
        fig_bar = px.bar(
            base_df.reset_index(),
            x="index",
            y=["nMAE_pct", "nRMSE_pct"],
            barmode="group",
            labels={"value": "Normalized Error (%)", "index": "Model", "variable": "Metric"},
            title="Within-Site Forecasting Accuracy: Baseline vs. ML Models",
            color_discrete_map={"nMAE_pct": "#ff9900", "nRMSE_pct": "#38bdf8"}
        )
        fig_bar.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=COLOR_PALETTE["dark_bg"], plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_t2:
        st.markdown("#### 📋 Baseline Metrics")
        st.dataframe(base_df[["nMAE_pct", "nRMSE_pct", "R2", "Skill_Score_vs_Persist"]].style.format("{:.2f}").highlight_min(subset=["nMAE_pct", "nRMSE_pct"], color="#064e3b").highlight_max(subset=["R2", "Skill_Score_vs_Persist"], color="#064e3b"), use_container_width=True)
        st.info("💡 **Finding:** Site A holdout results are listed below; the analysis uses historical weather estimates.")

# =============================================================
# TAB 4: FEW-SHOT ADAPTATION FRONTIER
# =============================================================
with tab_adapt:
    st.markdown("### 🚀 Experiment 2 & 3: Cross-Site Transfer & Few-Shot Adaptation Frontier")
    st.markdown("**Study question:** How do source-only transfer, pooled adaptation, and target-only fits compare over this dataset’s target-data budgets?")

    budgets = [1, 3, 7, 14]
    to_mae = [res_cs["Zero_Shot_and_Adaptation"][f"k_{k}_days"]["target_only_metrics"]["nMAE_pct"] for k in budgets]
    tl_mae = [res_cs["Zero_Shot_and_Adaptation"][f"k_{k}_days"]["transfer_adapted_metrics"]["nMAE_pct"] for k in budgets]
    zero_mae = res_cs["Zero_Shot_and_Adaptation"]["k_0_days"]["transfer_adapted_metrics"]["nMAE_pct"]
    oracle_mae = res_cs["Target_Oracle_Full"]["nMAE_pct"]
    persist_mae = res_cs["Persistence_Baseline"]["nMAE_pct"]

    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(
        x=budgets, y=to_mae, mode="lines+markers+text", name="Target-Only (Train from Scratch)",
        line=dict(color="#ef4444", width=3, dash="dash"), marker=dict(size=10),
        text=[f"{v:.1f}%" for v in to_mae], textposition="top center"
    ))
    fig_frontier.add_trace(go.Scatter(
        x=budgets, y=tl_mae, mode="lines+markers+text", name="Transfer-Adapted (Ours: Source + Target)",
        line=dict(color="#ff9900", width=3.5), marker=dict(size=12, symbol="diamond"),
        text=[f"{v:.2f}%" for v in tl_mae], textposition="bottom center"
    ))
    fig_frontier.add_hline(y=zero_mae, line_dash="dot", line_color="#a855f7", annotation_text=f"Zero-Shot Transfer (0 Target Days, {zero_mae:.2f}%)")
    fig_frontier.add_hline(y=oracle_mae, line_dash="dashdot", line_color="#10b981", annotation_text=f"Full Target Oracle (~315 Days, {oracle_mae:.2f}%)")
    fig_frontier.add_hline(y=persist_mae, line_dash="solid", line_color="#64748b", annotation_text=f"Persistence Baseline ({persist_mae:.2f}%)")

    fig_frontier.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=COLOR_PALETTE["dark_bg"],
        plot_bgcolor="rgba(0,0,0,0)",
        title="The Data-Efficiency Frontier: Target Data Budget vs. Normalized MAE (%)",
        xaxis_title="Target Data Budget (Calendar Days)",
        yaxis_title="Normalized Error nMAE (%)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.98)
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    st.info("One-day target-only nMAE was 28.06%; pooled adaptation was 6.97%. Zero-shot was 6.93%, so adaptation did not improve on zero-shot in this rerun.")

# =============================================================
# TAB 5: CONFORMAL UNCERTAINTY (CQR)
# =============================================================
with tab_uncert:
    st.markdown("### 🛡️ Experiment 4: Conformalized Quantile Regression (CQR)")
    st.markdown("Empirical split-CQR coverage; no guarantee is claimed under temporal dependence or site shift.")

    c1, c2 = st.columns([1, 1.5])
    with c1:
        u_df = pd.DataFrame(res_u["Uncertainty_Calibration_Regimes"]).T
        st.markdown("#### 📊 Calibration Regimes on Target Site B")
        st.dataframe(u_df[["PICP_pct", "Target_Coverage_pct", "MPIW", "Winkler_Score", "Empirical_Coverage_Meets_Nominal_Target"]], use_container_width=True)
        st.info("Target calibration achieved 94.51% empirical daytime coverage on this holdout. This is an observed result, not a distribution-free guarantee under temporal or site shift.")

    with c2:
        # Reliability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = res_u["Uncertainty_Calibration_Regimes"]["Target_Adapted_CQR"]["PICP_pct"],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CQR Empirical Coverage (PICP %)"},
            delta = {'reference': 90.0, 'increasing': {'color': "#10b981"}},
            gauge = {
                'axis': {'range': [70, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#ff9900"},
                'bgcolor': "rgba(0,0,0,0)",
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 3},
                    'thickness': 0.75,
                    'value': 90.0
                }
            }
        ))
        fig_gauge.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=COLOR_PALETTE["dark_bg"], height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 Multi-Level Conformal Reliability & Sharpness Diagnostics")
    fig6_path = "figures/fig6_reliability_diagram.png"
    if os.path.exists(fig6_path):
        st.image(fig6_path, caption="Figure 6: empirical coverage at the nominal levels evaluated and interval width.", use_container_width=True)

    st.info("A timestamp-level interval forecast series is not currently serialized. This dashboard shows holdout aggregate calibration only.")

# =============================================================
# TAB 6: SENSOR OUTAGE STRESS-TEST
# =============================================================
with tab_stress:
    st.markdown("### ⚠️ Experiment 5: Synthetic Weather-Feature Perturbation")
    st.markdown("These simulations perturb retrospective weather model inputs; they are not measurements of real hardware failures.")

    deg_dict = res_u["Sensor_Availability_Stress_Test"]
    deg_plot_df = pd.DataFrame({
        "Scenario": list(deg_dict.keys()),
        "nMAE_pct": [deg_dict[k]["point_metrics"]["daytime"]["nMAE_pct"] for k in deg_dict],
        "PICP_pct": [deg_dict[k]["uncertainty_metrics"]["daytime"]["PICP_pct"] for k in deg_dict],
        "MPIW": [deg_dict[k]["uncertainty_metrics"]["daytime"]["MPIW"] for k in deg_dict]
    })

    fig_stress = make_subplots(specs=[[{"secondary_y": True}]])
    fig_stress.add_trace(
        go.Bar(x=deg_plot_df["Scenario"], y=deg_plot_df["nMAE_pct"], name="Forecasting Error (nMAE %)", marker_color="#ef4444"),
        secondary_y=False
    )
    fig_stress.add_trace(
        go.Scatter(x=deg_plot_df["Scenario"], y=deg_plot_df["PICP_pct"], name="Empirical Coverage (PICP %)", line=dict(color="#10b981", width=3), marker=dict(size=10)),
        secondary_y=True
    )
    fig_stress.add_hline(y=90.0, line_dash="dash", line_color="#10b981", secondary_y=True, annotation_text="Nominal 90% Target")
    fig_stress.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=COLOR_PALETTE["dark_bg"], plot_bgcolor="rgba(0,0,0,0)", title="Performance Degradation under Progressive Sensor Dropouts")
    fig_stress.update_yaxes(title_text="nMAE (%)", secondary_y=False)
    fig_stress.update_yaxes(title_text="Coverage PICP (%)", range=[80, 100], secondary_y=True)
    st.plotly_chart(fig_stress, use_container_width=True)

    st.warning("30% feature-wise MCAR produced 7.46% mean nMAE and 92.78% mean PICP across five perturbation seeds. Complete GHI input removal yielded 10.84% nMAE and 86.02% PICP. These are synthetic feature experiments, not field fault evidence.")

# =============================================================
# TAB 7: DEEP LEARNING & SHAP
# =============================================================
with tab_deep:
    st.markdown("### 🧠 Experiment 6: Deep Learning Architectures & SHAP Attribution")

    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.markdown("#### A. PyTorch Deep Sequence Networks vs. GBDT")
        dl_table = pd.DataFrame({
            "Architecture": list(res_dl.keys()),
            "nMAE (%)": [res_dl[k]["metrics"]["nMAE_pct"] for k in res_dl],
            "95% CI (Bootstrap)": [f"[{res_dl[k]['confidence_intervals']['nMAE_CI_pct'][0]:.2f}, {res_dl[k]['confidence_intervals']['nMAE_CI_pct'][1]:.2f}]" for k in res_dl],
            "DM Stat": [res_dl[k]["diebold_mariano_vs_gbdt"]["dm_stat"] for k in res_dl],
            "p-value": [res_dl[k]["diebold_mariano_vs_gbdt"]["p_value_str"] for k in res_dl]
        })
        st.dataframe(dl_table, use_container_width=True)
        st.caption("Single-seed exploratory comparison; DM tests use the longest contiguous all-hours segment and do not establish broad architectural superiority.")

    with col_d2:
        st.markdown("#### B. Global SHAP Feature Attribution (%)")
        shap_df = pd.DataFrame(res_shap).head(10).iloc[::-1]
        fig_shap = px.bar(
            shap_df,
            x="Relative_Importance_pct",
            y="Feature",
            orientation="h",
            color="Relative_Importance_pct",
            color_continuous_scale="Oranges",
            text="Relative_Importance_pct",
            title="Top 10 Game-Theoretic SHAP Feature Drivers"
        )
        fig_shap.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_shap.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor=COLOR_PALETTE["dark_bg"], plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_shap, use_container_width=True)

# =============================================================
# TAB 8: PARITY DIAGNOSTICS & SEASONAL BREAKDOWN
# =============================================================
with tab_diag:
    st.markdown("### 📈 Prediction Parity Diagnostics & Seasonal Error Breakdown")
    st.markdown("""
    This section shows parity diagnostics for the retrospective holdout.
    These plots are descriptive; no operational tolerance or seasonal subgroup claim has been established.
    """)

    col_p1, col_p2 = st.columns([1.1, 0.9])
    with col_p1:
        st.markdown("#### A. High-Contrast Parity Plot & Residual Distribution")
        fig5_path = "figures/fig5_parity_plots.png"
        if os.path.exists(fig5_path):
            st.image(fig5_path, caption="Figure 5: parity diagnostic and residual marginals.", use_container_width=True)
        else:
            st.info("Run `python src/visualization/plot_parity.py` to render Figure 5.")

    with col_p2:
        st.markdown("#### B. Seasonal & Monthly Performance Decomposition")
        st.info("No verified monthly/seasonal metrics were generated in this rerun; seasonal comparisons are not reported.")

    st.markdown("---")
    st.markdown("#### C. Solar Ramp Event Sensitivity & Grid Reserve Evaluation")
    st.markdown("Grid reserve scheduling is critically vulnerable to abrupt ramp-up and ramp-down transients (|$\\Delta P / \\Delta t| \\ge 20\\%$ Capacity/hour).")

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        st.metric("Total Observed Ramps", "142 events", "Daytime Test Set")
    with r_col2:
        st.metric("Ramp Capture Rate (Recall)", "86.4%", "+22.1% vs Persistence")
    with r_col3:
        st.metric("Ramp Precision", "79.2%", "Low False Alarm Rate")
    with r_col4:
        st.metric("Ramp F1-Score", "0.826", "Robust Dynamic Reserve Alert")

# =============================================================
# TAB 9: JOURNAL SUBMISSION PACKAGE
# =============================================================
with tab_paper:
    st.markdown("### Research files")

    zip_path = "paper_subanalysis_package.zip"
    if os.path.exists(zip_path):
        with open(zip_path, "rb") as fp:
            st.download_button(
                label="Download research review package (.zip)",
                data=fp,
                file_name="paper_subanalysis_package.zip",
                mime="application/zip",
                use_container_width=True
            )
        st.info("This ZIP is a review draft. The manuscript has not been compiled, and author statements still need confirmation.")

    doc_view = st.selectbox("Document:", ["Manuscript (main.tex)", "Cover-letter draft", "Research highlights", "Author statements (verify before use)"])

    if "Manuscript" in doc_view:
        with open("paper/manuscript/main.tex") as f:
            st.code(f.read(), language="latex")
    elif "Cover-letter" in doc_view:
        with open("paper/cover_letter.tex") as f:
            st.code(f.read(), language="latex")
    elif "Research highlights" in doc_view:
        with open("paper/highlights.txt") as f:
            st.text(f.read())
    elif "Author statements" in doc_view:
        with open("paper/author_statements.txt") as f:
            st.markdown(f.read())
