"""
app.py
================================================================================
Interactive Streamlit Dashboard: The "Reconstruction Boom" Reality Check
Quantitative Analysis of Post-Disaster & Post-Conflict Equity Returns

Run Locally in VS Code / Terminal:
    streamlit run app.py

Features:
  - Sidebar Controls: Filter by Disaster Type (Natural Disasters vs. Man-Made
    Conflicts), Specific Historical Event, Proxy Tickers (CAT, VMC, FLR,
    Sector Basket, S&P 500), Data Source Mode, and Capital Allocation Simulator.
  - Main Metrics Banner: Displays the "Hype vs. Reality Gap"—contrasting immediate
    3-month post-disaster stock returns against 3-year and 5-year contract
    realization returns, plus benchmark-adjusted S&P 500 Alpha.
  - Interactive Plotly Visualizations:
      1. Dual-axis Aid Stabilization vs. Private Sector Equity Growth Timeline
         with annotated policy/contracting milestones.
      2. Multi-Horizon Return & Alpha Matrix (3-Month, 1-Year, 3-Year, 5-Year).
      3. Cross-Event Regime Comparison (Natural Disasters vs. Man-Made Wars).
      4. Retail Investor Entry Timing Simulator ($10,000+ deployment across T+0,
         T+3M, T+12M, and T+18M entry windows).
================================================================================
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_analysis import (
    HISTORICAL_EVENTS,
    PROXY_TICKERS,
    SECTOR_BASKET_KEY,
    SECTOR_BASKET_META,
    VERIFIED_ANNUAL_RETURNS_PCT,
    build_aid_vs_equity_timeline,
    calculate_horizon_metrics,
    fetch_event_market_data,
    get_cross_event_comparison,
    simulate_entry_timing,
)


# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==============================================================================

st.set_page_config(
    page_title="The Reconstruction Boom: Reality Check",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 10px;
        padding: 14px 16px;
    }
    .insight-callout {
        background-color: rgba(15, 23, 42, 0.65);
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. CACHED DATA WRAPPERS FOR STREAMLIT PERFORMANCE
# ==============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_event_data(event_id: str, prefer_live: bool):
    """Caches normalized price series and timeline data per event."""
    price_df, source_tag = fetch_event_market_data(
        event_id=event_id,
        prefer_live_yfinance=prefer_live,
    )
    timeline_df = build_aid_vs_equity_timeline(event_id, price_df)
    metrics_df = calculate_horizon_metrics(price_df)
    return price_df, timeline_df, metrics_df, source_tag


@st.cache_data(ttl=3600, show_spinner=False)
def load_cached_cross_event_summary(prefer_live: bool) -> pd.DataFrame:
    """Caches cross-event comparison metrics across all 4 historical events."""
    return get_cross_event_comparison(prefer_live_yfinance=prefer_live)


# ==============================================================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==============================================================================

with st.sidebar:
    st.title("🏗️ Reconstruction Analytics")
    st.caption(
        "Tracking the macroeconomic lag between sovereign disaster aid "
        "and private-sector stock returns."
    )
    st.divider()

    # 1. Disaster Type Selector
    disaster_type_choice = st.radio(
        "1. Select Disaster Regime:",
        options=["All Regimes", "Natural Disaster", "Man-Made Conflict"],
        index=0,
        help="Filter historical case studies by sudden natural shocks vs. prolonged geopolitical conflicts.",
    )

    # Filter available events based on selected disaster type
    filtered_event_ids = [
        ev_id
        for ev_id, ev in HISTORICAL_EVENTS.items()
        if disaster_type_choice == "All Regimes" or ev.disaster_type == disaster_type_choice
    ]

    # 2. Specific Event Selector
    selected_event_id = st.selectbox(
        "2. Select Historical Event:",
        options=filtered_event_ids,
        format_func=lambda eid: HISTORICAL_EVENTS[eid].name,
        index=0,
        help="Choose a specific post-disaster or post-conflict rebuilding window to inspect.",
    )
    active_event = HISTORICAL_EVENTS[selected_event_id]

    st.divider()

    # 3. Ticker & Benchmark Selection
    st.subheader("📊 Portfolio & Ticker Controls")
    selected_tickers = st.multiselect(
        "Select Reconstruction Proxies:",
        options=["CAT", "VMC", "FLR"],
        default=["CAT", "VMC", "FLR"],
        format_func=lambda t: f"{t} — {PROXY_TICKERS[t]['name']}",
        help="CAT = Heavy Machinery, VMC = Aggregates/Materials, FLR = EPC Infrastructure.",
    )

    show_sector_basket = st.checkbox(
        "Show Equal-Weight Reconstruction Basket",
        value=True,
        help="Displays the composite average of CAT, VMC, and FLR.",
    )
    show_aid_curves = st.checkbox(
        "Overlay Public Aid & Outlay S-Curves",
        value=True,
        help="Plots cumulative government aid obligation vs. actual physical contract outlays.",
    )
    show_milestones = st.checkbox(
        "Annotate Institutional Milestones",
        value=True,
        help="Adds vertical markers for major legislative appropriations and contract awards.",
    )

    st.divider()

    # 4. Data Source Mode & Capital Simulator Input
    st.subheader("⚙️ Engine & Simulator Settings")
    data_mode = st.selectbox(
        "Market Data Source:",
        options=[
            "Auto (Live yfinance + Verified Fallback)",
            "Verified Historical Benchmark Only",
        ],
        index=0,
        help="Attempts live Yahoo Finance API fetch first, falling back to verified historical prices if offline.",
    )
    prefer_live_yf = data_mode.startswith("Auto")

    sim_capital = st.number_input(
        "Simulated Retail Capital ($ USD):",
        min_value=1000.0,
        max_value=1000000.0,
        value=10000.0,
        step=1000.0,
        help="Initial investment amount for the Retail Entry Timing Simulator.",
    )


# ==============================================================================
# 4. LOAD EVENT DATA & COMPUTE METRICS
# ==============================================================================

price_df, timeline_df, metrics_df, data_source_tag = load_cached_event_data(
    selected_event_id,
    prefer_live=prefer_live_yf,
)
cross_summary_df = load_cached_cross_event_summary(prefer_live=False)

basket_metrics = metrics_df[metrics_df["Ticker"] == SECTOR_BASKET_KEY].iloc[0]
sp500_metrics = metrics_df[metrics_df["Ticker"] == "^GSPC"].iloc[0]


# ==============================================================================
# 5. HEADER & EVENT CONTEXT BANNER
# ==============================================================================

st.title("🏗️ The 'Reconstruction Boom' Reality Check")
st.markdown(
    f"**Active Case Study:** `{active_event.name}` &nbsp;|&nbsp; "
    f"**Regime:** `{active_event.disaster_type}` &nbsp;|&nbsp; "
    f"**Event Baseline ($T_0$):** `{active_event.start_date}` &nbsp;|&nbsp; "
    f"**Analysis Window End:** `{active_event.end_date}`"
)

# Top Macro Context Cards
macro_col1, macro_col2, macro_col3, macro_col4 = st.columns(4)
with macro_col1:
    st.metric(
        label="Estimated Total Damage",
        value=f"${active_event.estimated_damage_usd_bn:,.1f}B",
        delta=active_event.region.split("(")[0].strip(),
        delta_color="off",
    )
with macro_col2:
    st.metric(
        label="Public Aid / Sovereign Commitment",
        value=f"${active_event.public_aid_committed_usd_bn:,.1f}B",
        delta="Wave 1: Public Stabilization",
        delta_color="off",
    )
with macro_col3:
    st.metric(
        label="Private / NFIP Insurance Payouts",
        value=f"${active_event.private_insurance_payout_usd_bn:,.1f}B",
        delta="Fast Liquidity (Natural) vs. Excluded (War)",
        delta_color="off",
    )
with macro_col4:
    st.metric(
        label="Aid-to-Revenue Recognition Lag",
        value=f"{active_event.aid_to_revenue_lag_months} Months",
        delta=f"Optimal Entry: {active_event.optimal_entry_window_months[:18]}...",
        delta_color="off",
    )

st.markdown(
    f"""
    <div class="insight-callout">
        <strong>Macroeconomic Reality Check:</strong> {active_event.summary_narrative}<br>
        <small><strong>Institutional Transmission Mechanism:</strong> {active_event.institutional_mechanism}</small>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 6. MAIN METRICS: THE "HYPE VS. REALITY" GAP
# ==============================================================================

st.subheader("📉 The 'Hype vs. Reality' Performance Gap (Reconstruction Sector Basket vs. S&P 500)")
st.caption(
    "Comparing immediate 3-month 'Social Media / Cable News Hype' returns against "
    "1-year digestion, 3-year contract realization, and 5-year full-cycle outcomes."
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        label="3-Month 'Hype Phase' Return",
        value=f"{basket_metrics['3M_Return_Pct']:+.2f}%",
        delta=f"{basket_metrics['3M_Alpha_Pct']:+.2f}% Alpha vs. S&P 500 ({sp500_metrics['3M_Return_Pct']:+.1f}%)",
    )
with kpi2:
    st.metric(
        label="1-Year 'Digestion' Return",
        value=f"{basket_metrics['1Y_Return_Pct']:+.2f}%",
        delta=f"{basket_metrics['1Y_Alpha_Pct']:+.2f}% Alpha vs. S&P 500 ({sp500_metrics['1Y_Return_Pct']:+.1f}%)",
    )
with kpi3:
    st.metric(
        label="3-Year 'Contract Realization'",
        value=f"{basket_metrics['3Y_Return_Pct']:+.2f}%",
        delta=f"{basket_metrics['3Y_Alpha_Pct']:+.2f}% Alpha vs. S&P 500 ({sp500_metrics['3Y_Return_Pct']:+.1f}%)",
    )
with kpi4:
    st.metric(
        label="Hype-to-Reality Gap (3Y − 3M)",
        value=f"{basket_metrics['Hype_vs_Reality_Gap_Pct']:+.2f}%",
        delta=f"{basket_metrics['Alpha_Realization_Gap_Pct']:+.2f}% Excess Alpha Expansion",
    )
with kpi5:
    st.metric(
        label="5-Year Full-Cycle Return",
        value=f"{basket_metrics['5Y_Return_Pct']:+.2f}%",
        delta=f"{basket_metrics['5Y_Alpha_Pct']:+.2f}% Alpha vs. S&P 500 ({sp500_metrics['5Y_Return_Pct']:+.1f}%)",
    )


# ==============================================================================
# 7. INTERACTIVE PLOTLY VISUALIZATIONS (TABBED WORKSPACE)
# ==============================================================================

tab_timeline, tab_horizons, tab_regimes, tab_simulator = st.tabs([
    "📈 1. Aid Stabilization vs. Equity Timeline",
    "📊 2. Horizon Returns & Alpha Matrix",
    "🌍 3. Natural vs. Man-Made Regime Comparison",
    "⏱️ 4. Retail Entry Timing Simulator",
])


# ------------------------------------------------------------------------------
# TAB 1: TIME-SERIES CHART (PUBLIC AID STABILIZATION VS. PRIVATE EQUITY GROWTH)
# ------------------------------------------------------------------------------
with tab_timeline:
    st.markdown(
        "#### Sovereign Aid Stabilization vs. Private Sector Equity Growth ($T_0 = 100$)"
    )
    st.caption(
        "Solid lines track normalized stock performance (left axis, $100 indexed at $T_0$). "
        "Dashed/shaded curves on the secondary axis (right) model the progression from "
        "legislative aid obligation (Wave 1) to physical EPC contract outlays (Wave 2)."
    )

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Plot S&P 500 Baseline
    fig_ts.add_trace(
        go.Scatter(
            x=timeline_df.index,
            y=timeline_df["^GSPC"],
            name="S&P 500 Baseline (^GSPC)",
            line=dict(color=PROXY_TICKERS["^GSPC"]["color"], width=2.2, dash="dot"),
            hovertemplate="<b>S&P 500 (^GSPC)</b><br>Date: %{x|%Y-%m-%d}<br>Normalized Index: %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # 2. Plot Selected Proxy Tickers (CAT, VMC, FLR)
    for ticker in selected_tickers:
        if ticker in timeline_df.columns:
            fig_ts.add_trace(
                go.Scatter(
                    x=timeline_df.index,
                    y=timeline_df[ticker],
                    name=f"{ticker} ({PROXY_TICKERS[ticker]['name']})",
                    line=dict(color=PROXY_TICKERS[ticker]["color"], width=2.2),
                    hovertemplate=(
                        f"<b>{ticker} — {PROXY_TICKERS[ticker]['name']}</b><br>"
                        "Date: %{x|%Y-%m-%d}<br>Normalized Value: $%{y:.2f}<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

    # 3. Plot Equal-Weight Reconstruction Sector Basket
    if show_sector_basket and SECTOR_BASKET_KEY in timeline_df.columns:
        fig_ts.add_trace(
            go.Scatter(
                x=timeline_df.index,
                y=timeline_df[SECTOR_BASKET_KEY],
                name="Reconstruction Sector Basket (Equal-Weight)",
                line=dict(color=SECTOR_BASKET_META["color"], width=3.4),
                hovertemplate=(
                    "<b>Reconstruction Sector Basket</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>Normalized Value: $%{y:.2f}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

    # 4. Plot Aid Obligation & Physical Contract Outlay Curves on Secondary Y-Axis
    if show_aid_curves:
        fig_ts.add_trace(
            go.Scatter(
                x=timeline_df.index,
                y=timeline_df["Public_Aid_Obligated_Pct"],
                name="Wave 1: Public Aid / Insurance Obligated (%)",
                line=dict(color="rgba(56, 189, 248, 0.75)", width=1.8, dash="dash"),
                hovertemplate=(
                    "<b>Public Aid / Insurance Obligated</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>Committed Share: %{y:.1f}%<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=timeline_df.index,
                y=timeline_df["Physical_Contract_Outlay_Pct"],
                name="Wave 2: Physical EPC Contract Outlays (%)",
                line=dict(color="rgba(16, 185, 129, 0.85)", width=2.0, dash="longdash"),
                fill="tozeroy",
                fillcolor="rgba(16, 185, 129, 0.08)",
                hovertemplate=(
                    "<b>Physical EPC Contract Outlays</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>Disbursed Share: %{y:.1f}%<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

    # 5. Shade the initial 90-day "Social Media / Retail Hype Zone"
    hype_end_idx = min(63, len(timeline_df) - 1)
    fig_ts.add_vrect(
        x0=timeline_df.index[0],
        x1=timeline_df.index[hype_end_idx],
        fillcolor="rgba(239, 68, 68, 0.12)",
        opacity=0.8,
        layer="below",
        line_width=0,
        annotation_text="90-Day Retail Hype Zone",
        annotation_position="top left",
    )

    # 6. Annotate Institutional & Contracting Milestones
    if show_milestones:
        start_ts = timeline_df.index[0]
        end_ts = timeline_df.index[-1]
        for idx_m, ms in enumerate(active_event.milestones):
            ms_date = start_ts + pd.DateOffset(months=ms.months_post_event)
            if start_ts <= ms_date <= end_ts:
                y_pos = 0.92 - (idx_m % 3) * 0.14
                fig_ts.add_vline(
                    x=ms_date,
                    line_width=1.2,
                    line_dash="dot",
                    line_color="rgba(226, 232, 240, 0.45)",
                )
                fig_ts.add_annotation(
                    x=ms_date,
                    y=y_pos,
                    yref="paper",
                    text=f"<b>M+{ms.months_post_event}:</b> {ms.label}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-20,
                    bgcolor="rgba(15, 23, 42, 0.85)",
                    bordercolor="rgba(148, 163, 184, 0.4)",
                    font=dict(size=10, color="#F8FAFC"),
                )

    fig_ts.update_layout(
        height=540,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig_ts.update_yaxes(
        title_text="Normalized Equity Index ($100 at Event T0)",
        secondary_y=False,
        gridcolor="rgba(148, 163, 184, 0.12)",
    )
    fig_ts.update_yaxes(
        title_text="Cumulative Aid Obligated / Outlaid (%)",
        range=[0, 105],
        secondary_y=True,
        showgrid=False,
    )

    st.plotly_chart(fig_ts, use_container_width=True)

    # Milestone Log Table under the chart
    with st.expander("📋 View Verified Institutional & Contracting Milestones for This Event", expanded=True):
        ms_df = pd.DataFrame([
            {
                "Months Post-Event": f"Month +{m.months_post_event}",
                "Phase Category": m.category,
                "Milestone": m.label,
                "Analytical Detail & Revenue Impact": m.detail,
            }
            for m in active_event.milestones
        ])
        st.dataframe(ms_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 2: HORIZON RETURNS & BENCHMARK ALPHA MATRIX
# ------------------------------------------------------------------------------
with tab_horizons:
    st.markdown("#### Multi-Horizon Return & S&P 500 Alpha Breakdown (3M vs. 1Y vs. 3Y vs. 5Y)")
    col_mode, _ = st.columns([2, 3])
    with col_mode:
        bar_metric_mode = st.radio(
            "Select Chart Display Mode:",
            options=[
                "Cumulative Total Return (%)",
                "Excess Return / Alpha vs. S&P 500 (%)",
            ],
            horizontal=True,
        )

    is_alpha = bar_metric_mode.startswith("Excess")
    horizon_cols = (
        ["3M_Alpha_Pct", "1Y_Alpha_Pct", "3Y_Alpha_Pct", "5Y_Alpha_Pct"]
        if is_alpha
        else ["3M_Return_Pct", "1Y_Return_Pct", "3Y_Return_Pct", "5Y_Return_Pct"]
    )
    horizon_labels = [
        "3-Month (Hype Window)",
        "1-Year (Digestion)",
        "3-Year (Contract Realization)",
        "5-Year (Full Capital Cycle)",
    ]

    fig_bars = go.Figure()
    plot_df = metrics_df if not is_alpha else metrics_df[metrics_df["Ticker"] != "^GSPC"]

    for _, row in plot_df.iterrows():
        tkr = row["Ticker"]
        color = (
            PROXY_TICKERS[tkr]["color"]
            if tkr in PROXY_TICKERS
            else SECTOR_BASKET_META["color"]
        )
        y_vals = [row[c] for c in horizon_cols]
        fig_bars.add_trace(
            go.Bar(
                name=f"{tkr} — {row['Company / Index']}",
                x=horizon_labels,
                y=y_vals,
                marker_color=color,
                text=[f"{v:+.1f}%" for v in y_vals],
                textposition="outside",
            )
        )

    fig_bars.update_layout(
        barmode="group",
        height=460,
        template="plotly_dark",
        yaxis_title="Benchmark-Adjusted Alpha (%)" if is_alpha else "Cumulative Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    st.markdown("##### Quantitative Scorecard for Selected Event")
    display_cols = {
        "Ticker": "Ticker",
        "Company / Index": "Company / Index",
        "Subsector": "Sector Role",
        "3M_Return_Pct": "3M Return (%)",
        "1Y_Return_Pct": "1Y Return (%)",
        "3Y_Return_Pct": "3Y Return (%)",
        "5Y_Return_Pct": "5Y Return (%)",
        "3M_Alpha_Pct": "3M Alpha (%)",
        "3Y_Alpha_Pct": "3Y Alpha (%)",
        "Hype_vs_Reality_Gap_Pct": "3Y−3M Gap (%)",
        "Annualized_CAGR_Pct": "5Y CAGR (%)",
        "Max_Drawdown_Pct": "Max Drawdown (%)",
    }
    formatted_table = metrics_df[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(formatted_table, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 3: NATURAL DISASTERS VS. MAN-MADE CONFLICTS REGIME COMPARISON
# ------------------------------------------------------------------------------
with tab_regimes:
    st.markdown("#### Structural Comparison: Natural Disasters vs. Man-Made Conflicts")
    st.caption(
        "Contrasting aid-to-revenue lag, private insurance liquidity, and 3-Month vs. 3-Year "
        "Reconstruction Basket performance across all four historical case studies."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        fig_gap = go.Figure()
        fig_gap.add_trace(
            go.Bar(
                name="3-Month Hype Return (%)",
                x=cross_summary_df["Event"],
                y=cross_summary_df["Basket_3M_Return_Pct"],
                marker_color="#EF4444",
                text=[f"{v:+.1f}%" for v in cross_summary_df["Basket_3M_Return_Pct"]],
                textposition="outside",
            )
        )
        fig_gap.add_trace(
            go.Bar(
                name="1-Year Digestion Return (%)",
                x=cross_summary_df["Event"],
                y=cross_summary_df["Basket_1Y_Return_Pct"],
                marker_color="#F59E0B",
                text=[f"{v:+.1f}%" for v in cross_summary_df["Basket_1Y_Return_Pct"]],
                textposition="outside",
            )
        )
        fig_gap.add_trace(
            go.Bar(
                name="3-Year Contract Realization Return (%)",
                x=cross_summary_df["Event"],
                y=cross_summary_df["Basket_3Y_Return_Pct"],
                marker_color="#10B981",
                text=[f"{v:+.1f}%" for v in cross_summary_df["Basket_3Y_Return_Pct"]],
                textposition="outside",
            )
        )
        fig_gap.update_layout(
            title="Reconstruction Basket: 3M Hype vs. 1Y vs. 3Y Realization",
            barmode="group",
            height=430,
            template="plotly_dark",
            yaxis_title="Cumulative Return (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    with col_right:
        fig_lag = go.Figure()
        for _, r in cross_summary_df.iterrows():
            bubble_color = "#3B82F6" if r["Disaster_Type"] == "Natural Disaster" else "#EC4899"
            fig_lag.add_trace(
                go.Scatter(
                    x=[r["Aid_to_Revenue_Lag_Months"]],
                    y=[r["Hype_vs_Reality_Gap_Pct"]],
                    mode="markers+text",
                    name=r["Event"],
                    text=[r["Event"].split("(")[0].strip()],
                    textposition="top center",
                    marker=dict(
                        size=max(r["Damage_USD_Bn"] / 8.0, 22),
                        color=bubble_color,
                        opacity=0.85,
                        line=dict(width=2, color="#F8FAFC"),
                    ),
                    hovertemplate=(
                        f"<b>{r['Event']}</b><br>"
                        f"Regime: {r['Disaster_Type']}<br>"
                        f"Aid-to-Revenue Lag: {r['Aid_to_Revenue_Lag_Months']} Months<br>"
                        f"3Y−3M Return Spread: {r['Hype_vs_Reality_Gap_Pct']:+.1f}%<br>"
                        f"Estimated Damage: ${r['Damage_USD_Bn']:.1f}B<extra></extra>"
                    ),
                )
            )
        fig_lag.update_layout(
            title="Aid-to-Revenue Lag (Months) vs. 3Y−3M Realization Spread (Bubble = Damage $B)",
            height=430,
            template="plotly_dark",
            xaxis_title="Aid-to-Revenue Recognition Lag (Months Post-Event)",
            yaxis_title="Hype-to-Reality Return Spread (3Y minus 3M, %)",
            showlegend=False,
        )
        st.plotly_chart(fig_lag, use_container_width=True)

    st.markdown("##### Cross-Event Macroeconomic & Equity Summary Table")
    st.dataframe(
        cross_summary_df[[
            "Event",
            "Disaster_Type",
            "Damage_USD_Bn",
            "Public_Aid_USD_Bn",
            "Insurance_Payout_USD_Bn",
            "Aid_to_Revenue_Lag_Months",
            "Basket_3M_Return_Pct",
            "Basket_3Y_Return_Pct",
            "Basket_5Y_Return_Pct",
            "Hype_vs_Reality_Gap_Pct",
            "Optimal_Entry_Window",
        ]],
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------------------------------------------
# TAB 4: RETAIL INVESTOR ENTRY TIMING SIMULATOR
# ------------------------------------------------------------------------------
with tab_simulator:
    st.markdown(
        f"#### Retail Entry Timing Simulator — Deploying ${sim_capital:,.0f} in `{active_event.name}`"
    )
    st.caption(
        "Evaluates how entry timing (buying on Day 1 social media hype vs. waiting for post-hype "
        "pullbacks or institutional procurement mobilization) impacts annualized CAGR and max drawdown."
    )

    sim_col1, sim_col2 = st.columns([1, 3])
    with sim_col1:
        sim_ticker = st.selectbox(
            "Select Target Asset for Simulation:",
            options=[SECTOR_BASKET_KEY, "CAT", "VMC", "FLR"],
            format_func=lambda t: (
                SECTOR_BASKET_META["name"]
                if t == SECTOR_BASKET_KEY
                else f"{t} — {PROXY_TICKERS[t]['name']}"
            ),
        )
        st.info(
            f"**Recommended Entry Window for {active_event.name}:**\n\n"
            f"🗓️ `{active_event.optimal_entry_window_months}`\n\n"
            "Waiting for legislative appropriation and formal agency procurement "
            "avoids early supply-chain drawdowns and dead-money opportunity costs."
        )

    sim_df = simulate_entry_timing(
        price_df=price_df,
        ticker=sim_ticker,
        capital_usd=sim_capital,
    )

    with sim_col2:
        fig_sim = go.Figure()
        fig_sim.add_trace(
            go.Bar(
                name="Annualized CAGR (%)",
                x=sim_df["Entry_Phase"],
                y=sim_df["Annualized_CAGR_Pct"],
                marker_color="#10B981",
                text=[f"{v:+.2f}%/yr" for v in sim_df["Annualized_CAGR_Pct"]],
                textposition="outside",
            )
        )
        fig_sim.add_trace(
            go.Bar(
                name="Post-Entry Max Drawdown (%)",
                x=sim_df["Entry_Phase"],
                y=sim_df["Post_Entry_Max_Drawdown_Pct"],
                marker_color="#EF4444",
                text=[f"{v:.1f}% DD" for v in sim_df["Post_Entry_Max_Drawdown_Pct"]],
                textposition="outside",
            )
        )
        fig_sim.update_layout(
            title=f"Annualized CAGR vs. Post-Entry Drawdown by Entry Timing ({sim_ticker})",
            barmode="group",
            height=380,
            template="plotly_dark",
            yaxis_title="Percentage (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    st.dataframe(sim_df, use_container_width=True, hide_index=True)


# ==============================================================================
# 8. VERIFIED HISTORICAL ANNUAL RETURNS & PRIMARY SOURCES FOOTER
# ==============================================================================

st.divider()
foot_col1, foot_col2 = st.columns([3, 2])

with foot_col1:
    with st.expander("📚 Primary Institutional & Financial Data Sources (Clickable References)", expanded=False):
        for title, url in active_event.primary_sources:
            st.markdown(f"- [{title}]({url})")
        st.markdown(
            "- [S&P 500 Historical Annual Total Returns (Slickcharts)](https://www.slickcharts.com/sp500/returns)\n"
            "- [Caterpillar Inc. (CAT) Historical Annual Returns (Slickcharts)](https://www.slickcharts.com/symbol/CAT/returns)\n"
            "- [Vulcan Materials Co. (VMC) Historical Annual Returns (Slickcharts)](https://www.slickcharts.com/symbol/VMC/returns)\n"
            "- [Fluor Corporation (FLR) Historical Stock Performance (CompaniesMarketCap)](https://companiesmarketcap.com/fluor-corporation/stock-price-history/)"
        )

with foot_col2:
    with st.expander("📅 Verified Calendar-Year Returns (2002–2026)", expanded=False):
        ann_df = pd.DataFrame.from_dict(VERIFIED_ANNUAL_RETURNS_PCT, orient="index")
        ann_df.index.name = "Year"
        st.dataframe(ann_df.sort_index(ascending=False), use_container_width=True)

st.caption(
    f"Data Engine Status: `{data_source_tag}` | Educational & Quantitative Research Dashboard — "
    "Not individualized investment advice."
)