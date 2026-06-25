import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="RailPredict",
    page_icon="🚆",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding: 3rem 2.5rem 2rem 2.5rem;
}

.hero-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #f97316;
    margin-bottom: 6px;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
    margin-bottom: 8px;
}

.hero-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 0;
}

.divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 1.5rem 0;
}

.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 12px;
}

[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1e293b;
}

[data-testid="stSidebar"] .stButton button {
    background: #f97316;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    width: 100%;
    padding: 0.6rem;
    font-size: 0.95rem;
    margin-top: 0.5rem;
    transition: opacity 0.2s;
}

[data-testid="stSidebar"] .stButton button:hover {
    opacity: 0.85;
}

[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #f97316, #fb923c);
    border-radius: 99px;
}

.stProgress > div {
    background: #1e293b;
    border-radius: 99px;
}

.stDataFrame {
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}

thead tr th {
    background: #0f172a !important;
    color: #94a3b8 !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("train_eta_model.pkl")
schedule = pd.read_csv("12488 schedule.csv")
display_schedule = pd.read_csv("12488_stops.csv")

with st.sidebar:
    st.markdown('<div class="hero-label">Train 12488</div>', unsafe_allow_html=True)
    st.markdown("**ANVT → Jaynagar**")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    train_no = st.text_input("Train No.", value="12488")

    current_station = st.selectbox(
        "At Station",
        display_schedule["sttn_code"].tolist()
    )

    current_delay = st.number_input(
        "Delay (min)",
        value=10.0,
        step=1.0
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("Run Prediction →")

st.markdown('<div class="hero-label">RailPredict</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🚆 Train ETA Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Predicts delay propagation across stations based on historical journey data.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

if not predict:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 0;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🛤</div>
        <div style="font-size: 1rem; font-weight: 500; color: #475569;">Select a station and enter delay to get started.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

current_seq = schedule.loc[
    schedule["sttn_code"] == current_station, "seq_number"
].iloc[0]

total_stations = display_schedule["seq_number"].max()
progress = min(current_seq / total_stations, 1.0)

start_station = display_schedule.iloc[0]["sttn_code"]
end_station = display_schedule.iloc[-1]["sttn_code"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Train", train_no)
with col2:
    st.metric("At Station", current_station)
with col3:
    st.metric("Current Delay", f"{current_delay:.0f} min")
with col4:
    st.metric("Journey", f"{progress * 100:.0f}% done")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

st.markdown('<div class="section-label">Route Progress</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([1, 6, 1])
with col_a:
    st.caption(f"🚉 {start_station}")
with col_b:
    st.progress(progress)
with col_c:
    st.caption(f"🏁 {end_station}")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

future_stations = schedule[
    schedule["seq_number"] > current_seq
][[
    "seq_number", "sttn_code", "wtt_arvl_time_sec",
    "cum_dist", "intr_dist", "run_time", "block_section_speed"
]].sort_values("seq_number").reset_index(drop=True)

if future_stations.empty:
    st.warning("Train has reached its destination.")
    st.stop()

total_distance = schedule["cum_dist"].max()
input_delay = current_delay
prev_delay = current_delay
results = []

for _, row in future_stations.iterrows():
    sample = pd.DataFrame([{
        "seq_number": row["seq_number"],
        "cum_dist": row["cum_dist"],
        "intr_dist": row["intr_dist"],
        "run_time": row["run_time"],
        "block_section_speed": row["block_section_speed"],
        "delay_min_fixed": input_delay,
        "prev_delay": prev_delay,
        "distance_remaining": total_distance - row["cum_dist"]
    }])

    predicted_delay = round(np.clip(model.predict(sample)[0], -30, 180), 2)
    eta_sec = row["wtt_arvl_time_sec"] + predicted_delay * 60
    hours = int((eta_sec % 86400) // 3600)
    minutes = int((eta_sec % 3600) // 60)

    results.append([row["sttn_code"], round(predicted_delay, 2), f"{hours:02d}:{minutes:02d}"])

    prev_delay = predicted_delay

eta_df = pd.DataFrame(results, columns=["Station", "Predicted Delay (min)", "Predicted ETA"])
eta_df = eta_df[eta_df["Station"].isin(display_schedule["sttn_code"])].reset_index(drop=True)

st.markdown('<div class="section-label">Delay Map</div>', unsafe_allow_html=True)

route_df = future_stations[future_stations["sttn_code"].isin(display_schedule["sttn_code"])].copy()
route_df = route_df.merge(eta_df, left_on="sttn_code", right_on="Station", how="inner")
route_df["x"] = range(len(route_df))
route_df["y"] = 0

colors = []
for delay in route_df["Predicted Delay (min)"]:
    if delay < 0:
        colors.append("#22c55e")
    elif delay <= 5:
        colors.append("#eab308")
    elif delay <= 15:
        colors.append("#f97316")
    else:
        colors.append("#ef4444")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=route_df["x"], y=route_df["y"],
    mode="lines",
    line=dict(color="#1e293b", width=6),
    hoverinfo="skip", showlegend=False
))

fig.add_trace(go.Scatter(
    x=route_df["x"], y=route_df["y"],
    mode="markers+text",
    marker=dict(size=26, color=colors, line=dict(color="#0f172a", width=2)),
    text=route_df["sttn_code"],
    textposition="top center",
    textfont=dict(size=10, color="#94a3b8"),
    customdata=route_df[["Predicted ETA", "Predicted Delay (min)"]],
    hovertemplate="<b>%{text}</b><br>ETA: %{customdata[0]}<br>Delay: %{customdata[1]:.1f} min<extra></extra>",
    showlegend=False
))

fig.update_layout(
    height=300,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 0.8]),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8"),
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="display:flex; gap:1.5rem; font-size:12px; color:#64748b; margin-top:-10px; margin-bottom:1rem;">
    <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;margin-right:5px;"></span>Early</span>
    <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#eab308;margin-right:5px;"></span>0–5 min</span>
    <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#f97316;margin-right:5px;"></span>5–15 min</span>
    <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:5px;"></span>&gt;15 min</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

left, right = st.columns([3, 1])

with left:
    st.markdown('<div class="section-label">Station-by-Station Predictions</div>', unsafe_allow_html=True)
    st.dataframe(eta_df, use_container_width=True, hide_index=True)

with right:
    last = eta_df.iloc[-1]
    st.markdown('<div class="section-label">Final Destination</div>', unsafe_allow_html=True)
    st.metric("Station", last["Station"])
    st.metric("ETA", last["Predicted ETA"])
    delay_val = last["Predicted Delay (min)"]
    if delay_val < 0:
        badge_color = "#22c55e"
        badge_text = f"Early by {abs(delay_val):.0f} min"
    elif delay_val <= 5:
        badge_color = "#eab308"
        badge_text = f"{delay_val:.0f} min late"
    elif delay_val <= 15:
        badge_color = "#f97316"
        badge_text = f"{delay_val:.0f} min late"
    else:
        badge_color = "#ef4444"
        badge_text = f"{delay_val:.0f} min late"
    st.markdown(f"""
    <div style="margin-top:8px;">
        <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Status</div>
        <span style="background:{badge_color}22;color:{badge_color};padding:4px 12px;border-radius:99px;font-size:13px;font-weight:600;border:1px solid {badge_color}44;">
            {badge_text}
        </span>
    </div>
    """, unsafe_allow_html=True)