import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Settings ----------------
st.set_page_config(
    page_title="AI Train ETA Prediction",
    page_icon="🚆",
    layout="wide"
)

# ---------------- Load Files ----------------
model = joblib.load("train_eta_model.pkl")

# Full schedule for ML prediction
schedule = pd.read_csv("12488 schedule.csv")

# Only commercial stops for display
display_schedule = pd.read_csv("12488_stops.csv")

# ---------------- Sidebar ----------------
st.sidebar.title("🚆 Train Details")

train_no = st.sidebar.text_input("Train Number", value="12488")

current_station = st.sidebar.selectbox(
    "Current Station",
    display_schedule["sttn_code"].tolist()
)

current_delay = st.sidebar.number_input(
    "Current Delay (minutes)",
    value=10.0,
    step=1.0
)

predict = st.sidebar.button("🚆 Predict ETA")

# ---------------- Main Page ----------------
st.title("🚆 AI Train ETA Prediction System")
st.subheader("Centre for Railway Information Systems (CRIS)")
st.write("Welcome to the AI-powered Train ETA Prediction Dashboard.")

# ---------------- Prediction ----------------
if predict:

    st.success("Prediction Started Successfully!")

    st.subheader("📍 Current Train Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Train Number", train_no)

    with col2:
        st.metric("Current Station", current_station)

    with col3:
        st.metric("Input Delay", f"{current_delay:.2f} min")

    # Current station sequence
    current_seq = schedule.loc[
        schedule["sttn_code"] == current_station,
        "seq_number"
    ].iloc[0]

    # ---------------- Route Progress ----------------
    total_stations = display_schedule["seq_number"].max()

    progress = current_seq / total_stations

    st.subheader("🛤 Route Progress")

    st.progress(progress)
    st.caption(f"Journey Completed: {progress * 100:.1f}%")

    start_station = display_schedule.iloc[0]["sttn_code"]
    end_station = display_schedule.iloc[-1]["sttn_code"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write(f"🚉 Start: **{start_station}**")

    with c2:
        st.write(f"📍 Current: **{current_station}**")

    with c3:
        st.write(f"🏁 Destination: **{end_station}**")

    # Future stations from full schedule
    future_stations = schedule[
        schedule["seq_number"] > current_seq
    ][[
        "seq_number",
        "sttn_code",
        "wtt_arvl_time_sec",
        "cum_dist",
        "intr_dist",
        "run_time",
        "block_section_speed"
    ]]

    if future_stations.empty:
        st.warning("🚆 Train has already reached the destination.")
        st.stop()

    total_distance = schedule["cum_dist"].max()

    prev_delay = current_delay
    results = []

    # ETA Prediction Loop
    for _, row in future_stations.iterrows():

        sample = pd.DataFrame([{
            "seq_number": row["seq_number"],
            "cum_dist": row["cum_dist"],
            "intr_dist": row["intr_dist"],
            "run_time": row["run_time"],
            "block_section_speed": row["block_section_speed"],
            "delay_min_fixed": current_delay,
            "prev_delay": prev_delay,
            "distance_remaining": total_distance - row["cum_dist"]
        }])

        predicted_delay = model.predict(sample)[0]

        eta_sec = row["wtt_arvl_time_sec"] + predicted_delay * 60

        hours = int((eta_sec % 86400) // 3600)
        minutes = int((eta_sec % 3600) // 60)

        results.append([
            row["sttn_code"],
            round(predicted_delay, 2),
            f"{hours:02d}:{minutes:02d}"
        ])

        prev_delay = current_delay
        current_delay = predicted_delay

    eta_df = pd.DataFrame(
        results,
        columns=[
            "Station",
            "Predicted Delay (min)",
            "Predicted ETA"
        ]
    )

    # Show only commercial stops
    eta_df = eta_df[
        eta_df["Station"].isin(display_schedule["sttn_code"])
    ].reset_index(drop=True)

    st.subheader("🚉 Upcoming Station Predictions")
    st.dataframe(eta_df, use_container_width=True)
    st.subheader("🚆 Railway Route Delay Map")

    # Complete route from current station onwards (commercial stops only)
    route_df = future_stations[
        future_stations["sttn_code"].isin(display_schedule["sttn_code"])
    ].copy()

    # Add predicted values
    route_df = route_df.merge(
        eta_df,
        left_on="sttn_code",
        right_on="Station",
        how="inner"
    )

    # X-axis positions
    route_df["x"] = range(len(route_df))
    route_df["y"] = 0

    # Assign colors based on delay
    colors = []

    for delay in route_df["Predicted Delay (min)"]:

        if delay < 0:
            colors.append("green")

        elif delay <= 5:
            colors.append("yellow")

        elif delay <= 15:
            colors.append("orange")

        else:
            colors.append("red")

    # Create railway figure
    fig = go.Figure()

    # Railway Track
    fig.add_trace(go.Scatter(
        x=route_df["x"],
        y=route_df["y"],
        mode="lines",
        line=dict(color="gray", width=8),
        hoverinfo="skip",
        showlegend=False
    ))

    # Stations
    fig.add_trace(go.Scatter(
        x=route_df["x"],
        y=route_df["y"],
        mode="markers+text",
        marker=dict(
            size=28,
            color=colors,
            line=dict(color="black", width=1)
        ),
        text=route_df["sttn_code"],
        textposition="top center",
        textfont=dict(size=11),
        customdata=route_df[["Predicted ETA", "Predicted Delay (min)"]],
        hovertemplate=
            "<b>%{text}</b><br>" +
            "ETA: %{customdata[0]}<br>" +
            "Delay: %{customdata[1]:.2f} min<extra></extra>",
        showlegend=False
    ))

    # Train emoji marker at current station (if visible on display route)
    current_index = route_df.index[
        route_df["sttn_code"] == current_station
    ].tolist()

    if current_index:
        train_x = route_df.loc[current_index[0], "x"]

        fig.add_trace(go.Scatter(
            x=[train_x],
            y=[0.12],
            mode="text",
            text=["🚆"],
            textfont=dict(size=30),
            showlegend=False,
            hoverinfo="skip"
        ))

    fig.update_layout(
        height=350,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    st.plotly_chart(fig, use_container_width=True)

    last_station = eta_df.iloc[-1]

    st.subheader("🎯 Destination Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Destination", last_station["Station"])

    with c2:
        st.metric("Final ETA", last_station["Predicted ETA"])

    with c3:
        st.metric(
            "Final Delay",
            f"{last_station['Predicted Delay (min)']:.2f} min"
        )

    st.markdown("""
🟢 Early &nbsp;&nbsp;&nbsp;
🟡 0–5 min &nbsp;&nbsp;&nbsp;
🟠 5–15 min &nbsp;&nbsp;&nbsp;
🔴 >15 min
""")