import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Smart Energy Advisory", page_icon="⚡", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

@st.cache_data
def load_data():
    modelled = pd.read_csv(DATA_DIR / "data_modelled.csv", parse_dates=["timestamp"])
    daily = pd.read_csv(DATA_DIR / "data_daily.csv", parse_dates=["timestamp"])
    return modelled, daily

st.title("⚡ Smart Energy Consumption Analysis & Advisory")
st.caption("Explore energy usage patterns, identify unusual readings, and view practical energy-saving suggestions.")

try:
    df, daily = load_data()
except Exception as e:
    st.error("The application could not load its CSV files. Make sure the data folder is beside app.py and contains data_modelled.csv and data_daily.csv.")
    st.exception(e)
    st.stop()

if "is_anomaly" not in df.columns:
    st.error("The modelled dataset must contain the is_anomaly column.")
    st.stop()

df["date"] = df["timestamp"].dt.date
min_date = df["date"].min()
max_date = df["date"].max()

with st.sidebar:
    st.header("Filters")
    start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
    if start_date > end_date:
        st.warning("Start date must be before end date.")
        st.stop()

filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
if filtered.empty:
    st.warning("No readings found for this date range.")
    st.stop()

total = filtered["global_active_power"].sum()
average = filtered["global_active_power"].mean()
peak = filtered["global_active_power"].max()
anomalies = int(filtered["is_anomaly"].astype(bool).sum())
anomaly_pct = anomalies / len(filtered) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total recorded usage", f"{total:,.1f}", help="Sum of global_active_power values in the selected period.")
c2.metric("Average reading", f"{average:.2f}")
c3.metric("Highest reading", f"{peak:.2f}")
c4.metric("Flagged unusual readings", f"{anomalies:,}", f"{anomaly_pct:.1f}% of readings")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Usage patterns", "Anomaly detection", "Energy-saving advice"])

with tab1:
    st.subheader("Daily energy consumption")
    daily_view = daily.copy()
    daily_view["date"] = daily_view["timestamp"].dt.date
    daily_view = daily_view[(daily_view["date"] >= start_date) & (daily_view["date"] <= end_date)]
    if not daily_view.empty:
        fig = px.line(daily_view, x="timestamp", y="daily_total", markers=True,
                      labels={"timestamp": "Date", "daily_total": "Daily total"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Daily summary is not available for this selected range.")
    st.subheader("Recent readings")
    st.dataframe(filtered[["timestamp", "global_active_power", "voltage", "global_intensity", "cluster", "is_anomaly"]].tail(20),
                 use_container_width=True)

with tab2:
    st.subheader("Consumption by hour")
    hourly = filtered.groupby("hour", as_index=False)["global_active_power"].mean()
    fig = px.bar(hourly, x="hour", y="global_active_power",
                 labels={"hour": "Hour of day", "global_active_power": "Average usage"})
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Usage by day type")
    day_type = filtered.assign(day_type=filtered["is_weekend"].map({0: "Weekday", 1: "Weekend"}))
    summary = day_type.groupby("day_type", as_index=False)["global_active_power"].mean()
    if not summary.empty:
        st.plotly_chart(px.bar(summary, x="day_type", y="global_active_power",
                               labels={"day_type": "Day type", "global_active_power": "Average usage"}),
                        use_container_width=True)
    if "tou_segment" in filtered.columns:
        st.subheader("Usage by time-of-use segment")
        tou = filtered.groupby("tou_segment", as_index=False)["global_active_power"].mean()
        st.plotly_chart(px.bar(tou, x="tou_segment", y="global_active_power",
                               labels={"tou_segment": "Time segment", "global_active_power": "Average usage"}),
                        use_container_width=True)

with tab3:
    st.subheader("Readings flagged by the anomaly model")
    st.write("These are unusual readings flagged by the project's model. A flag is an alert for review, not proof of a fault or energy theft.")
    anomalies_df = filtered[filtered["is_anomaly"].astype(bool)]
    if anomalies_df.empty:
        st.success("No readings were flagged in this date range.")
    else:
        fig = px.scatter(filtered, x="timestamp", y="global_active_power", color="is_anomaly",
                         color_discrete_map={False: "#4C78A8", True: "#E45756"},
                         labels={"timestamp": "Time", "global_active_power": "Recorded usage",
                                 "is_anomaly": "Flagged anomaly"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(anomalies_df[["timestamp", "global_active_power", "voltage", "global_intensity"]],
                     use_container_width=True)

with tab4:
    st.subheader("Personalized energy-saving suggestions")
    st.write("Suggestions are generated from the selected period's observed patterns.")
    suggestions = []
    hour_avg = filtered.groupby("hour")["global_active_power"].mean()
    if not hour_avg.empty:
        busiest = int(hour_avg.idxmax())
        suggestions.append(f"Your highest average usage occurs around {busiest:02d}:00. If practical, shift flexible activities away from this time.")
    if "sub_metering_1" in filtered.columns and "sub_metering_2" in filtered.columns and "sub_metering_3" in filtered.columns:
        means = filtered[["sub_metering_1", "sub_metering_2", "sub_metering_3"]].mean()
        highest_meter = means.idxmax()
        suggestions.append(f"{highest_meter.replace('_', ' ').title()} has the highest average sub-meter reading. Review the appliances or circuits represented by this meter.")
    if anomaly_pct > 0:
        suggestions.append(f"{anomaly_pct:.1f}% of readings were flagged as unusual. Check those time periods and compare them with normal household activity.")
    else:
        suggestions.append("No unusual readings were flagged in this period. Continue monitoring usage for changes.")
    suggestions += [
        "Switch off lights and appliances when they are not needed.",
        "Avoid leaving devices on standby for long periods.",
        "Use high-power appliances efficiently and, where possible, during lower-demand periods."
    ]
    for item in suggestions:
        st.markdown(f"- {item}")
    st.caption("Advice is general and should be checked against your actual appliances, tariff, and household routine.")

st.divider()
st.caption(f"Showing {len(filtered):,} readings from {start_date} to {end_date}. Project demonstration dashboard.")
