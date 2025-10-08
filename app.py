import io
import pandas as pd
import pymysql
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


# -------------------- DB CONNECTION --------------------
def create_db_connection(host: str, user: str, password: str, database: str):
    return pymysql.connect(host=host, user=user, password=password, database=database)


# -------------------- FETCH DATA (Cached for 15 Days) --------------------
@st.cache_data(ttl=1296000, show_spinner=False)  # 15 days = 1,296,000 seconds
def fetch_data(host: str, user: str, password: str, database: str) -> pd.DataFrame:
    connection = create_db_connection(host, user, password, database)
    query = """
        SELECT id, user_id, scan_id, customer_name, device_name,
               request_time, response_time
        FROM oak_expert_review;
    """
    try:
        frame = pd.read_sql(query, connection)
    finally:
        connection.close()
    return frame


# -------------------- FILTERING --------------------
def apply_date_filter(df: pd.DataFrame, date_filter: str, start_date=None, end_date=None) -> pd.DataFrame:
    df["request_time"] = pd.to_datetime(df["request_time"], errors="coerce")
    now = datetime.now()

    if date_filter == "Yesterday":
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    elif date_filter == "Last 7 Days":
        start = now - timedelta(days=7)
        end = now
    elif date_filter == "Last 30 Days":
        start = now - timedelta(days=30)
        end = now
    elif date_filter == "Last 60 Days":
        start = now - timedelta(days=60)
        end = now
    elif date_filter == "Custom Range" and start_date and end_date:
        start = datetime.combine(start_date, datetime.min.time())
        end = datetime.combine(end_date, datetime.max.time())
    else:
        start = df["request_time"].min()
        end = df["request_time"].max()

    return df[(df["request_time"] >= start) & (df["request_time"] <= end)]


def filter_data(df: pd.DataFrame, date_filter: str, start_date=None, end_date=None, solved_only: bool = True) -> pd.DataFrame:
    df["request_time"] = pd.to_datetime(df["request_time"], errors="coerce")
    df["response_time"] = pd.to_datetime(df["response_time"], errors="coerce")

    # Remove unwanted customers
    df["customer_name"] = df["customer_name"].astype(str).str.strip()
    df = df[~df["customer_name"].str.contains("oakanalytics.com|nutanxt.com", case=False, na=False)]

    # Apply date filter
    df = apply_date_filter(df, date_filter, start_date, end_date)

    # Mark solved/unsolved
    df["Solved_ER"] = df["request_time"].notna() & df["response_time"].notna()

    # Optionally keep only solved
    if solved_only:
        df = df[df["Solved_ER"]]

    return df


# -------------------- DATA PROCESSING --------------------
def process_data(df: pd.DataFrame):
    frame = df.copy()

    frame["ER_Response_Duration_hr"] = (
        (frame["response_time"] - frame["request_time"]).dt.total_seconds() / 3600
    )

    frame["Shift_Name"] = frame["response_time"].dt.hour.apply(
        lambda h: "00-06 | Night" if 0 <= h < 6
        else "06-12 | Morning" if 6 <= h < 12
        else "12-18 | Afternoon" if 12 <= h < 18
        else "18-24 | Evening"
    )
    quarter_order = ["00-06 | Night", "06-12 | Morning", "12-18 | Afternoon", "18-24 | Evening"]
    quarter_summary = (
        frame.groupby("Shift_Name")
        .agg(ER_Count=("Shift_Name", "size"), Avg_Duration_hr=("ER_Response_Duration_hr", "mean"))
        .reindex(quarter_order)
        .reset_index()
    )

    device_summary = (
        frame.groupby("device_name")
        .agg(ER_Count=("device_name", "size"), Avg_Duration_hr=("ER_Response_Duration_hr", "mean"))
        .reset_index()
    )

    return frame, quarter_summary, device_summary


# -------------------- CHART BUILDERS --------------------
def build_er_pie_chart(df: pd.DataFrame) -> go.Figure:
    er_counts = df["Solved_ER"].value_counts().rename(index={True: "Solved ER", False: "Unsolved ER"})
    fig = px.pie(
        values=er_counts.values,
        names=er_counts.index,
        title="Solved vs Unsolved ERs (Filtered Customers + Date)",
        hole=0.3
    )
    return fig


def build_shift_chart(quarter_summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=quarter_summary["Shift_Name"], y=quarter_summary["ER_Count"],
                         name="ER Count", marker_color="#4F81BD"), secondary_y=False)
    fig.add_trace(go.Scatter(x=quarter_summary["Shift_Name"], y=quarter_summary["Avg_Duration_hr"],
                             name="Avg Duration (hrs)", mode="lines+markers"), secondary_y=True)
    fig.update_layout(title_text="ER Count & Avg Duration by Shift", legend_title_text="Series")
    return fig


def build_duration_histogram(frame: pd.DataFrame):
    bins = list(range(0, 25))
    labels = [f"{i}-{i+1} hr" for i in range(0, 24)] + [">24 hr"]

    frame["Duration_Bucket"] = pd.cut(
        frame["ER_Response_Duration_hr"], bins=bins + [float("inf")],
        labels=labels, right=False
    )

    bucket_summary = frame["Duration_Bucket"].value_counts().reindex(labels, fill_value=0).reset_index()
    bucket_summary.columns = ["Bucket", "Count"]

    fig = go.Figure(go.Bar(x=bucket_summary["Bucket"], y=bucket_summary["Count"],
                           marker_color="#4F81BD", name="ER Count"))
    fig.update_layout(title="Count of ERs by Response Duration Buckets",
                      xaxis_title="Response Duration (hrs)", yaxis_title="ER Count")
    return fig, bucket_summary


def build_device_chart(device_summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=device_summary["device_name"], y=device_summary["ER_Count"],
                         name="ER Count", marker_color="#F79646"), secondary_y=False)
    fig.add_trace(go.Scatter(x=device_summary["device_name"], y=device_summary["Avg_Duration_hr"],
                             name="Avg Duration (hrs)", mode="lines+markers"), secondary_y=True)
    fig.update_layout(title_text="ER Count & Avg Duration by Device", legend_title_text="Series")
    return fig


# -------------------- EXCEL EXPORT --------------------
def build_excel_bytes(raw_data, processed, quarter_summary, device_summary):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        raw_data.to_excel(writer, sheet_name="Raw_DB_Data", index=False)
        processed.to_excel(writer, sheet_name="Processed_Data", index=False)
        quarter_summary.to_excel(writer, sheet_name="Shiftwise_Summary", index=False)
        device_summary.to_excel(writer, sheet_name="Device_Summary", index=False)
    output.seek(0)
    return output.read()


# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(page_title="ER Response Dashboard", layout="wide")

    st.title("ER Response Dashboard")
    st.caption("Data auto-refreshes every 15 days . Use filters below for analysis.")

    host = "oak-centralized-db.clyejxhqj10e.us-west-1.rds.amazonaws.com"
    user = "admin"
    password = "oak_admin"
    database = "oak_db"

    with st.spinner("Fetching data from database..."):
        try:
            df_raw = fetch_data(host, user, password, database)
        except Exception as exc:
            st.error(f"Failed to fetch data: {exc}")
            return

    if df_raw.empty:
        st.warning("No data returned from the database.")
        return

    # -------------------- FILTERS --------------------
    st.sidebar.header("📅 Date Filters")
    date_filter = st.sidebar.selectbox(
        "Select Range Type",
        ["Yesterday", "Last 7 Days", "Last 30 Days", "Last 60 Days", "All", "Custom Range"],
        index=4
    )

    start_date, end_date = None, None
    if date_filter == "Custom Range":
        start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        end_date = st.sidebar.date_input("End Date", value=datetime.now())
        if start_date > end_date:
            st.sidebar.error("⚠️ End Date must be after Start Date.")
            st.stop()

    # Filtered datasets
    df_filtered_solved = filter_data(df_raw.copy(), date_filter, start_date, end_date, solved_only=True)
    df_filtered_all = filter_data(df_raw.copy(), date_filter, start_date, end_date, solved_only=False)

    # Process solved-only data
    df_processed, quarter_summary, device_summary = process_data(df_filtered_solved)

    # -------------------- TABS --------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Database (Raw DB Data)", "Shiftwise Summary", "Response Durations",
         "Device Summary", "Solved vs Unsolved ERs"]
    )

    with tab1:
        st.subheader("Raw DB Data (Filtered)")
        st.dataframe(df_filtered_solved, use_container_width=True)

    with tab2:
        st.subheader("Shiftwise Summary (Solved ERs Only)")
        st.dataframe(quarter_summary, use_container_width=True)
        st.plotly_chart(build_shift_chart(quarter_summary), use_container_width=True)

    with tab3:
        st.subheader("Response Durations (hrs, Solved ERs Only)")
        fig_hist, bucket_summary = build_duration_histogram(df_processed)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.dataframe(bucket_summary, use_container_width=True)

    with tab4:
        st.subheader("Device Summary (Solved ERs Only)")
        st.dataframe(device_summary, use_container_width=True)
        st.plotly_chart(build_device_chart(device_summary), use_container_width=True)

    with tab5:
        st.subheader("Solved vs Unsolved ERs")
        st.plotly_chart(build_er_pie_chart(df_filtered_all), use_container_width=True)

    # Excel export
    excel_bytes = build_excel_bytes(df_raw, df_processed, quarter_summary, device_summary)
    st.download_button(
        label="⬇️ Download Excel (All Tabs)",
        data=excel_bytes,
        file_name="ER_Response_AllTabs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()