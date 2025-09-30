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


@st.cache_data(show_spinner=False)
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
def apply_date_filter(df: pd.DataFrame, date_filter: str) -> pd.DataFrame:
    now = datetime.now()

    if date_filter == "Yesterday":
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        df = df[(df["request_time"] >= start) & (df["request_time"] < end)]
    elif date_filter == "Last 7 Days":
        start = now - timedelta(days=7)
        df = df[df["request_time"] >= start]
    elif date_filter == "Last 30 Days":
        start = now - timedelta(days=30)
        df = df[df["request_time"] >= start]
    elif date_filter == "Last 60 Days":
        start = now - timedelta(days=60)
        df = df[df["request_time"] >= start]
    # else All → no filter

    return df


def filter_data(df: pd.DataFrame, date_filter: str, solved_only: bool = True) -> pd.DataFrame:
    # Convert to datetime
    df["request_time"] = pd.to_datetime(df["request_time"], errors="coerce")
    df["response_time"] = pd.to_datetime(df["response_time"], errors="coerce")

    # Remove unwanted customers
    df["customer_name"] = df["customer_name"].astype(str).str.strip()
    df = df[~df["customer_name"].str.contains("oakanalytics.com|nutanxt.com", case=False, na=False)]

    # Apply date filter
    df = apply_date_filter(df, date_filter)

    # Mark solved/unsolved
    df["Solved_ER"] = df["request_time"].notna() & df["response_time"].notna()

    # Optionally keep only solved
    if solved_only:
        df = df[df["Solved_ER"]]

    return df


# -------------------- DATA PROCESSING --------------------
def process_data(df: pd.DataFrame):
    frame = df.copy()

    # Response duration in hours
    frame["ER_Response_Duration_hr"] = (
        (frame["response_time"] - frame["request_time"]).dt.total_seconds() / 3600
    )

    # Shift summary
    frame["Shift_Name"] = frame["response_time"].dt.hour.apply(
        lambda h: "00-06 | Night" if 0 <= h < 6
        else "06-12 | Morning" if 6 <= h < 12
        else "12-18 | Afternoon" if 12 <= h < 18
        else "18-24 | Evening"
    )
    quarter_order = ["00-06 | Night", "06-12 | Morning", "12-18 | Afternoon", "18-24 | Evening"]
    quarter_summary = (
        frame.groupby("Shift_Name")
        .agg(
            ER_Count=("Shift_Name", "size"),
            Avg_Duration_hr=("ER_Response_Duration_hr", "mean")
        )
        .reindex(quarter_order)
        .reset_index()
    )

    # Device summary
    device_summary = (
        frame.groupby("device_name")
        .agg(
            ER_Count=("device_name", "size"),
            Avg_Duration_hr=("ER_Response_Duration_hr", "mean")
        )
        .reset_index()
    )

    return frame, quarter_summary, device_summary


# -------------------- PIE CHART --------------------
def build_er_pie_chart(df: pd.DataFrame) -> go.Figure:
    # Count solved vs unsolved
    er_counts = df["Solved_ER"].value_counts().rename(index={True: "Solved ER", False: "Unsolved ER"})

    fig = px.pie(
        values=er_counts.values,
        names=er_counts.index,
        title="Solved vs Unsolved ERs (Filtered Customers + Date)",
        hole=0.3
    )
    return fig


# -------------------- CHART BUILDERS --------------------
def build_shift_chart(quarter_summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=quarter_summary["Shift_Name"],
            y=quarter_summary["ER_Count"],
            name="ER Count",
            marker_color="#4F81BD",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=quarter_summary["Shift_Name"],
            y=quarter_summary["Avg_Duration_hr"],
            name="Avg Duration (hrs)",
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.update_layout(title_text="ER Count & Avg Duration by Shift", legend_title_text="Series")
    return fig


def build_duration_chart(frame: pd.DataFrame) -> go.Figure:
    """Scatter plot of individual ER response durations."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["ER_Response_Duration_hr"],
            mode="markers",
            name="Response Duration",
            marker=dict(color="#9BBB59", size=6, opacity=0.7),
        )
    )
    fig.update_layout(
        title="ER Response Durations (Raw Records)",
        xaxis_title="Record Index",
        yaxis_title="Response Duration (hrs)",
    )
    return fig


def build_duration_histogram(frame: pd.DataFrame):
    """Histogram: count of ERs in hour-buckets (0–1, 1–2, … 23–24, >24)."""
    bins = list(range(0, 25))
    labels = [f"{i}-{i+1} hr" for i in range(0, 24)]
    labels.append(">24 hr")

    frame["Duration_Bucket"] = pd.cut(
        frame["ER_Response_Duration_hr"],
        bins=bins + [float("inf")],
        labels=labels,
        right=False
    )

    bucket_summary = (
        frame["Duration_Bucket"]
        .value_counts()
        .reindex(labels, fill_value=0)
        .reset_index()
    )
    bucket_summary.columns = ["Bucket", "Count"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bucket_summary["Bucket"],
            y=bucket_summary["Count"],
            marker_color="#4F81BD",
            name="ER Count"
        )
    )
    fig.update_layout(
        title="Count of ERs by Response Duration Buckets",
        xaxis_title="Response Duration (hrs)",
        yaxis_title="ER Count"
    )
    return fig, bucket_summary


def build_device_chart(device_summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=device_summary["device_name"],
            y=device_summary["ER_Count"],
            name="ER Count",
            marker_color="#F79646",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=device_summary["device_name"],
            y=device_summary["Avg_Duration_hr"],
            name="Avg Duration (hrs)",
            mode="lines+markers",
        ),
        secondary_y=True,
    )
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
    st.caption("View raw DB data and processed analytics.")

    host = "oak-centralized-db.clyejxhqj10e.us-west-1.rds.amazonaws.com"
    user = "admin"
    password = "oak_admin"
    database = "oak_db"

    with st.spinner("Connecting and fetching data..."):
        try:
            df_raw = fetch_data(host, user, password, database)
        except Exception as exc:
            st.error(f"Failed to fetch data: {exc}")
            return

    if df_raw.empty:
        st.warning("No data returned from the database.")
        return

    # Date filter
    date_filter = st.selectbox(
        "Select Date Range:",
        ["Yesterday", "Last 7 Days", "Last 30 Days", "Last 60 Days", "All"],
        index=4
    )

    # Filtered datasets
    df_filtered_solved = filter_data(df_raw.copy(), date_filter, solved_only=True)   # For Shiftwise, Device, Duration
    df_filtered_all = filter_data(df_raw.copy(), date_filter, solved_only=False)     # For Pie Chart

    # Process solved-only data
    df_processed, quarter_summary, device_summary = process_data(df_filtered_solved)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Database (Raw DB Data)",
            "Shiftwise Summary",
            "Response Durations",
            "Device Summary",
            "Solved vs Unsolved ERs",
        ]
    )

    # Tab 1
    with tab1:
        st.subheader(f"Raw DB Data (Filtered → {date_filter})")
        st.dataframe(df_filtered_solved, use_container_width=True)

    # Tab 2
    with tab2:
        st.subheader(f"Shiftwise Summary ({date_filter}, Solved ERs Only)")
        st.dataframe(quarter_summary, use_container_width=True)
        st.plotly_chart(build_shift_chart(quarter_summary), use_container_width=True)

    # Tab 3
    with tab3:
        st.subheader(f"Response Durations (hrs, {date_filter}, Solved ERs Only)")
        # st.plotly_chart(build_duration_chart(df_processed), use_container_width=True)

        fig_hist, bucket_summary = build_duration_histogram(df_processed)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.dataframe(bucket_summary, use_container_width=True)

    # Tab 4
    with tab4:
        st.subheader(f"Device Summary ({date_filter}, Solved ERs Only)")
        st.dataframe(device_summary, use_container_width=True)
        st.plotly_chart(build_device_chart(device_summary), use_container_width=True)

    # Tab 5
    with tab5:
        st.subheader(f"Solved vs Unsolved ERs (Filtered Customers + {date_filter})")
        st.plotly_chart(build_er_pie_chart(df_filtered_all), use_container_width=True)

    # Excel export
    excel_bytes = build_excel_bytes(df_raw, df_processed, quarter_summary, device_summary)
    st.download_button(
        label="Download Excel (All Tabs)",
        data=excel_bytes,
        file_name="ER_Response_AllTabs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()