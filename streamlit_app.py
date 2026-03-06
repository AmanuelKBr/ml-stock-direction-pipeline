import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Stock ML Monitor", layout="wide")

API_URL = "https://ml-stock-direction-pipeline.onrender.com"


# ------------------------------------------------
# Page Styling
# ------------------------------------------------

st.markdown(
    """
<style>

html, body, [class*="css"]  {
    font-size: 18px;
}

h1 {
    font-size: 42px !important;
}

h2 {
    font-size: 30px !important;
}

table {
    text-align:center;
}

thead tr th {
    text-align:center !important;
}

tbody tr td {
    text-align:center !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------
# Load Logs
# ------------------------------------------------

def load_logs():

    try:

        r = requests.get(f"{API_URL}/logs")
        data = r.json()

        df = pd.DataFrame(data)

        if not df.empty:

            df["timestamp"] = pd.to_datetime(df["timestamp"])

            if "actual" not in df.columns:
                df["actual"] = "Pending"

        return df

    except:

        return pd.DataFrame(
            columns=["timestamp", "price", "prediction", "actual", "probability"]
        )


df = load_logs()

st.title("📈 ML Stock Direction Prediction Monitor")

# ------------------------------------------------
# Sidebar Controls
# ------------------------------------------------

st.sidebar.header("Controls")

if st.sidebar.button("Get New Prediction"):

    try:
        requests.get(f"{API_URL}/predict")
        st.sidebar.success("Prediction triggered")
    except:
        st.sidebar.error("Prediction request failed")

    df = load_logs()

if st.sidebar.button("Retrain Model"):

    try:
        requests.post(f"{API_URL}/retrain")
        st.sidebar.success("Retraining triggered")
    except:
        st.sidebar.error("Retraining request failed")


# ------------------------------------------------
# Latest Prediction Section
# ------------------------------------------------

if not df.empty:

    latest = df.iloc[-1]

    st.subheader("Latest Prediction")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Last Prediction", latest["prediction"])

    col2.metric(
        "Probability %",
        f"{latest['probability']*100:.2f}%",
    )

    col3.metric(
        "Last SPY Price",
        f"${latest['price']:.2f}",
    )

    col4.metric(
        "Model Version",
        "v1.0",
    )


st.divider()

# ------------------------------------------------
# Model Health Section
# ------------------------------------------------

if not df.empty:

    st.subheader("Model Health Overview")

    total_predictions = len(df)

    avg_confidence = df["probability"].mean()

    up_ratio = (df["prediction"] == "UP").mean()

    down_ratio = (df["prediction"] == "DOWN").mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", total_predictions)

    col2.metric(
        "Average Confidence",
        f"{avg_confidence*100:.2f}%",
    )

    col3.metric(
        "UP Prediction Ratio",
        f"{up_ratio*100:.1f}%",
    )

    col4.metric(
        "DOWN Prediction Ratio",
        f"{down_ratio*100:.1f}%",
    )


# ------------------------------------------------
# Prediction History Table
# ------------------------------------------------

st.subheader("Prediction History")

if not df.empty:

    display_df = df.copy()

    display_df["probability"] = display_df["probability"].astype(float).round(8)

    st.table(display_df.tail(25))

else:

    st.warning("No prediction logs available yet.")


# ------------------------------------------------
# Probability Trend
# ------------------------------------------------

if not df.empty:

    fig = px.line(
        df,
        x="timestamp",
        y="probability",
        title="Prediction Probability Over Time",
    )

    fig.update_layout(
        title_font_size=24,
        xaxis_title="Timestamp",
        yaxis_title="Probability",
    )

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# Price Trend
# ------------------------------------------------

if not df.empty:

    fig2 = px.line(
        df,
        x="timestamp",
        y="price",
        title="SPY Price at Prediction Time",
    )

    fig2.update_layout(
        title_font_size=24,
        xaxis_title="Timestamp",
        yaxis_title="Price",
    )

    st.plotly_chart(fig2, use_container_width=True)