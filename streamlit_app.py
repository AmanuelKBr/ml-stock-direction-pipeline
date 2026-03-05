import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Stock ML Monitor", layout="wide")

st.title("📈 ML Stock Direction Prediction Monitor")

API_URL = "https://ml-stock-direction-pipeline.onrender.com"


def load_logs():

    try:
        r = requests.get(f"{API_URL}/logs")
        data = r.json()
        df = pd.DataFrame(data)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    except:
        return pd.DataFrame(columns=["timestamp", "price", "prediction", "probability"])


df = load_logs()


st.sidebar.header("Controls")

if st.sidebar.button("Get New Prediction"):
    requests.get(f"{API_URL}/predict")
    st.sidebar.success("Prediction triggered")
    df = load_logs()

if st.sidebar.button("Retrain Model"):
    requests.post(f"{API_URL}/retrain")
    st.sidebar.success("Retraining triggered")


if not df.empty:

    latest = df.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric("Last Prediction", latest["prediction"])
    col2.metric("Probability UP", round(latest["probability"], 3))
    col3.metric("SPY Price", round(latest["price"], 2))


st.subheader("Prediction History")

st.dataframe(df.tail(20), use_container_width=True)


if not df.empty:

    fig = px.line(
        df,
        x="timestamp",
        y="probability",
        title="Prediction Probability Over Time",
    )

    st.plotly_chart(fig, use_container_width=True)


if not df.empty:

    fig2 = px.line(
        df,
        x="timestamp",
        y="price",
        title="SPY Price at Prediction Time",
    )

    st.plotly_chart(fig2, use_container_width=True)