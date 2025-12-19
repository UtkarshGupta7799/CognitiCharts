import streamlit as st
import pandas as pd
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cogniti_charts.utils import load_config, CLASS_NAMES
from cogniti_charts.data import load_prices, label_patterns
from cogniti_charts.features import build_features
from cogniti_charts.infer import predict

st.set_page_config(page_title="CognitiCharts", layout="wide")
st.title("📈 CognitiCharts – AI Chart Pattern Recognition")

cfg = load_config()
lookback = cfg["data"]["lookback"]
framework = st.sidebar.selectbox("Framework", ["tf", "torch"], index=0)

# -----------------------------
# Upload or use sample
# -----------------------------
st.header("1) Upload CSV or use sample")
uploaded = st.file_uploader("Upload a CSV with 'date' and 'close' columns", type=["csv"])
use_sample = st.checkbox("Use sample_data/sample_prices.csv", value=True if uploaded is None else False)

def _read_df(file):
    """Helper to preview CSV as DataFrame"""
    if file is None:
        return load_prices("sample_data/sample_prices.csv")
    else:
        file.seek(0)  # rewind in case file was read before
        return pd.read_csv(file, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

try:
    df = _read_df(None if use_sample else uploaded)
    st.dataframe(df.head(10))
except Exception as e:
    st.error(f"Could not read file: {e}")
    df = None

# -----------------------------
# Predict
# -----------------------------
st.header("2) Predict Patterns")
if st.button("Run Prediction", use_container_width=True):
    try:
        # default to sample file
        source = "sample_data/sample_prices.csv"
        if not use_sample and uploaded is not None:
            uploaded.seek(0)  # rewind again before second read
            source = uploaded

        preds = predict(source, lookback=lookback, framework=framework)
        st.session_state["preds"] = preds
        st.success("Predictions ready!")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------------
# Results
# -----------------------------
if "preds" in st.session_state:
    st.header("3) Results")
    preds = st.session_state["preds"]
    st.dataframe(preds.head(20))

    st.subheader("Label distribution")
    st.bar_chart(preds["pred_label"].value_counts())
