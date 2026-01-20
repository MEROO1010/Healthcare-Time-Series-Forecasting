import sys
from pathlib import Path

# -----------------------------
# Fix Python Path for Streamlit
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st
import pandas as pd

from models.arima_model import train_arima
from models.prophet_model import train_prophet
from evaluation.metrics import evaluate

from models.lstm_model import prepare_lstm_data, train_lstm
import numpy as np
# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Patient Volume Forecasting",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "patient_visits.csv"

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

TEST_DAYS = 60
train = df.iloc[:-TEST_DAYS]
test = df.iloc[-TEST_DAYS:]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["ARIMA", "Prophet", "LSTM"]
)

# -----------------------------
# Run Model
# -----------------------------
if model_choice == "ARIMA":
    model = train_arima(train["patient_visits"])
    forecast = model.forecast(steps=TEST_DAYS)

elif model_choice == "Prophet":
    model = train_prophet(train)
    future = model.make_future_dataframe(periods=TEST_DAYS)
    forecast_df = model.predict(future)
    forecast = forecast_df.tail(TEST_DAYS)["yhat"].values

elif model_choice == "LSTM":
    series = df["patient_visits"]

    X, y, scaler = prepare_lstm_data(series)
    split = len(X) - TEST_DAYS

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = train_lstm(X_train, y_train)

    predictions = model.predict(X_test)
    forecast = scaler.inverse_transform(predictions).flatten()    

# -----------------------------
# Evaluation
# -----------------------------
metrics = evaluate(test["patient_visits"], forecast) 
# -----------------------------
# Smart Recommendation
# -----------------------------
st.subheader("🧠 Model Recommendation")

if model_choice == "ARIMA":
    recommendation = (
        "ARIMA provides stable short-term forecasts and can be useful "
        "for quick operational adjustments."
    )
else:
    recommendation = (
        "Prophet shows strong performance in capturing seasonality and trends, "
        "making it more suitable for hospital capacity and staff planning."
    )

st.success(recommendation)
# -----------------------------
# Main Dashboard
# -----------------------------
st.title("🏥 Patient Volume Forecasting Dashboard")

# Metrics
col1, col2 = st.columns(2)
col1.metric("MAE", round(metrics["MAE"], 2))
col2.metric("RMSE", round(metrics["RMSE"], 2))

# Forecast Plot
plot_df = pd.DataFrame({
    "Date": test["date"],
    "Actual": test["patient_visits"],
    "Forecast": forecast
}).set_index("Date")

st.subheader("📈 Forecast vs Actual")
st.line_chart(plot_df)

# -----------------------------
# Business Insights
# -----------------------------
st.subheader("📊 Business Insights")

st.markdown("""
- Accurate patient volume forecasting helps optimize staff scheduling.
- Lower RMSE indicates better reliability for operational planning.
- Forecasts can support bed management and peak-day preparation.
""")

st.info(
    "This dashboard demonstrates how predictive analytics can support hospital decision-making."
)