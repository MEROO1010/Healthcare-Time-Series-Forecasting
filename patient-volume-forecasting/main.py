import pandas as pd
from pathlib import Path

from features.feature_engineering import add_time_features
from models.arima_model import train_arima
from models.prophet_model import train_prophet
from evaluation.metrics import evaluate
from visualization.plots import plot_forecast


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "patient_visits.csv"


# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = add_time_features(df)

# -----------------------------
# Train / Test Split
# -----------------------------
TEST_DAYS = 60
train = df.iloc[:-TEST_DAYS]
test = df.iloc[-TEST_DAYS:]


# =============================
# ARIMA
# =============================
print("\nTraining ARIMA model...")
arima_model = train_arima(train["patient_visits"])
arima_forecast = arima_model.forecast(steps=TEST_DAYS)

arima_metrics = evaluate(
    test["patient_visits"],
    arima_forecast
)

print("ARIMA Results:", arima_metrics)

plot_forecast(
    train,
    test,
    arima_forecast,
    title="ARIMA Patient Volume Forecast"
)


# =============================
# Prophet
# =============================
print("\nTraining Prophet model...")
prophet_model = train_prophet(train)

future = prophet_model.make_future_dataframe(periods=TEST_DAYS)
forecast = prophet_model.predict(future)

prophet_forecast = forecast.tail(TEST_DAYS)["yhat"].values

prophet_metrics = evaluate(
    test["patient_visits"],
    prophet_forecast
)

print("Prophet Results:", prophet_metrics)

plot_forecast(
    train,
    test,
    prophet_forecast,
    title="Prophet Patient Volume Forecast"
)


# =============================
# Model Comparison
# =============================
results = pd.DataFrame([
    {"Model": "ARIMA", **arima_metrics},
    {"Model": "Prophet", **prophet_metrics}
])

print("\nModel Comparison:")
print(results)