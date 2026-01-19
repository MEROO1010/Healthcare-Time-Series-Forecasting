import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "patient_visits.csv"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

st.title("🏥 Patient Volume Forecasting Dashboard")
st.line_chart(df.set_index("date")["patient_visits"])

st.markdown("""
### Business Use
- Staff planning
- Bed management
- Peak day detection
""")