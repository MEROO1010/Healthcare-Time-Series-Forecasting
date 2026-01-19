import pandas as pd
import numpy as np

def generate_patient_visits(start_date="2023-01-01", days=730):
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=days)

    trend = np.linspace(100, 130, days)
    weekly = 15 * np.sin(2 * np.pi * dates.dayofweek / 7)
    yearly = 20 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = np.random.normal(0, 10, days)

    visits = trend + weekly + yearly + noise
    visits = np.maximum(visits.astype(int), 20)

    df = pd.DataFrame({
        "date": dates,
        "patient_visits": visits
    })

    return df

if __name__ == "__main__":
    df = generate_patient_visits()
    df.to_csv("data/patient_visits.csv", index=False)