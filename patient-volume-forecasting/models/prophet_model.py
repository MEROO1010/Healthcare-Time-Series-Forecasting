from prophet import Prophet

def train_prophet(df):
    prophet_df = df.rename(columns={"date": "ds", "patient_visits": "y"})
    model = Prophet()
    model.fit(prophet_df)
    return model