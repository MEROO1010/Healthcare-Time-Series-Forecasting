def add_time_features(df):
    df["day_of_week"] = df["date"].dt.dayofweek
    df["weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    return df