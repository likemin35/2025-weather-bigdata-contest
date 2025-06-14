import pandas as pd
import numpy as np
import holidays

def add_date_features(df):
    df = df.copy()
    df["tm"] = pd.to_datetime(df["tm"])

    df["year"] = df["tm"].dt.year
    df["month"] = df["tm"].dt.month
    df["dayofweek"] = df["tm"].dt.dayofweek
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    kr_holidays = holidays.Korea()
    df["is_holiday"] = df["tm"].dt.date.apply(lambda x: x in kr_holidays).astype(int)

    return df