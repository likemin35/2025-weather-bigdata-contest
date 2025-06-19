import pandas as pd
import holidays

def add_date_features(df):
    df = df.copy()
    df["tm"] = pd.to_datetime(df["tm"])

    df["year"] = df["tm"].dt.year
    df["month"] = df["tm"].dt.month
    df["dayofweek"] = df["tm"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    kr_holidays = holidays.Korea()
    df["is_holiday"] = df["tm"].dt.date.apply(lambda x: x in kr_holidays).astype("int8")

    # ===========================
    # ★★★ 새로 추가된 피처 ★★★
    # ===========================

    df["dayofyear"] = df["tm"].dt.dayofyear
    df["weekofyear"] = df["tm"].dt.isocalendar().week.astype(int)

    df["is_vacation_season"] = ((df["month"] == 7) & (df["tm"].dt.day >= 20) | (df["month"] == 8) & (df["tm"].dt.day <= 20)).astype("int8")
    df["is_holiday_weekend"] = ((df["is_weekend"] == 1) & (df["is_holiday"] == 1)).astype("int8")

    return df