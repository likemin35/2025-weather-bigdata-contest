import pandas as pd
import numpy as np

def add_weather_features(df):
    df = df.copy()

    df["is_rainy"] = (df["rn_day"] > 0).astype("int8")
    df["rn_day_bin"] = pd.cut(
        df["rn_day"],
        bins=[-1, 0, 10.5, 33.0, 80, np.inf],
        labels=["none", "light", "moderate", "heavy", "very_heavy"],
        right=True  # 0 == none
    )
    df["rn_day_3d_sum"] = (
        df.groupby(["sub_address"], observed=False)["rn_day"]
        .rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    Ta = df[["ta_max", "ta_min"]].mean(axis=1)
    RH = df[["hm_max", "hm_min"]].mean(axis=1)
    Tw = (
        Ta * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) +
        np.arctan(Ta + RH) -
        np.arctan(RH - 1.67633) +
        0.00391838 * (RH ** 1.5) * np.arctan(0.023101 * RH) -
        4.686035
    )
    df["apparent_temp"] = (-0.2442 + 0.55399 * Tw + 0.45535 * Ta - 0.0022 * (Tw ** 2) + 0.00278 * Tw * Ta + 3.0)
    df["discomfort_index"] = ((9 / 5) * Ta - 0.55 * (1 - RH / 100) * ((9 / 5) * Ta - 26) + 32)

    # ===========================
    # ★★★ 새로 추가된 피처 ★★★
    # ===========================

    df["wind_gust_factor"] = df["ws_ins_max"] / (df["ws_max"] + 1e-6)  # zero division 방지
    df["is_rainy_and_windy"] = ((df["rn_day"] > 30) & (df["ws_max"] > 10)).astype("int8")

    return df