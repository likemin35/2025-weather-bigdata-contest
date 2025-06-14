import pandas as pd
from datetime import timedelta

def add_prev_year_features(df):
    df = df.copy()
    df["tm"] = pd.to_datetime(df["tm"])

    df_prev = df[["tm", "address_gu", "sub_address", "call_count"]].copy()
    df_prev["tm"] += pd.DateOffset(years=1)
    df_all = pd.concat([
        df_prev.assign(tm=df_prev["tm"] + timedelta(days=d)) for d in [-2, -1, 0, 1, 2]
    ])
    df_avg = (
        df_all.groupby(["tm", "address_gu", "sub_address"], observed=False)["call_count"]
        .mean()
        .reset_index()
        .rename(columns={"call_count": "call_count_prev_year"})
    )

    df = df.merge(df_avg, on=["tm", "address_gu", "sub_address"], how="left")

    df.loc[df["year"] != 2020, "call_count_prev_year"] = (
        df.loc[df["year"] != 2020, "call_count_prev_year"].fillna(0)
    )
    df["has_prev_year_data"] = df["call_count_prev_year"].notna().astype(int)

    return df