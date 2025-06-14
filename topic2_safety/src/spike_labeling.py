import pandas as pd

def add_daily_spike_flags(df: pd.DataFrame) -> pd.DataFrame:
    daily_calls = df.groupby("tm")[["call_count"]].sum()

    # IQR
    q1 = daily_calls["call_count"].quantile(0.25)
    q3 = daily_calls["call_count"].quantile(0.75)
    iqr = q3 - q1
    daily_calls["daily_spike_iqr"] = daily_calls["call_count"] > (q3 + 1.5 * iqr)

    # 평균 + 3표준편차
    mean = daily_calls["call_count"].mean()
    std = daily_calls["call_count"].std()
    daily_calls["daily_spike_mean_std"] = daily_calls["call_count"] > (mean + 3 * std)

    # 이동평균
    rolling_mean = daily_calls["call_count"].rolling(window=7, center=True, min_periods=1).mean()
    rolling_std = daily_calls["call_count"].rolling(window=7, center=True, min_periods=1).std()
    daily_calls["daily_spike_moving_avg"] = daily_calls["call_count"] > (rolling_mean + 2 * rolling_std)

    # 분위수
    q90 = daily_calls["call_count"].quantile(0.90)
    q95 = daily_calls["call_count"].quantile(0.95)
    daily_calls["daily_spike_top10pct"] = daily_calls["call_count"] > q90
    daily_calls["daily_spike_top5pct"] = daily_calls["call_count"] > q95

    daily_flags = daily_calls.filter(like="daily_spike_").reset_index()
    return df.merge(daily_flags, on="tm", how="left")

def add_local_spike_flags(df: pd.DataFrame) -> pd.DataFrame:
    local_calls = df.groupby(["tm", "address_gu"])[["call_count"]].sum()

    # IQR
    q1 = local_calls.groupby("address_gu")["call_count"].quantile(0.25)
    q3 = local_calls.groupby("address_gu")["call_count"].quantile(0.75)
    iqr = q3 - q1
    threshold_iqr = q3 + 1.5 * iqr
    local_calls["local_spike_iqr"] = local_calls.apply(
        lambda row: row["call_count"] > threshold_iqr[row.name[1]], axis=1
    )

    # 평균 + 3표준편차
    mean = local_calls.groupby("address_gu")["call_count"].mean()
    std = local_calls.groupby("address_gu")["call_count"].std()
    threshold_std = mean + 3 * std
    local_calls["local_spike_mean_std"] = local_calls.apply(
        lambda row: row["call_count"] > threshold_std[row.name[1]], axis=1
    )

    # 이동평균
    local_calls["local_spike_moving_avg"] = False
    for gu, group in local_calls.groupby("address_gu"):
        rolling_mean = group["call_count"].rolling(window=7, center=True, min_periods=1).mean()
        rolling_std = group["call_count"].rolling(window=7, center=True, min_periods=1).std()
        threshold = rolling_mean + 2 * rolling_std
        mask = group["call_count"] > threshold
        local_calls.loc[group.index, "local_spike_moving_avg"] = mask

    # 분위수
    q90 = local_calls.groupby("address_gu")["call_count"].quantile(0.90)
    q95 = local_calls.groupby("address_gu")["call_count"].quantile(0.95)
    local_calls["local_spike_top10pct"] = local_calls.apply(
        lambda row: row["call_count"] > q90[row.name[1]], axis=1
    )
    local_calls["local_spike_top5pct"] = local_calls.apply(
        lambda row: row["call_count"] > q95[row.name[1]], axis=1
    )

    local_flags = local_calls.filter(like="local_spike_").reset_index()
    return df.merge(local_flags, on=["tm", "address_gu"], how="left")