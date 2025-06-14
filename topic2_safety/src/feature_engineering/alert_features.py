import pandas as pd

def add_alert_features(df, df_wrn):
    df = df.copy()
    df_wrn = df_wrn.copy()

    df["tm"] = pd.to_datetime(df["tm"])
    df_wrn["발효시각"] = pd.to_datetime(df_wrn["발효시각"])
    df_wrn["발효일"] = df_wrn["발효시각"].dt.normalize()

    df_wrn_filtered = df_wrn[df_wrn["특보명령"] != "해제"].copy()

    level_score_map = {"주의보": 1, "경보": 2}
    type_score_map = {"강풍": 1, "건조": 1, "태풍": 3, "폭염": 2, "폭풍해일": 3, "풍랑": 2, "호우": 2}

    df_wrn_filtered["type_score"] = df_wrn_filtered["특보종류"].map(type_score_map)
    df_wrn_filtered["level_score"] = df_wrn_filtered["특보수준"].map(level_score_map)

    df_agg = (
        df_wrn_filtered.groupby("발효일", observed=False)
        .agg(
            has_alert = ("특보종류", lambda x: 1),
            alert_count = ("특보종류", "count"),
            alert_type = ("특보종류", lambda x: set(x)),
            alert_type_score = ("type_score", "sum"),
            alert_level_score = ("level_score", "sum")
        )
        .reset_index()
        .sort_values("발효일")
        .reset_index(drop=True)
    )

    streaks = []
    count = 0
    for i in range(len(df_agg)):
        if i == 0:
            count = 1
        else:
            prev = df_agg.loc[i - 1, "발효일"]
            curr = df_agg.loc[i, "발효일"]
            count = count + 1 if (curr - prev).days == 1 else 1
        streaks.append(count)
    df_agg["alert_streak"] = streaks

    df = df.merge(df_agg, left_on="tm", right_on="발효일", how="left").drop(columns=["발효일"])
    df["has_alert"] = df["has_alert"].fillna(0).astype(int)
    df["alert_count"] = df["alert_count"].fillna(0).astype(int)
    df["alert_type"] = df["alert_type"].apply(lambda x: x if isinstance(x, set) else set())
    df["alert_streak"] = df["alert_streak"].fillna(0).astype(int)
    df["alert_type_score"] = df["alert_type_score"].fillna(0)
    df["alert_level_score"] = df["alert_level_score"].fillna(0)

    return df