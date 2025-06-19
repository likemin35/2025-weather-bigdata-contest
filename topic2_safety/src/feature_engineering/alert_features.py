import pandas as pd

def add_alert_features(df, df_wrn):
    df = df.copy()
    df_wrn = df_wrn.copy()

    df["tm"] = pd.to_datetime(df["tm"])
    df_wrn["발효시각"] = pd.to_datetime(df_wrn["발효시각"])
    df_wrn["발효일"] = df_wrn["발효시각"].dt.normalize()
    df_wrn_filtered = df_wrn[df_wrn["특보명령"] != "해제"].copy()

    level_score_map = {"주의보": 1, "경보": 2}
    type_score_map = {"강풍": 2, "건조": 2, "태풍": 3, "폭염": 3, "폭풍해일": 2, "풍랑": 1, "호우": 3}
    df_wrn_filtered["type_score"] = df_wrn_filtered["특보종류"].map(type_score_map)
    df_wrn_filtered["level_score"] = df_wrn_filtered["특보수준"].map(level_score_map)

    df_agg = (
        df_wrn_filtered.groupby("발효일", observed=False)
        .agg(
            has_alert = ("특보종류", lambda x: 1),
            alert_count = ("특보종류", "count"),
            alert_type_score = ("type_score", "sum"),
            alert_level_score = ("level_score", "sum")
        )
        .reset_index()
        .sort_values("발효일")
        .reset_index(drop=True)
    )

    df_wrn_dummies = pd.get_dummies(df_wrn_filtered.set_index('발효일')['특보종류']).astype(int)
    df_wrn_dummies = df_wrn_dummies.groupby('발효일').sum()

    df_agg_combined = df_agg.merge(df_wrn_dummies, on='발효일', how='outer')
    df_agg_combined = df_agg_combined.reset_index()

    numeric_cols = df_agg_combined.columns.drop("발효일")
    df_agg_combined[numeric_cols] = df_agg_combined[numeric_cols].fillna(0).astype(int)
    df_agg_combined = df_agg_combined.sort_values("발효일").reset_index(drop=True)

    is_new_streak = df_agg_combined["발효일"].diff().dt.days != 1
    streak_group_id = is_new_streak.cumsum()
    df_agg_combined["alert_streak"] = df_agg_combined.groupby(streak_group_id).cumcount() + 1
    df_agg_combined.loc[df_agg_combined['has_alert'] == 0, 'alert_streak'] = 0

    df = df.merge(df_agg_combined, left_on="tm", right_on="발효일", how="left").drop(columns=["발효일"])

    all_alert_cols = list(df_agg_combined.columns.drop("발효일"))
    for col in all_alert_cols:
        df[col] = df[col].fillna(0).astype(int)

    lag_target_cols = all_alert_cols
    for col in lag_target_cols:
        df[f'{col}_D+1'] = df[col].shift(1).fillna(0).astype(int)

    return df