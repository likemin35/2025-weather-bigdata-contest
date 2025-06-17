import pandas as pd
from datetime import timedelta

def create_prev_year_lookup(df_train):
    """1. 훈련 데이터만을 사용하여 전년도 인접일 평균 call_count 조회 테이블을 만듭니다."""
    df_prev = df_train[["tm", "sub_address", "call_count"]].copy()
    df_prev["tm"] += pd.DateOffset(years=1)
    
    # 5일치 윈도우 데이터 생성
    df_all = pd.concat([
        df_prev.assign(tm=df_prev["tm"] + timedelta(days=d)) for d in [-2, -1, 0, 1, 2]
    ])
    
    # 전년도 인접일 평균 call_count 계산
    df_avg_lookup = (
        df_all.groupby(["tm", "sub_address"], observed=False)["call_count"]
        .mean()
        .reset_index()
        .rename(columns={"call_count": "call_count_prev_year"})
    )
    return df_avg_lookup

def apply_prev_year_features(df, lookup_table):
    """2. 생성된 조회 테이블을 사용하여 피처를 추가합니다."""
    df = df.copy()
    
    # 훈련 데이터로 만든 조회 테이블을 merge
    df = df.merge(lookup_table, on=["tm", "sub_address"], how="left")

    # 전년도 데이터 존재 유무 플래그 생성
    df["has_prev_year_data"] = (df["year"] != 2020).astype("int8")

    # 결측치 처리
    mask_to_fill = (df["year"] != 2020)
    df.loc[mask_to_fill, "call_count_prev_year"] = df.loc[mask_to_fill, "call_count_prev_year"].fillna(0)
    
    df["call_count_prev_year"] = df["call_count_prev_year"].fillna(-1)
    
    return df