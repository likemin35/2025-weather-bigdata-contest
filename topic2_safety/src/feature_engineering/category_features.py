import pandas as pd

def add_category_features(df, df_cat, target_subcats=None):
    df = df.copy()
    df_cat = df_cat.copy()
    df["tm"] = pd.to_datetime(df["tm"])
    df_cat["tm"] = pd.to_datetime(df_cat["tm"])

    if target_subcats is None:
        target_subcats = ["부상", "교통사고", "업무운행", "구급기타", "자연재해"]
    
    df_cat["has_call"] = 1

    # 동 단위 발생 여부 피처 생성
    df_cat_cat = df_cat.pivot_table(
        index=["tm", "address_gu", "sub_address"],
        columns="cat",
        values="has_call",
        aggfunc="max",
        fill_value=0
    ).reset_index()
    cols_to_rename_cat = df_cat_cat.columns.drop(["tm", "address_gu", "sub_address"])
    df_cat_cat.rename(columns={col: f"{col}_발생여부" for col in cols_to_rename_cat}, inplace=True)

    df_cat_sub = df_cat[df_cat["sub_cat"].isin(target_subcats)].pivot_table(
        index=["tm", "address_gu", "sub_address"],
        columns="sub_cat",
        values="has_call",
        aggfunc="max",
        fill_value=0
    ).reset_index()
    cols_to_rename_subcat = df_cat_sub.columns.drop(["tm", "address_gu", "sub_address"])
    df_cat_sub.rename(columns={col: f"{col}_발생여부" for col in cols_to_rename_subcat}, inplace=True)

    df_sub_address_features = df_cat_cat.merge(df_cat_sub, on=["tm", "address_gu", "sub_address"], how="outer")

    # 구 단위 발생 동 개수 피처 생성
    df_gu_counts = df_cat.groupby(["tm", "address_gu", "cat"])["sub_address"].nunique().unstack(level="cat", fill_value=0)
    df_gu_counts.columns = [f"{cat}_발생동개수" for cat in df_gu_counts.columns]
    df_gu_counts.reset_index(inplace=True)
    
    # 병합
    df = df.merge(df_sub_address_features, on=["tm", "address_gu", "sub_address"], how="left")
    df = df.merge(df_gu_counts, on=["tm", "address_gu"], how="left")

    sub_address_cols_to_fill = df_sub_address_features.columns.drop(["tm", "address_gu", "sub_address"])
    gu_cols_to_fill = df_gu_counts.columns.drop(["tm", "address_gu"])
    
    for col in sub_address_cols_to_fill:
        df[col] = df[col].fillna(0).astype("int8")
        
    for col in gu_cols_to_fill:
        df[col] = df[col].fillna(0).astype("int8")

    return df