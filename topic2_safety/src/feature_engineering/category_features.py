def add_category_features(df, df_cat):
    df_cat = df_cat.copy()
    df_cat["tm"] = pd.to_datetime(df_cat["tm"])

    df_cat["has_call"] = 1

    df_cat_cat = (
        df_cat.pivot_table(
            index=["tm", "address_gu", "sub_address"],
            columns="cat",
            values="has_call",
            aggfunc="max",
            fill_value=0
        )
        .reset_index()
    )

    target_subcats = ["부상", "교통사고", "업무운행", "구급기타", "자연재해"]
    df_cat_sub = (
        df_cat[df_cat["sub_cat"].isin(target_subcats)]
        .pivot_table(
            index=["tm", "address_gu", "sub_address"],
            columns="sub_cat",
            values="has_call",
            aggfunc="max",
            fill_value=0
        )
        .reset_index()
    )

    df_cat_binary = df_cat_cat.merge(df_cat_sub, on=["tm", "address_gu", "sub_address"], how="outer").fillna(0)
    df = df.merge(df_cat_binary, on=["tm", "address_gu", "sub_address"], how="left")

    return df