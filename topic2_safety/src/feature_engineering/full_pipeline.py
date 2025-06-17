from .date_features import add_date_features
from .weather_features import add_weather_features
from .alert_features import add_alert_features
from .category_features import add_category_features
from .prev_year_features import create_prev_year_lookup, apply_prev_year_features

# --- 1. 학습용 파이프라인 ---
def feature_engineering_fit(df_call_train, df_cat_train, df_wrn):
    """훈련 데이터에 대한 피처 엔지니어링을 수행하고, 예측에 필요한 객체(lookup_table)를 반환"""
    
    df = add_date_features(df_call_train)
    df = add_weather_features(df)
    df = add_alert_features(df, df_wrn)
    df = add_category_features(df, df_cat_train) 
    
    # 전년도 조회 테이블은 '훈련 데이터'로만 생성
    lookup_table = create_prev_year_lookup(df_call_train) 
    df = apply_prev_year_features(df, lookup_table)
    
    print("훈련 데이터 피처 엔지니어링 완료")
    return df, lookup_table

# --- 2. 예측(테스트)용 파이프라인 ---
def feature_engineering_transform(df_call_test, df_cat_test, df_wrn, lookup_table):
    """예측 데이터에 대한 피처 엔지니어링을 수행. 학습 단계에서 생성된 lookup_table을 사용"""

    df = add_date_features(df_call_test)
    df = add_weather_features(df)
    df = add_alert_features(df, df_wrn)
    df = add_category_features(df, df_cat_test)
    
    # 전년도 조회 테이블은 '학습 단계에서 생성된 것'을 그대로 적용
    df = apply_prev_year_features(df, lookup_table) 
    
    print("테스트 데이터 피처 엔지니어링 완료")
    return df

# --- 최종 사용 예시 ---
# # 1. 훈련 단계
# train_featured, prev_year_lookup = feature_engineering_fit(train_call, train_cat, wrn_raw)

# # 2. 예측 단계
# test_featured = feature_engineering_transform(test_call, test_cat, wrn_raw, prev_year_lookup)