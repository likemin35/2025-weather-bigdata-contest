from .date_features import add_date_features
from .prev_year_features import add_prev_year_features
from .weather_features import add_weather_features
from .alert_features import add_alert_features
from .category_features import add_category_features

def feature_engineering(df, df_cat, df_wrn):
    df = add_date_features(df)
    df = add_prev_year_features(df)
    df = add_weather_features(df)
    df = add_alert_features(df, df_wrn)
    df = add_category_features(df, df_cat)
    return df