# Feature Engineering 모듈

날짜, 전년도 신고, 날씨, 기상특보, 신고 분류 기반 파생 변수를 자동으로 생성합니다.

## 📦 구성

```
feature_engineering/
├── __init__.py
├── date_features.py
├── prev_year_features.py
├── weather_features.py
├── alert_features.py
├── category_features.py
└── full_pipeline.py
```

## 💡 주요 함수

- `feature_engineering(df, df_cat, df_wrn)`

  → 전체 피처 엔지니어링 파이프라인을 한 번에 적용

## 🖥️ 사용법 안내

### ▶️ Jupyter Notebook (로컬)

1. 저장소 클론

```bash
git clone -b safety https://github.com/s0nghyunje0ng/2025-weather-bigdata-contest.git
cd weather-bigdata-contest
```

2. 경로 추가

```python
import sys
sys.path.append("../topic2_safety/src")
```

2. 사용 예시

```python
from feature_engineering.full_pipeline import feature_engineering

df_call = pd.read_csv("../topic2_safety/data/processed/preprocessed_C_neighbor_only.csv")
df_cat = pd.read_csv("../topic2_safety/data/processed/preprocessed_cat119.csv")
df_wrn = pd.read_csv("../topic2_safety/data/processed/busan_weather_warning.csv")

df = feature_engineering(df_call, df_cat, df_wrn)
```

---

### ▶️ Google Colab

1. 저장소 클론

```python
!git clone -b safety https://github.com/s0nghyunje0ng/2025-weather-bigdata-contest.git
```

2. 경로 추가

```python
import sys
sys.path.append("/content/2025-weather-bigdata-contest/topic2_safety/src")
```

3. 사용 예시

```python
from feature_engineering.full_pipeline import feature_engineering

base_url = "https://raw.githubusercontent.com/s0nghyunje0ng/2025-weather-bigdata-contest/safety/topic2_safety/data/processed/"

df_call = pd.read_csv(base_url + "preprocessed_C_neighbor_only.csv")
df_cat = pd.read_csv(base_url + "preprocessed_cat119.csv")
df_wrn = pd.read_csv(base_url + "busan_weather_warning.csv")

df = feature_engineering(df_call, df_cat, df_wrn)
```
