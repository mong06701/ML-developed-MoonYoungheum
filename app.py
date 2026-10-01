# TODO
# 지금까지 작성한 코드를 하나의 app.py로 연결하세요.

import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")   # ARIMA 경고 메시지 숨김

# 1. 제목
st.title("Bike Sharing 시계열 예측")
st.write("월별 자전거 대여량을 확인하고 미래 값을 예측합니다.")

# 2. 파일 업로드
uploaded_file = st.file_uploader("Bike_Sharing_Demand.csv 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    # 3. DataFrame으로 불러오기
    df = pd.read_csv(uploaded_file)

    st.subheader("원본 데이터")
    st.dataframe(df)

    # 4. datetime 변환
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 5. 정렬 후 월별 count 합계
    df = df.sort_values("datetime")
    df["month"] = df["datetime"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")[["count"]].sum()

    # 6. 월별 데이터 표 + 선 그래프
    st.subheader("월별 대여량")
    st.dataframe(monthly)
    st.line_chart(monthly)

    # 7. 예측 기간 선택
    forecast_period = st.selectbox("예측 기간(개월)을 선택하세요", [1, 3, 6])

    # 8. 전체 월별 count로 ARIMA(2, 1, 1) 학습
    model = ARIMA(monthly["count"], order=(2, 1, 1))
    model_fit = model.fit()

    # 9. 선택한 기간만큼 예측
    forecast = model_fit.forecast(steps=forecast_period)

    last_month = monthly.index[-1]
    future_dates = pd.date_range(
        start=last_month + pd.DateOffset(months=1),
        periods=forecast_period,
        freq="MS"   # 매월 1일
    )

    forecast_df = pd.DataFrame({
        "month": future_dates,
        "forecast": forecast.values
    }).set_index("month")

    st.subheader(f"예측 결과")
    st.dataframe(forecast_df.round(0))

    chart_df = pd.concat([
        monthly.rename(columns={"count": "actual"}),
        forecast_df
    ], axis=1)

    st.subheader("실제값과 미래 예측값")
    st.line_chart(chart_df)