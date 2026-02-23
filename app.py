import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


st.title("📈 AAPL Stock Price Forecasting App ")


uploaded_file = "AAPL (4) (3).csv"  #

df = pd.read_csv(uploaded_file)

df.columns = df.columns.str.strip()
if "Date" not in df.columns:
    st.error(" The CSV must contain a 'Date' column.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.error(" End date must be after start date.")
    st.stop()

filtered_df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

st.subheader("📊 Historical AAPL Stock Prices")
st.line_chart(filtered_df.set_index("Date")["Close"])

st.subheader(" ARIMA 30-Day Forecast")

data_series = filtered_df["Close"]

try:
    model = ARIMA(data_series, order=(5, 1, 0))
    model_fit = model.fit()

    forecast_steps = 30
    forecast = model_fit.forecast(steps=forecast_steps)


    future_dates = pd.date_range(filtered_df["Date"].max() + timedelta(days=1), periods=forecast_steps)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": forecast})

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(filtered_df["Date"], filtered_df["Close"], label="Historical", color="blue")
    ax.plot(forecast_df["Date"], forecast_df["Predicted Close"], label="Forecast", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.set_title("AAPL Stock Price Forecast (Next 30 Days)")
    ax.legend()
    st.pyplot(fig)

    st.write("### 📅 Forecasted Prices")
    st.dataframe(forecast_df)

except Exception as e:
    st.error(f"Model training or forecasting failed: {e}")
