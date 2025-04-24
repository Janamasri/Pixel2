import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

from prophet import Prophet
import cmdstanpy

# Ensure CmdStan is installed (do this once in Streamlit)
if not cmdstanpy.cmdstan_path():
    cmdstanpy.install_cmdstan()


# ================================
# Pixel Digital - Growth Dashboard (Extended Version)
# ================================

# Set page configuration
st.set_page_config(
    page_title="Pixel Digital",
    layout="wide"
)

# === Inject Custom CSS ===
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background-color: #ffffff;
        }
        h1, h2, h3, h4 {
            color: #24486B;
        }
        .big-title {
            font-size: 48px;
            color: #24486B;
            text-align: center;
            font-weight: bold;
        }
        .blue-line {
            border: none;
            height: 2px;
            background-color: #24486B;
            margin: 30px 0;
        }
    </style>
""", unsafe_allow_html=True)

# === Sidebar Navigation ===
page = st.sidebar.radio(
    " ",
    ["Home", "KPI Dashboard", "Forecasting Simulator"]
)

# === HOME PAGE ===
if page == "Home":
    st.image("Pixel.jpg", width=300)
    st.markdown('<div class="big-title">Pixel Digital</div>', unsafe_allow_html=True)
    st.markdown('<hr class="blue-line"/>', unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: justify; font-size: 18px;'>
    <b>Pixel Digital Office Equipment</b> is Lebanon's trusted partner for advanced IT and imaging solutions, powered by decades of excellence, resilience, and innovation. We deliver cutting-edge technologies, strategic client services, and sustainable growth models tailored for a volatile economy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Key Highlights")
    st.markdown("""
    - Official distributor of RICOH technologies in Lebanon  
    - Serving NGOs, education, public sector, and corporate clients  
    - Expertise across imaging, document solutions, and IT systems  
    - Proven adaptability across economic crises and sectoral shifts  
    """)

    st.markdown("### Vision & Mission")
    st.markdown("""
    - Empower Lebanese industries with world-class digital solutions  
    - Drive client success through forecasting and strategic planning  
    - Expand operational excellence across Lebanon and regional markets  
    """)

    st.markdown('<hr class="blue-line"/>', unsafe_allow_html=True)
    st.info("This digital dashboard is part of Pixel Digital's transformation toward predictive, insight-driven growth.")

# === KPI DASHBOARD ===
elif page == "KPI Dashboard":
    st.title("Pixel Digital KPI Dashboard")
    st.markdown("Explore interactive Tableau dashboards showcasing sales, client dynamics, sector trends, and internal performance.")

    st.subheader("1 – Sales by Year and Sector")
    components.iframe(
        "https://public.tableau.com/views/Salesbyyearandsector/Dashboard1?:embed=y&:display_count=yes&:showVizHome=no",
        height=827, width=1600
    )

    st.subheader("2 – Client Portfolio Insights")
    components.iframe(
        "https://public.tableau.com/views/ClientAnalysis_17451765178320/Dashboard2?:embed=y&:display_count=yes&:showVizHome=no",
        height=827, width=1600
    )

    st.subheader("3 – Industry and Product Category Trends")
    components.iframe(
        "https://public.tableau.com/views/IndustryandCategory/Dashboard3?:embed=y&:display_count=yes&:showVizHome=no",
        height=827, width=1600
    )

    st.subheader("4 – Sales Team Performance")
    components.iframe(
        "https://public.tableau.com/views/SalesbySalesperson_17451766336230/Dashboard4?:embed=y&:display_count=yes&:showVizHome=no",
        height=827, width=1600
    )


# === FORECASTING SIMULATOR ===
elif page == "Forecasting Simulator":
    st.title("📊 Forecasting Model Simulator")
    st.markdown("Upload your Excel data and run the Prophet + XGBoost **weighted ensemble forecast** interactively.")

    uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
        df["jvdate"] = pd.to_datetime(df["jvdate"])

        # Monthly aggregation
        monthly = df.groupby(pd.Grouper(key="jvdate", freq="M")).agg({
            "Net_Amount_USD": "sum",
            "Exchange Rate (LBP to USD)": "mean"
        }).reset_index()
        monthly.columns = ["ds", "y", "exchange_rate"]
        monthly.dropna(inplace=True)

        # Prophet
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.8)
        prophet_model.fit(monthly[["ds", "y"]])
        future = prophet_model.make_future_dataframe(periods=24, freq='M')
        forecast_prophet = prophet_model.predict(future)[["ds", "yhat"]]

        # XGBoost features
        monthly["month"] = monthly["ds"].dt.month
        monthly["quarter"] = monthly["ds"].dt.quarter
        monthly["lag1"] = monthly["y"].shift(1)
        monthly["lag2"] = monthly["y"].shift(2)
        monthly["rolling_mean_3"] = monthly["y"].rolling(3).mean()
        monthly.dropna(inplace=True)

        future_dates = pd.date_range(start=monthly["ds"].max() + pd.DateOffset(months=1), periods=24, freq='M')
        future_df = pd.DataFrame({"ds": future_dates})
        future_df["month"] = future_df["ds"].dt.month
        future_df["quarter"] = future_df["ds"].dt.quarter
        future_df["exchange_rate"] = monthly["exchange_rate"].mean()
        future_df["lag1"] = monthly["y"].iloc[-1]
        future_df["lag2"] = monthly["y"].iloc[-2]
        future_df["rolling_mean_3"] = monthly["y"].iloc[-3:].mean()

        features = ["month", "quarter", "exchange_rate", "lag1", "lag2", "rolling_mean_3"]
        X = monthly[features]
        y = monthly["y"]
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(X, y)
        future_df["xgb_forecast"] = xgb_model.predict(future_df[features])

        ensemble = forecast_prophet.merge(future_df[["ds", "xgb_forecast"]], on="ds")
        ensemble["weighted_forecast"] = 0.7 * ensemble["yhat"] + 0.3 * ensemble["xgb_forecast"]

        # Evaluate
        prophet_insample = prophet_model.predict(monthly[["ds"]])[["ds", "yhat"]]
        xgb_insample = monthly[["ds"]].copy()
        xgb_insample["xgb_forecast"] = xgb_model.predict(X)
        eval_df = monthly[["ds", "y"]].merge(prophet_insample, on="ds").merge(xgb_insample, on="ds")
        eval_df["weighted_forecast"] = 0.7 * eval_df["yhat"] + 0.3 * eval_df["xgb_forecast"]
        eval_last12 = eval_df.dropna().iloc[-12:]
        rmse = np.sqrt(mean_squared_error(eval_last12["y"], eval_last12["weighted_forecast"]))
        mape = mean_absolute_percentage_error(eval_last12["y"], eval_last12["weighted_forecast"]) * 100

        st.subheader("📉 Evaluation")
        st.write(f"**RMSE (last 12 months):** {rmse:.2f}")
        st.write(f"**MAPE (last 12 months):** {mape:.2f}%")

        st.subheader("📈 Forecast Plot")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(monthly["ds"], monthly["y"], label="Historical", color="blue")
        ax.plot(ensemble["ds"], ensemble["weighted_forecast"], label="Ensemble Forecast", color="green", linestyle="--")
        ax.plot(ensemble["ds"], ensemble["yhat"], label="Prophet", color="purple", linestyle=":")
        ax.plot(ensemble["ds"], ensemble["xgb_forecast"], label="XGBoost", color="orange", linestyle=":")
        ax.axvline(x=monthly["ds"].max(), color="gray", linestyle="--", label="Forecast Start")
        ax.set_title("Ensemble Sales Forecast (2025–2026)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales in USD")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
