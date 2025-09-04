# app.py
# Streamlit Stock Market Analyzer with Full Red/Green Line Animation

import streamlit as st
import pandas as pd
import datetime, time, random
import plotly.graph_objs as go
from fpdf import FPDF

from spark_model import (
    get_spark,
    load_stock_data,
    preprocess_pd,
    pandas_to_spark,
    train_simple_lr,
    train_forecast_model_pandas,
    forecast_recursive,
)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Stock Market Analyzer", layout="wide")

# --- Global CSS Styling ---
st.markdown("""
    <style>
    /* Futuristic Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
        color: white;
        border-right: 2px solid #00f2fe;
        box-shadow: 2px 0px 20px rgba(0,0,0,0.4);
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        color: #00f2fe;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("<div class='sidebar-title'>⚙️ Configuration</div>", unsafe_allow_html=True)
ticker = st.sidebar.text_input("Stock ticker", "AAPL")
period = st.sidebar.selectbox("Historical period", ["6mo", "1y", "2y", "5y", "max"], index=1)
forecast_days = st.sidebar.number_input("Forecast days", min_value=1, max_value=60, value=7)
investment = st.sidebar.number_input("Investment amount (₹)", min_value=1000, value=10000, step=1000)
run_btn = st.sidebar.button("🚀 Run Analysis")

# --- PDF Helper ---
def create_pdf_bytes(ticker, display_forecast, metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Stock Report - {ticker}", ln=True, align="C")
    pdf.cell(200, 10, txt=f"RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}", ln=True)

    pdf.set_font("Arial", size=10)
    pdf.cell(40, 10, "Date", 1)
    pdf.cell(40, 10, "Predicted Close", 1)
    pdf.cell(40, 10, "BB Upper", 1)
    pdf.cell(40, 10, "BB Lower", 1)
    pdf.cell(30, 10, "RSI", 1)
    pdf.ln()

    for _, row in display_forecast.iterrows():
        pdf.cell(40, 10, str(row["Date"].date()), 1)
        pdf.cell(40, 10, f"{row['Predicted_Close']:.2f}", 1)
        pdf.cell(40, 10, f"{row['BB_Upper']:.2f}", 1)
        pdf.cell(40, 10, f"{row['BB_Lower']:.2f}", 1)
        pdf.cell(30, 10, f"{row['RSI']:.2f}", 1)
        pdf.ln()

    pdf_str = pdf.output(dest="S")
    return pdf_str.encode("latin-1")

# --- Welcome Section ---
if not run_btn:
    today = datetime.datetime.now().strftime("%A, %d %B %Y")

    st.markdown(f"""
    # 👋 Welcome to **Stock Market Analyzer**  
    ### Today is {today} – AI-powered insights at your fingertips 🚀
    """)

    # --- Stock Animation with Full Red/Green History ---
    price_placeholder = st.empty()
    chart_placeholder = st.empty()

    price = 100.0
    prices = [price]
    times = [0]

    i = 0
    while not run_btn:
        prev_price = prices[-1]
        price += random.uniform(-1, 1)  # simulate movement
        i += 1
        prices.append(price)
        times.append(i * 0.2)  # sharper x-axis

        # Update stock value
        arrow = "🔺" if price >= prev_price else "🔻"
        price_placeholder.metric("Stock Price", f"${price:.2f}", arrow)

        # Keep last 50 points
        if len(prices) > 50:
            prices = prices[-50:]
            times = times[-50:]

        # Create chart with colored segments
        fig = go.Figure()
        for j in range(1, len(prices)):
            seg_color = "lime" if prices[j] >= prices[j-1] else "red"
            fig.add_trace(go.Scatter(
                x=[times[j-1], times[j]],
                y=[prices[j-1], prices[j]],
                mode="lines",
                line=dict(color=seg_color, width=3),
                showlegend=False
            ))

        # Last point marker
        fig.add_trace(go.Scatter(
            x=[times[-1]], y=[prices[-1]],
            mode="markers",
            marker=dict(color="lime" if price >= prev_price else "red", size=8),
            showlegend=False
        ))

        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.8)

# --- Main Logic (Run Analysis) ---
if run_btn:
    with st.spinner("Fetching data and training models..."):
        df_pd = load_stock_data(ticker, period=period)
        if df_pd.empty:
            st.error("No data found for ticker: " + ticker)
        else:
            df_pd = preprocess_pd(df_pd)

            spark = get_spark()
            sdf = pandas_to_spark(spark, df_pd)
            spark_model, spark_predictions, spark_metrics = train_simple_lr(sdf)

            sk_model, feature_cols = train_forecast_model_pandas(df_pd)
            forecasted_df = forecast_recursive(df_pd, sk_model, feature_cols, days=int(forecast_days))
            predicted_only = forecasted_df[forecasted_df["Predicted"]].copy().reset_index(drop=True)
            display_forecast = predicted_only[["Date", "Close", "BB_Upper", "BB_Lower", "RSI"]].rename(columns={"Close": "Predicted_Close"})

            # --- Metrics ---
            st.subheader("📊 Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{spark_metrics['rmse']:.4f}")
            col2.metric("R² Score", f"{spark_metrics['r2']:.4f}")

            # --- Tabs ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📑 Forecast", "💰 Portfolio", "📥 Reports"])

            with tab1:
                candlestick_fig = go.Figure()
                candlestick_fig.add_trace(
                    go.Candlestick(
                        x=forecasted_df["Date"],
                        open=forecasted_df["Open"],
                        high=forecasted_df["High"],
                        low=forecasted_df["Low"],
                        close=forecasted_df["Close"],
                        name="OHLC",
                        increasing_line_color="green",
                        decreasing_line_color="red",
                    )
                )
                if "MA50" in forecasted_df.columns:
                    candlestick_fig.add_trace(go.Scatter(x=forecasted_df["Date"], y=forecasted_df["MA50"], name="MA50"))
                if "MA200" in forecasted_df.columns:
                    candlestick_fig.add_trace(go.Scatter(x=forecasted_df["Date"], y=forecasted_df["MA200"], name="MA200"))
                if "BB_Upper" in forecasted_df.columns and "BB_Lower" in forecasted_df.columns:
                    candlestick_fig.add_trace(go.Scatter(x=forecasted_df["Date"], y=forecasted_df["BB_Upper"], name="BB Upper", line=dict(dash="dash")))
                    candlestick_fig.add_trace(go.Scatter(x=forecasted_df["Date"], y=forecasted_df["BB_Lower"], name="BB Lower", line=dict(dash="dash")))
                candlestick_fig.update_layout(template="plotly_dark", title=dict(text="Candlestick + Indicators", x=0.5))
                st.plotly_chart(candlestick_fig, use_container_width=True)

                st.subheader("🔍 Sample predictions (Spark test set)")
                sample_pdf = spark_predictions.select("prediction", "label").limit(20).toPandas()
                st.dataframe(sample_pdf)

            with tab2:
                st.dataframe(display_forecast)
                if not display_forecast.empty:
                    forecast_fig = go.Figure()
                    forecast_fig.add_trace(go.Scatter(
                        x=display_forecast["Date"], y=display_forecast["Predicted_Close"],
                        mode="lines+markers", name="Predicted Close", line=dict(color="blue")
                    ))
                    forecast_fig.update_layout(template="plotly_dark", height=400, title="Future Predicted Prices")
                    st.plotly_chart(forecast_fig, use_container_width=True)

            with tab3:
                if not display_forecast.empty:
                    last_close = df_pd["Close"].iloc[-1]
                    shares = investment / last_close
                    future_values = shares * display_forecast["Predicted_Close"].values
                    sim_df = pd.DataFrame({
                        "Date": display_forecast["Date"],
                        "Predicted_Close": display_forecast["Predicted_Close"],
                        "Portfolio_Value": future_values
                    })
                    st.dataframe(sim_df)
                    port_fig = go.Figure()
                    port_fig.add_trace(go.Scatter(
                        x=sim_df["Date"], y=sim_df["Portfolio_Value"],
                        mode="lines+markers",
                        name="Portfolio Value",
                        line=dict(color="green")
                    ))
                    port_fig.update_layout(template="plotly_dark", height=400, title="Portfolio Value Simulation")
                    st.plotly_chart(port_fig, use_container_width=True)

            with tab4:
                csv = display_forecast.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Forecast as CSV", csv, file_name=f"{ticker}_forecast.csv", mime="text/csv")
                pdf_bytes = create_pdf_bytes(ticker, display_forecast, spark_metrics)
                st.download_button("⬇️ Download Forecast Report as PDF", pdf_bytes, file_name=f"{ticker}_report.pdf", mime="application/pdf")

            
