# app_cloud.py
# Streamlit Stock Market Analyzer (Light Version - No PySpark)

import streamlit as st
import pandas as pd
import datetime, time, random
import plotly.graph_objs as go
from fpdf import FPDF
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Stock Market Analyzer", layout="wide")

# --- Global CSS Styling ---
st.markdown("""
    <style>
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
def create_pdf_bytes(ticker, display_forecast):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Stock Report - {ticker}", ln=True, align="C")

    pdf.set_font("Arial", size=10)
    pdf.cell(40, 10, "Date", 1)
    pdf.cell(40, 10, "Predicted Close", 1)
    pdf.cell(30, 10, "RSI", 1)
    pdf.ln()

    for _, row in display_forecast.iterrows():
        pdf.cell(40, 10, str(row["Date"].date()), 1)
        pdf.cell(40, 10, f"{row['Predicted_Close']:.2f}", 1)
        pdf.cell(30, 10, f"{row['RSI']:.2f}", 1)
        pdf.ln()

    pdf_str = pdf.output(dest="S")
    return pdf_str.encode("latin-1")

# --- RSI Helper ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Welcome Section ---
if not run_btn:
    today = datetime.datetime.now().strftime("%A, %d %B %Y")

    st.markdown(f"""
    # 👋 Welcome to **Stock Market Analyzer**  
    ### Today is {today} – AI-powered insights at your fingertips 🚀
    """)

    # --- Stock Animation with Red/Green History ---
    price_placeholder = st.empty()
    chart_placeholder = st.empty()

    price = 100.0
    prices = [price]
    times = [0]

    i = 0
    for _ in range(40):  # limit animation cycles
        prev_price = prices[-1]
        price += random.uniform(-1, 1)
        i += 1
        prices.append(price)
        times.append(i * 0.2)

        arrow = "🔺" if price >= prev_price else "🔻"
        price_placeholder.metric("Stock Price", f"${price:.2f}", arrow)

        if len(prices) > 50:
            prices = prices[-50:]
            times = times[-50:]

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
        fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis=dict(showgrid=False, showticklabels=False),
                          yaxis=dict(showgrid=False))
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.5)

# --- Main Logic ---
if run_btn:
    with st.spinner("Fetching data and training model..."):
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            st.error("No data found for ticker: " + ticker)
        else:
            df.reset_index(inplace=True)
            df["RSI"] = compute_rsi(df["Close"]).fillna(50)

            # Train simple Linear Regression for forecasting
            df["t"] = np.arange(len(df))
            X = df[["t"]]
            y = df["Close"]
            model = LinearRegression().fit(X, y)

            future_t = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
            preds = model.predict(future_t)

            future_dates = pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds, "RSI": 50})

            # --- Tabs ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📑 Forecast", "💰 Portfolio", "📥 Reports"])

            with tab1:
                # Historical candlestick
                candlestick_fig = go.Figure()
                candlestick_fig.add_trace(go.Candlestick(
                    x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    name="OHLC", increasing_line_color="green", decreasing_line_color="red"
                ))
                candlestick_fig.update_layout(template="plotly_dark", title=dict(text="Historical Candlestick", x=0.5))
                st.plotly_chart(candlestick_fig, use_container_width=True)

            with tab2:
                st.subheader("🔮 Forecasted Prices")
                st.dataframe(forecast_df)
                forecast_fig = go.Figure()
                forecast_fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted_Close"],
                                                  mode="lines+markers", name="Predicted Close", line=dict(color="blue")))
                forecast_fig.update_layout(template="plotly_dark", height=400, title="Future Forecast")
                st.plotly_chart(forecast_fig, use_container_width=True)

            with tab3:
                last_close = df["Close"].iloc[-1]
                shares = investment / last_close
                future_values = shares * forecast_df["Predicted_Close"].values
                sim_df = pd.DataFrame({"Date": forecast_df["Date"], "Predicted_Close": forecast_df["Predicted_Close"],
                                       "Portfolio_Value": future_values})
                st.subheader("💰 Portfolio Simulation")
                st.dataframe(sim_df)
                port_fig = go.Figure()
                port_fig.add_trace(go.Scatter(x=sim_df["Date"], y=sim_df["Portfolio_Value"],
                                              mode="lines+markers", name="Portfolio Value", line=dict(color="green")))
                port_fig.update_layout(template="plotly_dark", height=400, title="Portfolio Value Simulation")
                st.plotly_chart(port_fig, use_container_width=True)

            with tab4:
                csv = forecast_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Forecast as CSV", csv, file_name=f"{ticker}_forecast.csv", mime="text/csv")
                pdf_bytes = create_pdf_bytes(ticker, forecast_df)
                st.download_button("⬇️ Download Forecast Report as PDF", pdf_bytes, file_name=f"{ticker}_report.pdf", mime="application/pdf")

            
