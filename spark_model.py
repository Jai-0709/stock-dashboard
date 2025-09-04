# spark_model.py
# PySpark data loader + Linear Regression model with RSI, Bollinger Bands, metrics,
# plus a pandas-based recursive forecast helper for demo forecasting.

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import yfinance as yf
import pandas as pd
import numpy as np

# sklearn for simple pandas-based forecasting
from sklearn.linear_model import LinearRegression as SKLinearRegression

def get_spark(app_name="StockPredictionApp"):
    # Run Spark locally using all cores
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName(app_name) \
        .getOrCreate()
    return spark


def load_stock_data(ticker, period="1y"):
    """
    Download historical OHLCV data using yfinance and return a pandas DataFrame.
    """
    data = yf.download(ticker, period=period, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    data = data.reset_index()

    # Flatten MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    expected = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    data = data[[c for c in data.columns if c in expected]]

    # Ensure Date is datetime
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def preprocess_pd(df_pd):
    """
    Compute moving averages, RSI, and Bollinger Bands using pandas.
    Returns a copy of df_pd with added columns:
    - MA50, MA200
    - RSI
    - BB_Mid, BB_Std, BB_Upper, BB_Lower
    """
    df_pd = df_pd.copy().reset_index(drop=True)

    # Moving averages
    df_pd["MA50"] = df_pd["Close"].rolling(window=50, min_periods=1).mean()
    df_pd["MA200"] = df_pd["Close"].rolling(window=200, min_periods=1).mean()

    # RSI (Relative Strength Index)
    delta = df_pd["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    rs = avg_gain / avg_loss
    df_pd["RSI"] = 100 - (100 / (1 + rs))
    # fill initial NaNs with neutral 50
    df_pd["RSI"] = df_pd["RSI"].fillna(50)

    # Bollinger Bands
    window = 20
    df_pd["BB_Mid"] = df_pd["Close"].rolling(window=window, min_periods=1).mean()
    df_pd["BB_Std"] = df_pd["Close"].rolling(window=window, min_periods=1).std().fillna(0)
    df_pd["BB_Upper"] = df_pd["BB_Mid"] + (2 * df_pd["BB_Std"])
    df_pd["BB_Lower"] = df_pd["BB_Mid"] - (2 * df_pd["BB_Std"])

    # Fill remaining NaNs sensibly
    df_pd = df_pd.fillna(method="ffill").fillna(0)
    return df_pd


def pandas_to_spark(spark, df_pd):
    """
    Convert pandas dataframe to spark dataframe and cast numeric columns to DoubleType.
    """
    sdf = spark.createDataFrame(df_pd)
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume",
              "MA50", "MA200", "RSI", "BB_Upper", "BB_Lower"]:
        if c in sdf.columns:
            sdf = sdf.withColumn(c, col(c).cast(DoubleType()))
    return sdf


def train_simple_lr(sdf):
    """
    Train a Linear Regression model predicting Close from features (same-day Close).
    Returns model, predictions, and evaluation metrics (rmse, r2).
    """
    feature_cols = [
        c for c in [
            "Open", "High", "Low", "Volume",
            "MA50", "MA200", "RSI", "BB_Upper", "BB_Lower"
        ] if c in sdf.columns
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(sdf).select("features", col("Close").alias("label"))

    train, test = data.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=50)
    model = lr.fit(train)

    predictions = model.transform(test)

    # Evaluate
    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    return model, predictions, {"rmse": rmse, "r2": r2}


# -----------------------------
# Forecast helper (pandas + sklearn)
# -----------------------------
def _build_feature_matrix_for_forecast(df_pd):
    """
    Given a preprocessed pandas df (with indicators), build feature matrix X and target y_next:
    - X: features at day t
    - y: Close at day t+1
    This is used to train a next-day predictor.
    """
    df = df_pd.copy().reset_index(drop=True)
    # target is next day's Close
    df["target_next_close"] = df["Close"].shift(-1)
    # drop last row because it has no next day target
    df = df.iloc[:-1].copy()

    feature_cols = [c for c in [
        "Open", "High", "Low", "Volume", "MA50", "MA200", "RSI", "BB_Upper", "BB_Lower"
    ] if c in df.columns]

    X = df[feature_cols].values
    y = df["target_next_close"].values
    return X, y, feature_cols


def train_forecast_model_pandas(df_pd):
    """
    Train a simple sklearn LinearRegression model to predict next-day Close
    using the indicators. Returns the trained sklearn model and feature column names.
    Operates on pandas DataFrame (expects preprocess_pd already applied).
    """
    X, y, feature_cols = _build_feature_matrix_for_forecast(df_pd)
    if len(X) == 0:
        # Not enough data
        model = SKLinearRegression()
        model.fit(np.zeros((1, len(feature_cols))), np.zeros(1))
        return model, feature_cols

    model = SKLinearRegression()
    model.fit(X, y)
    return model, feature_cols


def forecast_recursive(df_pd, model, feature_cols, days=7):
    """
    Recursively forecast the next `days` Close prices.
    Approach: start with the last available row in df_pd, then:
      - extract features for last row
      - predict next close
      - append a new row with predicted Close (and recompute indicators)
      - repeat
    Returns a pandas DataFrame with appended predicted rows (Date extended, Predicted flag).
    """
    df = df_pd.copy().reset_index(drop=True)

    # Make sure Date is datetime
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])

    last_date = df["Date"].iloc[-1]
    results = df.copy()
    predicted_rows = []

    for i in range(days):
        # Recompute indicators on current results to have up-to-date MAs/RSI/BB for the last row
        temp = preprocess_pd(results)
        last_row = temp.iloc[-1]

        # Build feature vector in same order as feature_cols
        feat = []
        for c in feature_cols:
            # if column missing, use 0
            feat.append(last_row[c] if c in last_row.index else 0.0)
        feat_arr = np.array(feat).reshape(1, -1)

        # Predict next day's close
        next_close = float(model.predict(feat_arr)[0])

        # Create next row: we set Open/High/Low to be last_close (naive), Volume to last volume (naive)
        next_date = last_date + pd.Timedelta(days=1)
        next_row = {
            "Date": next_date,
            "Open": last_row["Close"],
            "High": max(last_row["Close"], next_close),
            "Low": min(last_row["Close"], next_close),
            "Close": next_close,
            "Adj Close": next_close,
            "Volume": last_row.get("Volume", 0),
        }
        # append to results
        results = pd.concat([results, pd.DataFrame([next_row])], ignore_index=True)
        last_date = next_date
        predicted_rows.append(next_row)

    # Mark predicted rows
    results["Predicted"] = False
    results.loc[results.index >= (len(df)), "Predicted"] = True

    # Recompute indicators for full results and return
    results = preprocess_pd(results)
    return results


# -----------------------------
# convenience end-to-end runner
# -----------------------------
def run_for_ticker(ticker="AAPL", period="1y"):
    """
    Runs the full pipeline:
    - load data (pandas)
    - preprocess (indicators)
    - convert to spark, train same-day LR, get sample predictions & metrics
    - train pandas-based next-day forecast model (sklearn) for use in UI forecasting
    Returns:
      df_pd: preprocessed pandas DataFrame
      spark_model_info: (model, predictions, metrics) from train_simple_lr
      forecast_model_info: (sklearn_model, feature_cols)
    """
    spark = get_spark()
    df_pd = load_stock_data(ticker, period)
    if df_pd.empty:
        return pd.DataFrame(), None, None

    df_pd = preprocess_pd(df_pd)
    sdf = pandas_to_spark(spark, df_pd)
    model, predictions, metrics = train_simple_lr(sdf)

    # train pandas-based forecast (next-day)
    sk_model, feature_cols = train_forecast_model_pandas(df_pd)

    return df_pd, (model, predictions, metrics), (sk_model, feature_cols)
