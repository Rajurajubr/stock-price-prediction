# StockSight AI — Stock Price Prediction & Candlestick Analysis

An educational BCA project starter built with Flask, Python, scikit-learn, yfinance, HTML, CSS and JavaScript.

## Features
- Fetches historical daily OHLCV data for a ticker using Yahoo Finance via `yfinance`.
- Trains a time-ordered Random Forest regression demo to estimate the next closing price.
- Displays Bullish / Bearish / Neutral direction based on the estimated percentage move.
- Shows a recent candle body observation and a holdout mean absolute error (MAE).
- Accepts chart screenshots and runs a basic red/green pixel-count demonstration.
- Responsive dark dashboard with a comparison chart.

## Setup (Windows)
1. Install Python 3.10 or newer.
2. Extract the ZIP file.
3. Open Command Prompt in the extracted `stock_price_prediction_project` folder.
4. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Start the website:
   ```bash
   python app.py
   ```
7. Open `http://127.0.0.1:5000` in your browser.

## Example ticker symbols
- Indian NSE: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- US: `AAPL`, `MSFT`, `NVDA`

Ticker availability and data access depend on Yahoo Finance and your internet connection.

## Important limitations
- This is an academic demonstration, not a validated trading model.
- A next-day price estimate is uncertain and can be wrong. The model is trained on historical data only and does not know future events.
- The reported holdout MAE is a simple historical test metric, not a confidence percentage or guarantee.
- The image feature only counts approximate red and green pixels. It does not genuinely identify candlestick patterns or read chart prices. A robust version needs chart-specific computer vision and/or the underlying OHLCV data.
- "Worth analysis" is intentionally not presented as a buy/sell recommendation. Do not use this project to make real investment decisions.

## Suggested improvements for a later version
- Add a proper candlestick chart from OHLCV data.
- Add RSI, MACD, SMA and EMA indicators.
- Build a labeled image dataset and train/evaluate a candle-pattern classifier.
- Use walk-forward validation and compare against simple baseline forecasts.
- Store prediction history in SQLite.
- Add unit tests, rate limiting, secure upload naming, and production deployment settings.
