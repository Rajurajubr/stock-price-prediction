from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_market_data(symbol, period="1y"):
    """Fetch historical OHLCV data. Requires internet and yfinance."""
    try:
        import yfinance as yf
        data = yf.download(symbol.strip().upper(), period=period, interval="1d",
                           auto_adjust=True, progress=False)
        if data is None or data.empty:
            raise ValueError("No market data found for this symbol.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        data = data[needed].dropna()
        if len(data) < 40:
            raise ValueError("Not enough historical data to build a meaningful demo model.")
        return data
    except ImportError:
        raise ValueError("The yfinance package is not installed. Run: pip install -r requirements.txt")
    except Exception as exc:
        raise ValueError(str(exc))

def make_features(data):
    d = data.copy()
    d["return_1"] = d["Close"].pct_change()
    d["return_3"] = d["Close"].pct_change(3)
    d["ma_5"] = d["Close"].rolling(5).mean()
    d["ma_10"] = d["Close"].rolling(10).mean()
    d["volatility_5"] = d["return_1"].rolling(5).std()
    d["volume_change"] = d["Volume"].pct_change()
    d["range_pct"] = (d["High"] - d["Low"]) / d["Close"]
    d["target"] = d["Close"].shift(-1)
    return d.replace([np.inf, -np.inf], np.nan).dropna()

def predict_next(data):
    featured = make_features(data)
    if len(featured) < 20:
        raise ValueError("Not enough clean historical rows for prediction.")
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "return_1",
                    "return_3", "ma_5", "ma_10", "volatility_5",
                    "volume_change", "range_pct"]
    X = featured[feature_cols]
    y = featured["target"]
    # Time-ordered holdout; never randomly shuffle time-series observations.
    split = max(int(len(featured) * 0.8), len(featured) - 20)
    split = min(split, len(featured) - 1)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=180, max_depth=5,
                                  min_samples_leaf=2, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    predicted = float(model.predict(X.iloc[[-1]])[0])
    actual = float(data["Close"].iloc[-1])
    # A simple holdout error indicator, not a probability of being correct.
    test_pred = model.predict(X.iloc[split:])
    mae = float(np.mean(np.abs(test_pred - y.iloc[split:]))) if len(test_pred) else None
    pct = ((predicted - actual) / actual) * 100 if actual else 0
    direction = "Bullish" if pct > 0.15 else "Bearish" if pct < -0.15 else "Neutral"
    recent = data.tail(5)
    last = recent.iloc[-1]
    body = abs(float(last["Close"]) - float(last["Open"]))
    candle_range = max(float(last["High"]) - float(last["Low"]), 1e-9)
    body_ratio = body / candle_range
    candle_note = "Small-body candle; momentum is not decisive."
    if float(last["Close"]) > float(last["Open"]) and body_ratio > 0.6:
        candle_note = "Recent candle is a strong bullish-bodied candle."
    elif float(last["Close"]) < float(last["Open"]) and body_ratio > 0.6:
        candle_note = "Recent candle is a strong bearish-bodied candle."
    elif body_ratio < 0.15:
        candle_note = "Recent candle has a small body, similar to a Doji; indecision may be present."
    return {
        "symbol": str(data.attrs.get("symbol", "Selected stock")),
        "last_date": str(data.index[-1].date()),
        "previous_close": round(actual, 2),
        "predicted_close": round(predicted, 2),
        "change_pct": round(pct, 2),
        "direction": direction,
        "mae": round(mae, 2) if mae is not None else None,
        "candle_note": candle_note,
        "rows": int(len(data))
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    symbol = str(payload.get("symbol", "")).strip()
    if not symbol:
        return jsonify(error="Enter a stock symbol, for example RELIANCE.NS or AAPL."), 400
    try:
        data = load_market_data(symbol)
        data.attrs["symbol"] = symbol.upper()
        result = predict_next(data)
        result["disclaimer"] = "Educational estimate only. It is not financial advice or a guaranteed forecast."
        return jsonify(result)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/image")
def api_image():
    from PIL import Image, ImageOps
    import uuid

    image = request.files.get("chart")

    if not image or not image.filename:
        return jsonify(error="Please select a candlestick chart image."), 400

    if not allowed_file(image.filename):
        return jsonify(
            error="Only PNG, JPG, JPEG and WEBP images are allowed."
        ), 400

    try:
        # Create dedicated candles directory
        upload_folder = os.path.join("static", "candles")
        os.makedirs(upload_folder, exist_ok=True)

        # Generate unique filename
        original_name = secure_filename(image.filename)
        extension = original_name.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{extension}"

        path = os.path.join(upload_folder, filename)

        # Open and normalize image
        img = Image.open(image.stream)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        # Resize large images
        img.thumbnail((1200, 1200))

        # Save image
        img.save(path, format="JPEG", quality=95)

        # Convert image into NumPy array
        arr = np.asarray(img, dtype=np.uint8)

        # Validate image dimensions
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Image must have three RGB channels.")

        # Extract RGB channels
        red_channel = arr[:, :, 0]
        green_channel = arr[:, :, 1]
        blue_channel = arr[:, :, 2]

        # Detect green candles
        green_mask = (
            (green_channel > 90) &
            (green_channel > red_channel * 1.18) &
            (green_channel > blue_channel * 1.05)
        )

        # Detect red candles
        red_mask = (
            (red_channel > 100) &
            (red_channel > green_channel * 1.18) &
            (red_channel > blue_channel * 1.10)
        )

        green_count = int(np.count_nonzero(green_mask))
        red_count = int(np.count_nonzero(red_mask))

        total = green_count + red_count

        if total == 0:
            bias = "Inconclusive"
            message = "No clear red or green chart colors detected."

        else:
            green_share = green_count / total

            if green_share > 0.58:
                bias = "Bullish"
                message = (
                    "Green pixels are more common. "
                    "This suggests a possible bullish visual bias."
                )

            elif green_share < 0.42:
                bias = "Bearish"
                message = (
                    "Red pixels are more common. "
                    "This suggests a possible bearish visual bias."
                )

            else:
                bias = "Neutral"
                message = (
                    "Red and green pixels are relatively balanced. "
                    "The visual direction is inconclusive."
                )

        return jsonify(
            success=True,
            filename=filename,
            image_url="/" + path.replace("\\", "/"),
            green_pixels=green_count,
            red_pixels=red_count,
            visual_bias=bias,
            explanation=message,
            limitation=(
                "This is a basic color analysis, not a validated "
                "candlestick pattern recognition model."
            )
        )

    except Exception as exc:
        traceback.print_exc()

        return jsonify(
            success=False,
            error="Could not process image: " + str(exc)
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
s