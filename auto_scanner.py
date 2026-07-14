import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import tomllib
from datetime import datetime

WATCHLIST = ["SILVERBEES", "GOLDBEES", "TMPV"]
STATE_FILE = "signal_state.json"

# Load Telegram secrets
with open(".streamlit/secrets.toml", "rb") as f:
    secrets = tomllib.load(f)

TELEGRAM_BOT_TOKEN = secrets["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = str(secrets["TELEGRAM_CHAT_ID"])


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
        timeout=15
    ).raise_for_status()


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def calc_rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def analyze(symbol):
    ticker = symbol.upper()
    if not ticker.endswith(".NS"):
        ticker += ".NS"

    d = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if len(d) < 220:
        raise ValueError("Not enough historical data")

    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA200"] = d["Close"].ewm(span=200, adjust=False).mean()
    d["RSI"] = calc_rsi(d["Close"])

    e12 = d["Close"].ewm(span=12, adjust=False).mean()
    e26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = e12 - e26
    d["MACDS"] = d["MACD"].ewm(span=9, adjust=False).mean()

    d["VOL20"] = d["Volume"].rolling(20).mean()

    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"] - d["Close"].shift()).abs()
    ], axis=1).max(axis=1)

    d["ATR"] = tr.rolling(14).mean()

    up = d["High"].diff()
    down = -d["Low"].diff()

    pdm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=d.index
    )

    mdm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=d.index
    )

    atrs = tr.ewm(alpha=1/14, adjust=False).mean()

    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atrs.replace(0, np.nan)
    mdi = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / atrs.replace(0, np.nan)

    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean()

    d["RES20"] = d["High"].shift(1).rolling(20).max()

    r = d.iloc[-1]

    score = 0
    if r.Close > r.EMA20: score += 20
    if r.EMA20 > r.EMA50: score += 20
    if r.Close > r.EMA200: score += 15
    if 50 <= r.RSI <= 70: score += 15
    if r.MACD > r.MACDS: score += 10
    if r.ADX >= 20: score += 10
    if r.Volume > r.VOL20 * 1.2: score += 5
    if r.Close > r.RES20: score += 5

    if score >= 85:
        signal = "🟢 STRONG BUY"
    elif score >= 70:
        signal = "🟢 BUY"
    elif score >= 55:
        signal = "🟠 WATCH"
    elif score >= 35:
        signal = "🟡 WAIT"
    else:
        signal = "🔴 AVOID"

    price = float(r.Close)
    atr = float(r.ATR)
    stop = max(0.01, price - 1.5 * atr)
    risk = max(price - stop, 0.01)
    target = price + 2 * risk

    return ticker, signal, score, price, stop, target


state = load_state()

print("Swing Analyzer V4 automatic scan started...")
print(datetime.now())

for symbol in WATCHLIST:
    try:
        ticker, signal, score, price, stop, target = analyze(symbol)
        old_signal = state.get(ticker)

        print(f"{ticker}: {signal} | Score {score}/100 | ₹{price:.2f}")

        if old_signal is not None and old_signal != signal:
            message = (
                f"🔔 SWING SIGNAL CHANGE\n\n"
                f"{ticker}\n"
                f"{old_signal} → {signal}\n\n"
                f"Price: ₹{price:.2f}\n"
                f"Score: {score}/100\n"
                f"Stop: ₹{stop:.2f}\n"
                f"2R Target: ₹{target:.2f}"
            )

            send_telegram(message)
            print("Telegram alert sent.")

        state[ticker] = signal

    except Exception as e:
        print(f"{symbol}: ERROR — {e}")

save_state(state)

print("Scan completed.")