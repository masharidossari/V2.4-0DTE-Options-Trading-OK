# -*- coding: utf-8 -*-
"""
indicators.py — Options Scalper Bot v5.0
Computes all technical indicators needed by signal_engine.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the DataFrame.
    Input columns: open, high, low, close, volume
    """
    if df is None or len(df) < 20:
        return df

    df = df.copy()
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── EMAs ──────────────────────────────────────────────────────────────────
    df["ema_fast"] = close.ewm(span=9,   adjust=False).mean()
    df["ema_slow"] = close.ewm(span=21,  adjust=False).mean()
    df["ema50"]    = close.ewm(span=50,  adjust=False).mean()
    df["ema200"]   = close.ewm(span=200, adjust=False).mean()

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["rsi"] = _rsi(close, 14)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_mid"]   = bb_mid
    df["bb_width"] = (bb_std * 2 / bb_mid * 100)

    # ── Stochastic ────────────────────────────────────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k_raw    = 100 * (close - low14) / (high14 - low14 + 1e-10)
    df["stoch_k"]  = stoch_k_raw.rolling(3).mean()
    df["stoch_d"]  = df["stoch_k"].rolling(3).mean()

    # ── ATR ───────────────────────────────────────────────────────────────────
    df["atr"] = _atr(df, 14)

    # ── VWAP (intraday approximation) ─────────────────────────────────────────
    typical = (high + low + close) / 3
    df["vwap"] = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)

    # ── Volume Ratio ──────────────────────────────────────────────────────────
    vol_avg        = volume.rolling(20).mean()
    df["vol_ratio"] = volume / vol_avg.replace(0, np.nan)

    # ── ADX ───────────────────────────────────────────────────────────────────
    df["adx"] = _adx(df, 14)

    return df


def check_rsi_divergence(df: pd.DataFrame, lookback: int = 5) -> str:
    """
    Detect RSI divergence (bullish / bearish / none).
    """
    if df is None or len(df) < lookback + 2:
        return "none"
    if "rsi" not in df.columns:
        df = compute_all_indicators(df)
    if "rsi" not in df.columns:
        return "none"

    close = df["close"].values
    rsi   = df["rsi"].values

    # Look at last `lookback` bars
    price_slice = close[-lookback:]
    rsi_slice   = rsi[-lookback:]

    price_min_idx = int(np.argmin(price_slice))
    price_max_idx = int(np.argmax(price_slice))
    rsi_min_idx   = int(np.argmin(rsi_slice))
    rsi_max_idx   = int(np.argmax(rsi_slice))

    # Bullish divergence: price makes lower low, RSI makes higher low
    if price_min_idx > rsi_min_idx and price_slice[-1] < price_slice[0] and rsi_slice[-1] > rsi_slice[0]:
        return "bullish"

    # Bearish divergence: price makes higher high, RSI makes lower high
    if price_max_idx > rsi_max_idx and price_slice[-1] > price_slice[0] and rsi_slice[-1] < rsi_slice[0]:
        return "bearish"

    return "none"


# ─── Private helpers ──────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr14     = tr.rolling(period).mean()
    plus_di   = 100 * plus_dm.rolling(period).mean()  / atr14.replace(0, np.nan)
    minus_di  = 100 * minus_dm.rolling(period).mean() / atr14.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()
