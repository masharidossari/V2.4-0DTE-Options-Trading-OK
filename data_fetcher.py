# -*- coding: utf-8 -*-
"""
data_fetcher.py — Options Scalper Bot v5.0
Fetches OHLCV bars using yfinance (free, real-time delayed).
Falls back gracefully if Alpaca is unavailable.
"""
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Timeframe mapping: internal → yfinance interval
_TF_MAP = {
    "1Min":  "1m",
    "5Min":  "5m",
    "15Min": "15m",
    "1Hour": "60m",
    "1Day":  "1d",
}

def get_bars(symbol: str, timeframe: str = "5Min", limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV bars for a symbol.
    Returns DataFrame with lowercase columns: open, high, low, close, volume
    """
    interval = _TF_MAP.get(timeframe, "5m")
    try:
        period = "1d" if interval in ("1m", "5m") else "5d"
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is None or len(df) < 10:
            logger.warning(f"{symbol}: insufficient data ({len(df) if df is not None else 0} bars)")
            return None
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.dropna()
        return df.tail(limit)
    except Exception as e:
        logger.error(f"get_bars({symbol}, {timeframe}): {e}")
        return None


def get_atm_option(symbol: str, direction: str) -> Optional[dict]:
    """
    Get At-The-Money option info for a symbol.
    Returns dict with: strike, expiry, option_symbol, bid, ask, iv, delta, gamma
    """
    try:
        ticker = yf.Ticker(symbol)
        current_price = ticker.fast_info.get("lastPrice") or ticker.fast_info.get("regularMarketPrice")
        if not current_price:
            return None

        # Get nearest expiry
        expirations = ticker.options
        if not expirations:
            return None

        # Use 0DTE or nearest expiry
        today = datetime.now().strftime("%Y-%m-%d")
        nearest_exp = expirations[0]
        for exp in expirations:
            if exp >= today:
                nearest_exp = exp
                break

        # Get option chain
        chain = ticker.option_chain(nearest_exp)
        options = chain.calls if direction == "CALL" else chain.puts

        if options is None or options.empty:
            return None

        # Find ATM strike
        options = options.copy()
        options["diff"] = abs(options["strike"] - current_price)
        atm = options.nsmallest(1, "diff").iloc[0]

        return {
            "strike":        float(atm["strike"]),
            "expiry":        nearest_exp,
            "option_symbol": atm.get("contractSymbol", f"{symbol}{nearest_exp}{direction[0]}{atm['strike']:.0f}"),
            "bid":           float(atm.get("bid", 0)),
            "ask":           float(atm.get("ask", 0)),
            "iv":            float(atm.get("impliedVolatility", 0)),
            "delta":         float(atm.get("delta", 0.5 if direction == "CALL" else -0.5)),
            "gamma":         float(atm.get("gamma", 0)),
            "volume":        int(atm.get("volume", 0) or 0),
            "openInterest":  int(atm.get("openInterest", 0) or 0),
        }
    except Exception as e:
        logger.debug(f"get_atm_option({symbol}, {direction}): {e}")
        return None
