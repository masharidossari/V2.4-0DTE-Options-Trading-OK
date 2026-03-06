# -*- coding: utf-8 -*-
"""
market_context.py — Options Scalper Bot v5.0
Analyzes overall market context using SPY, QQQ, VIX.
"""
import logging
import yfinance as yf
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)


def get_market_context() -> dict:
    """
    Fetch SPY, QQQ, VIX and compute overall market bias.
    Returns dict with: bias, vix_level, spy_trend, qqq_trend, vix_value
    """
    ctx = {
        "bias":       "NEUTRAL",
        "vix_level":  "NORMAL",
        "spy_trend":  "NEUTRAL",
        "qqq_trend":  "NEUTRAL",
        "vix_value":  20.0,
        "spy_change": 0.0,
        "qqq_change": 0.0,
    }
    try:
        # VIX
        vix_data = yf.Ticker("^VIX").history(period="2d", interval="5m")
        if vix_data is not None and len(vix_data) > 0:
            vix_val = float(vix_data["Close"].iloc[-1])
            ctx["vix_value"] = vix_val
            if vix_val < 15:
                ctx["vix_level"] = "LOW"
            elif vix_val < 20:
                ctx["vix_level"] = "NORMAL"
            elif vix_val < 25:
                ctx["vix_level"] = "ELEVATED"
            elif vix_val < 35:
                ctx["vix_level"] = "HIGH"
            else:
                ctx["vix_level"] = "EXTREME"

        # SPY
        spy_data = yf.Ticker("SPY").history(period="2d", interval="5m")
        if spy_data is not None and len(spy_data) > 5:
            spy_close  = spy_data["Close"]
            spy_change = float((spy_close.iloc[-1] - spy_close.iloc[-6]) / spy_close.iloc[-6] * 100)
            ctx["spy_change"] = spy_change
            ctx["spy_trend"]  = "BULLISH" if spy_change > 0.1 else ("BEARISH" if spy_change < -0.1 else "NEUTRAL")

        # QQQ
        qqq_data = yf.Ticker("QQQ").history(period="2d", interval="5m")
        if qqq_data is not None and len(qqq_data) > 5:
            qqq_close  = qqq_data["Close"]
            qqq_change = float((qqq_close.iloc[-1] - qqq_close.iloc[-6]) / qqq_close.iloc[-6] * 100)
            ctx["qqq_change"] = qqq_change
            ctx["qqq_trend"]  = "BULLISH" if qqq_change > 0.1 else ("BEARISH" if qqq_change < -0.1 else "NEUTRAL")

        # Overall bias
        bull_count = sum(1 for t in [ctx["spy_trend"], ctx["qqq_trend"]] if t == "BULLISH")
        bear_count = sum(1 for t in [ctx["spy_trend"], ctx["qqq_trend"]] if t == "BEARISH")

        if ctx["vix_level"] in ("HIGH", "EXTREME"):
            ctx["bias"] = "AVOID"
        elif bull_count == 2:
            ctx["bias"] = "BULLISH" if ctx["vix_level"] != "ELEVATED" else "SLIGHT_BULLISH"
        elif bear_count == 2:
            ctx["bias"] = "BEARISH" if ctx["vix_level"] != "ELEVATED" else "SLIGHT_BEARISH"
        elif bull_count > bear_count:
            ctx["bias"] = "SLIGHT_BULLISH"
        elif bear_count > bull_count:
            ctx["bias"] = "SLIGHT_BEARISH"
        else:
            ctx["bias"] = "NEUTRAL"

    except Exception as e:
        logger.warning(f"get_market_context error: {e}")

    return ctx


def get_key_levels(symbol: str) -> dict:
    """
    Get key price levels: previous day high/low, weekly high/low, round numbers.
    """
    levels = {"pdh": None, "pdl": None, "wh": None, "wl": None, "round_numbers": []}
    try:
        ticker  = yf.Ticker(symbol)
        df_day  = ticker.history(period="10d", interval="1d")
        if df_day is not None and len(df_day) >= 2:
            levels["pdh"] = float(df_day["High"].iloc[-2])
            levels["pdl"] = float(df_day["Low"].iloc[-2])
        if df_day is not None and len(df_day) >= 5:
            levels["wh"] = float(df_day["High"].iloc[-5:].max())
            levels["wl"] = float(df_day["Low"].iloc[-5:].min())

        # Round numbers near current price
        current = float(ticker.fast_info.get("lastPrice") or df_day["Close"].iloc[-1])
        step = 5 if current > 100 else 1
        base = round(current / step) * step
        levels["round_numbers"] = [base - step, base, base + step]
    except Exception as e:
        logger.debug(f"get_key_levels({symbol}): {e}")
    return levels


def check_signal_alignment(direction: str, market_ctx: dict, symbol: str) -> Tuple[bool, float, str]:
    """
    Check if signal aligns with overall market context.
    Returns: (aligned: bool, strength_boost: float, reason: str)
    """
    bias = market_ctx.get("bias", "NEUTRAL")

    if bias == "AVOID":
        return False, 0.0, "🚫 VIX مرتفع جداً — تجنب التداول"

    if direction == "CALL":
        if bias in ("BULLISH",):
            return True, 0.5, "✅ السوق صاعد — يدعم CALL"
        elif bias == "SLIGHT_BULLISH":
            return True, 0.2, "📈 السوق صاعد قليلاً — يدعم CALL"
        elif bias == "NEUTRAL":
            return True, 0.0, "➡️ السوق محايد"
        elif bias == "SLIGHT_BEARISH":
            return True, -0.3, "⚠️ السوق هابط قليلاً — ضعف CALL"
        else:  # BEARISH
            return False, 0.0, "🐻 السوق هابط — لا CALL"

    else:  # PUT
        if bias in ("BEARISH",):
            return True, 0.5, "✅ السوق هابط — يدعم PUT"
        elif bias == "SLIGHT_BEARISH":
            return True, 0.2, "📉 السوق هابط قليلاً — يدعم PUT"
        elif bias == "NEUTRAL":
            return True, 0.0, "➡️ السوق محايد"
        elif bias == "SLIGHT_BULLISH":
            return True, -0.3, "⚠️ السوق صاعد قليلاً — ضعف PUT"
        else:  # BULLISH
            return False, 0.0, "🐂 السوق صاعد — لا PUT"
