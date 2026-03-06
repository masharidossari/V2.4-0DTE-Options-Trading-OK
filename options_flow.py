# -*- coding: utf-8 -*-
"""
options_flow.py — Options Scalper Bot v5.0
Analyzes options flow: PCR, OI, GEX, Max Pain, Unusual Activity.
"""
import logging
import yfinance as yf
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def get_options_snapshot(symbol: str) -> dict:
    """
    Get options flow snapshot for a symbol.
    Returns: pcr, max_pain, gex_bias, uoa, call_oi, put_oi, flow_bias
    """
    snapshot = {
        "pcr":       1.0,
        "max_pain":  None,
        "gex_bias":  "NEUTRAL",
        "uoa":       False,
        "call_oi":   0,
        "put_oi":    0,
        "flow_bias": "NEUTRAL",
        "iv_rank":   50.0,
    }
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return snapshot

        # Use nearest expiry
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        nearest_exp = expirations[0]
        for exp in expirations:
            if exp >= today:
                nearest_exp = exp
                break

        chain = ticker.option_chain(nearest_exp)
        calls = chain.calls
        puts  = chain.puts

        if calls is None or puts is None or calls.empty or puts.empty:
            return snapshot

        # PCR (Put/Call Ratio by OI)
        total_call_oi = int(calls["openInterest"].fillna(0).sum())
        total_put_oi  = int(puts["openInterest"].fillna(0).sum())
        snapshot["call_oi"] = total_call_oi
        snapshot["put_oi"]  = total_put_oi

        if total_call_oi > 0:
            pcr = total_put_oi / total_call_oi
            snapshot["pcr"] = round(pcr, 2)

        # Max Pain
        current_price = float(ticker.fast_info.get("lastPrice") or calls["strike"].median())
        snapshot["max_pain"] = _calculate_max_pain(calls, puts, current_price)

        # GEX Bias (simplified)
        if snapshot["pcr"] > 1.3:
            snapshot["gex_bias"] = "BEARISH"
        elif snapshot["pcr"] < 0.7:
            snapshot["gex_bias"] = "BULLISH"
        else:
            snapshot["gex_bias"] = "NEUTRAL"

        # Unusual Options Activity (volume >> OI)
        call_vol = calls["volume"].fillna(0).sum()
        put_vol  = puts["volume"].fillna(0).sum()
        if call_vol > total_call_oi * 0.5 or put_vol > total_put_oi * 0.5:
            snapshot["uoa"] = True

        # Overall flow bias
        if snapshot["pcr"] < 0.7 and snapshot["gex_bias"] == "BULLISH":
            snapshot["flow_bias"] = "BULLISH"
        elif snapshot["pcr"] > 1.3 and snapshot["gex_bias"] == "BEARISH":
            snapshot["flow_bias"] = "BEARISH"
        else:
            snapshot["flow_bias"] = "NEUTRAL"

    except Exception as e:
        logger.debug(f"get_options_snapshot({symbol}): {e}")

    return snapshot


def _calculate_max_pain(calls, puts, current_price: float) -> float:
    """Calculate max pain strike price."""
    try:
        strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
        if not strikes:
            return current_price

        min_pain = float("inf")
        max_pain_strike = current_price

        for strike in strikes:
            # Pain for call holders
            call_pain = calls[calls["strike"] <= strike]["openInterest"].fillna(0).sum() * 0
            for _, row in calls.iterrows():
                if strike > row["strike"]:
                    call_pain += (strike - row["strike"]) * row.get("openInterest", 0)

            # Pain for put holders
            put_pain = 0
            for _, row in puts.iterrows():
                if strike < row["strike"]:
                    put_pain += (row["strike"] - strike) * row.get("openInterest", 0)

            total_pain = call_pain + put_pain
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = strike

        return float(max_pain_strike)
    except Exception:
        return current_price
