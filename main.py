# -*- coding: utf-8 -*-
"""
=============================================================
  Options Scalper Bot v5.0 - main.py
  AI-Powered Options Trading Signal System
  Magnificent 7 + QQQ + SPY
  3-Layer AI: XGBoost + LSTM + Reinforcement Learning
=============================================================
"""

import sys
import io
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import logging
import time
from datetime import datetime, timezone
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import (
    WATCHLIST, SCAN_INTERVAL_SECONDS, COOLDOWN_MINUTES,
    SCALP_TIMEFRAME, DAY_TIMEFRAME,
)
from data_fetcher import get_bars, get_atm_option
from indicators import compute_all_indicators
from signal_engine import SignalEngine, Signal
from notifier import send_signal, send_startup_message, send_error_alert
from market_context import get_market_context, get_key_levels, check_signal_alignment
from options_flow import get_options_snapshot
from risk_manager import RiskManager
from market_regime import get_liquidity_levels

# AI Brain
try:
    from ai_brain import initialize_ai, ai_evaluate_signal, format_ai_signal_message
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scalper_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def is_market_open() -> bool:
    try:
        from zoneinfo import ZoneInfo
        now_ny = datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        import pytz
        now_ny = datetime.now(pytz.timezone("America/New_York"))
    if now_ny.weekday() >= 5:
        return False
    market_open  = now_ny.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_ny.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now_ny <= market_close


def time_to_open() -> str:
    try:
        from zoneinfo import ZoneInfo
        now_ny = datetime.now(ZoneInfo("America/New_York"))
    except ImportError:
        import pytz
        now_ny = datetime.now(pytz.timezone("America/New_York"))
    if now_ny.weekday() >= 5:
        days = 7 - now_ny.weekday()
        return f"{days} days"
    market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_ny < market_open:
        delta = market_open - now_ny
        hours, rem = divmod(int(delta.total_seconds()), 3600)
        mins = rem // 60
        return f"{hours}h {mins}m"
    return "after close"


def enrich_signal(signal: Signal, market_ctx: dict) -> Signal:
    try:
        option_type = "call" if signal.direction == "CALL" else "put"
        contract = get_atm_option(signal.symbol, signal.entry_price, option_type)
        if contract:
            signal.option_symbol = contract.get("symbol", "N/A")
            signal.option_strike = float(contract.get("strike_price", 0))
            signal.option_expiry = contract.get("expiration_date", "N/A")
    except Exception as e:
        logger.warning("Option enrichment failed for %s: %s", signal.symbol, e)

    try:
        levels = get_key_levels(signal.symbol, signal.entry_price)
        if levels:
            nr = levels.get("nearest_resistance")
            ns = levels.get("nearest_support")
            if signal.direction == "CALL" and nr:
                dist = levels.get("dist_to_resistance", 0)
                signal.reasons.append(f"Resistance at ${nr} ({dist}% away)")
            elif signal.direction == "PUT" and ns:
                dist = levels.get("dist_to_support", 0)
                signal.reasons.append(f"Support at ${ns} ({dist}% away)")
    except Exception as e:
        logger.warning("Key levels failed for %s: %s", signal.symbol, e)

    bias = market_ctx.get("market_bias", "NEUTRAL")
    vix  = market_ctx.get("vix_regime", {})
    signal.reasons.append(
        f"Market: {bias} | VIX: {vix.get('level','?')} ({vix.get('value','?')})"
    )
    return signal


def enrich_with_options_flow(signal: Signal) -> tuple:
    flow_boost = 0.0
    flow_data  = None
    try:
        flow_data = get_options_snapshot(signal.symbol, signal.entry_price)
        if flow_data:
            flow_bias = flow_data.get("flow_bias", "NEUTRAL")
            uoa_bias  = flow_data.get("uoa_bias", "NEUTRAL")
            if signal.direction == "CALL":
                if flow_bias == "BULLISH":
                    flow_boost += 1.0
                    signal.reasons.append(f"Options flow BULLISH (PCR={flow_data['pcr']:.2f})")
                elif flow_bias == "BEARISH":
                    flow_boost -= 1.0
                    signal.reasons.append(f"Options flow BEARISH - caution (PCR={flow_data['pcr']:.2f})")
                if uoa_bias == "CALL":
                    flow_boost += 0.5
                    signal.reasons.append(f"Unusual CALL activity ({flow_data['uoa_calls']} contracts)")
            elif signal.direction == "PUT":
                if flow_bias == "BEARISH":
                    flow_boost += 1.0
                    signal.reasons.append(f"Options flow BEARISH (PCR={flow_data['pcr']:.2f})")
                elif flow_bias == "BULLISH":
                    flow_boost -= 1.0
                    signal.reasons.append(f"Options flow BULLISH - caution (PCR={flow_data['pcr']:.2f})")
                if uoa_bias == "PUT":
                    flow_boost += 0.5
                    signal.reasons.append(f"Unusual PUT activity ({flow_data['uoa_puts']} contracts)")
            mp = flow_data.get("max_pain")
            if mp:
                signal.reasons.append(f"Max Pain: ${mp}")
    except Exception as e:
        logger.warning("Options flow enrichment failed for %s: %s", signal.symbol, e)
    return signal, flow_data, flow_boost


def build_ai_features(signal: Signal, market_ctx: dict,
                       flow_data: dict, df_latest) -> dict:
    """Build feature dict for AI Brain evaluation."""
    row = {}
    if df_latest is not None and len(df_latest) > 0:
        row = df_latest.iloc[-1].to_dict()

    sym_flow = flow_data or {}
    return {
        "technical": {
            "rsi":          row.get("rsi", 50),
            "macd":         row.get("macd", 0),
            "macd_signal":  row.get("macd_signal", 0),
            "bb_position":  row.get("bb_position", 0.5),
            "vwap_dist":    row.get("vwap_dist", 0),
            "ema9":         row.get("ema9", 0),
            "ema21":        row.get("ema21", 0),
            "ema50":        row.get("ema50", 0),
            "ema200":       row.get("ema200", 0),
            "volume_ratio": row.get("volume_ratio", 1),
            "stoch_k":      row.get("stoch_k", 50),
            "stoch_d":      row.get("stoch_d", 50),
            "atr":          row.get("atr", 1),
            "adx":          row.get("adx", 25),
        },
        "market_context": market_ctx,
        "options_flow": {
            "put_call_ratio": sym_flow.get("pcr", 1.0),
            "oi_change":      sym_flow.get("oi_change", 0),
            "gamma_exposure": sym_flow.get("gamma_exposure", 0),
            "unusual_flow":   sym_flow.get("uoa_bias", "NEUTRAL") != "NEUTRAL",
        },
        "time": {
            "hour_of_day":       datetime.now().hour,
            "day_of_week":       datetime.now().weekday(),
            "minutes_to_close":  max(0, (16 - datetime.now().hour) * 60 - datetime.now().minute),
        },
    }


def run_scanner():
    engine   = SignalEngine()
    risk_mgr = RiskManager()
    last_signal_time = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))

    market_ctx        = {}
    last_ctx_refresh  = 0
    CTX_REFRESH_SECS  = 300

    # v5.0: بيانات السيولة اليومية (تُحدَّث مرة عند الفتح)
    daily_liquidity   = {}   # symbol -> dict من get_liquidity_levels
    last_liq_refresh  = 0
    LIQ_REFRESH_SECS  = 1800  # كل 30 دقيقة

    # Price sequences for LSTM (per symbol)
    price_sequences = defaultdict(list)
    recent_headlines = []
    last_headlines_refresh = 0
    HEADLINES_REFRESH_SECS = 600

    # Initialize AI
    ai_ready = False
    if AI_AVAILABLE:
        logger.info("Initializing AI Brain (3 layers)...")
        ai_ready = initialize_ai()
        logger.info("AI Brain: %s", "READY" if ai_ready else "FALLBACK MODE")

    logger.info("=" * 60)
    logger.info("  Options Scalper Bot v5.0 - STARTED")
    logger.info("  AI: %s", "3-Layer Active" if ai_ready else "Disabled")
    logger.info("  Watching: %s", ", ".join(WATCHLIST))
    logger.info("  Scan interval: %ds", SCAN_INTERVAL_SECONDS)
    logger.info("=" * 60)

    send_startup_message(ai_ready=ai_ready)

    while True:
        try:
            if not is_market_open():
                logger.info("Market closed. Opens in: %s", time_to_open())
                time.sleep(300)
                continue

            now_ts = time.time()

            # Refresh market context every 5 min
            if now_ts - last_ctx_refresh > CTX_REFRESH_SECS:
                logger.info("Refreshing market context...")
                try:
                    market_ctx = get_market_context()
                    last_ctx_refresh = now_ts
                    bias = market_ctx.get("market_bias", "UNKNOWN")
                    vix  = market_ctx.get("vix_regime", {})
                    logger.info("Market bias: %s | VIX: %s (%s)",
                                bias, vix.get("level","?"), vix.get("value","?"))
                    if bias == "AVOID":
                        logger.warning("EXTREME VOLATILITY - pausing 10 min")
                        time.sleep(600)
                        continue
                except Exception as e:
                    logger.error("Market context refresh failed: %s", e)
                    market_ctx = {"market_bias": "NEUTRAL"}

            # Refresh headlines every 10 min
            if ai_ready and now_ts - last_headlines_refresh > HEADLINES_REFRESH_SECS:
                try:
                    from data_fetcher import get_news_headlines
                    recent_headlines = get_news_headlines(WATCHLIST)
                    last_headlines_refresh = now_ts
                    logger.info("Headlines refreshed: %d items", len(recent_headlines))
                except Exception as e:
                    logger.warning("Headlines fetch failed: %s", e)

            # v5.0: تحديث مستويات السيولة اليومية
            if now_ts - last_liq_refresh > LIQ_REFRESH_SECS:
                logger.info("Refreshing daily liquidity levels...")
                for sym in WATCHLIST:
                    try:
                        df_d = get_bars(sym, timeframe="1Day", limit=15)
                        if df_d is not None and len(df_d) >= 5:
                            current_price = df_d.iloc[-1]["close"]
                            daily_liquidity[sym] = get_liquidity_levels(df_d, current_price)
                            logger.info("%s liquidity: PDH=%.2f PDL=%.2f",
                                        sym,
                                        daily_liquidity[sym].get("PDH", 0),
                                        daily_liquidity[sym].get("PDL", 0))
                    except Exception as e:
                        logger.warning("Liquidity refresh failed for %s: %s", sym, e)
                last_liq_refresh = now_ts

            logger.info("Scanning — %s | Bias: %s",
                        datetime.now().strftime("%H:%M:%S"),
                        market_ctx.get("market_bias","?"))

            for symbol in WATCHLIST:
                try:
                    now_utc  = datetime.now(timezone.utc)
                    elapsed  = (now_utc - last_signal_time[symbol]).total_seconds() / 60
                    if elapsed < COOLDOWN_MINUTES:
                        continue

                    logger.info("Analyzing %s...", symbol)

                    # Scalping (1-5 min)
                    signal   = None
                    df_scalp = None
                    df_scalp = get_bars(symbol, timeframe=SCALP_TIMEFRAME, limit=100)
                    if df_scalp is not None and len(df_scalp) >= 30:
                        df_scalp = compute_all_indicators(df_scalp, mode="scalp")
                        signal   = engine.analyze_scalp(symbol, df_scalp)

                    # Day Trading (15-60 min)
                    df_day = None
                    if signal is None:
                        df_day = get_bars(symbol, timeframe=DAY_TIMEFRAME, limit=200)
                        if df_day is not None and len(df_day) >= 50:
                            df_day = compute_all_indicators(df_day, mode="day")
                            signal = engine.analyze_day(symbol, df_day)

                    # v5.0: إضافة بيانات السيولة اليومية للإشارة
                    if signal is not None and symbol in daily_liquidity:
                        liq = daily_liquidity[symbol]
                        signal.pdh = liq.get("PDH")
                        signal.pdl = liq.get("PDL")
                        signal.poc = liq.get("POC")
                        # تحقق من قرب السعر من مستويات السيولة
                        near_liq = liq.get("near_liquidity_level")
                        if near_liq is not None:
                            signal.near_liquidity = True
                            signal.liquidity_type = liq.get("near_liquidity_type", "")
                            signal.reasons.append(
                                f"💧 قرب مستوى سيولة: {liq.get('near_liquidity_type','')} @ ${near_liq:.2f}"
                            )

                    if signal is None:
                        logger.info("%s: No signal", symbol)
                        time.sleep(0.5)
                        continue

                    # Market context filter
                    aligned, ctx_boost, ctx_reason = check_signal_alignment(
                        signal.direction, market_ctx, symbol
                    )
                    if not aligned:
                        logger.info("%s: Signal blocked by market context (%s)", symbol, ctx_reason)
                        continue
                    signal.strength += ctx_boost
                    signal.reasons.append(ctx_reason)

                    # Risk manager
                    if not risk_mgr.can_trade(signal):
                        logger.info("%s: Risk manager blocked signal", symbol)
                        continue

                    # Enrich signal
                    signal = enrich_signal(signal, market_ctx)
                    signal, flow_data, flow_boost = enrich_with_options_flow(signal)
                    signal.strength = min(10.0, signal.strength + flow_boost)

                    # Update price sequence for LSTM
                    df_for_seq = df_scalp if df_scalp is not None else df_day
                    if df_for_seq is not None and len(df_for_seq) > 0:
                        latest_row = df_for_seq.iloc[-1].to_dict()
                        price_sequences[symbol].append(latest_row)
                        if len(price_sequences[symbol]) > 60:
                            price_sequences[symbol] = price_sequences[symbol][-60:]

                    # AI Evaluation (3 layers)
                    if ai_ready:
                        raw_features = build_ai_features(
                            signal, market_ctx, flow_data, df_for_seq
                        )
                        ai_result = ai_evaluate_signal(
                            signal, raw_features,
                            price_sequences[symbol],
                            recent_headlines,
                        )

                        if ai_result["final_action"] == "SKIP":
                            logger.info(
                                "AI SKIP: %s %s (conf=%.0f%% | L1=%.1f L2=%.1f)",
                                symbol, signal.direction,
                                ai_result["final_confidence"] * 100,
                                ai_result["l1_score"], ai_result["l2_score"],
                            )
                            continue

                        # Use AI-enhanced message
                        msg = format_ai_signal_message(signal, ai_result)
                        signal.strength = ai_result["ai_strength"]
                    else:
                        # Fallback: standard message
                        msg = None
                        ai_result = None

                    # Send signal
                    send_signal(signal, custom_message=msg)
                    risk_mgr.record_signal(signal)
                    last_signal_time[symbol] = datetime.now(timezone.utc)

                    logger.info(
                        "SIGNAL SENT: %s %s %s | Strength=%.1f | AI=%s",
                        symbol, signal.direction, signal.signal_type,
                        signal.strength,
                        ai_result["final_action"] if ai_result else "N/A",
                    )

                except Exception as e:
                    logger.error("Error processing %s: %s", symbol, e, exc_info=False)
                    time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            try:
                from notifier import send_message
                send_message("*Options Scalper Bot v5.0 - STOPPED*")
            except Exception:
                pass
            break
        except Exception as e:
            logger.error("Main loop error: %s", e, exc_info=True)

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_scanner()
