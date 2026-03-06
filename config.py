# -*- coding: utf-8 -*-
"""
config.py — Options Scalper Bot v5.0
Central configuration for all modules.
"""
import os

# ─── Telegram ────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8481177703:AAFWVUUGRQ811z41HkjjH5M_kE5kVbpQBu4")
TELEGRAM_CHAT_IDS  = ["992860154", "777385885"]

# ─── Alpaca API ───────────────────────────────────────────────────────────────
ALPACA_KEY_ID     = os.environ.get("ALPACA_KEY_ID",     "AK3W7IMWGZNLFSVDTK6JAZFL4Y")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "4rJ6apKPqpLMnfDE9mmM59T7quvJ7rscHJ7AfPM4MDZw")
ALPACA_BASE_URL   = "https://data.alpaca.markets"

# ─── Watchlist ────────────────────────────────────────────────────────────────
WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "QQQ", "SPY"]

# ─── Scan Settings ────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 60        # كل دقيقة
COOLDOWN_MINUTES      = 30        # لا إشارتين لنفس السهم خلال 30 دقيقة

# ─── Timeframes ───────────────────────────────────────────────────────────────
SCALP_TIMEFRAME = "1Min"
DAY_TIMEFRAME   = "15Min"

# ─── Scalping Parameters ─────────────────────────────────────────────────────
SCALP_RSI_OVERBOUGHT = 70
SCALP_RSI_OVERSOLD   = 30
SCALP_STOCH_OB       = 80
SCALP_STOCH_OS       = 20
SCALP_VOLUME_MULT    = 1.5
SCALP_TARGET_PCT     = 0.008    # 0.8%
SCALP_STOP_PCT       = 0.004    # 0.4%
SCALP_TIME_MIN       = 1
SCALP_TIME_MAX       = 5

# ─── Day Trading Parameters ──────────────────────────────────────────────────
DAY_RSI_OVERBOUGHT = 65
DAY_RSI_OVERSOLD   = 35
DAY_TARGET_PCT     = 0.02    # 2%
DAY_STOP_PCT       = 0.01    # 1%
DAY_TIME_MIN       = 15
DAY_TIME_MAX       = 60

# ─── Signal Quality ───────────────────────────────────────────────────────────
SIGNAL_MIN_STRENGTH = 4.5    # الحد الأدنى لقوة الإشارة

# ─── Risk Management ─────────────────────────────────────────────────────────
MAX_DAILY_SIGNALS    = 10
MAX_DAILY_LOSSES     = 3
RISK_PER_TRADE_PCT   = 0.02   # 2% من رأس المال لكل صفقة
