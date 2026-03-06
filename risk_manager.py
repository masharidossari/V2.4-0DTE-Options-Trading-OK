# -*- coding: utf-8 -*-
"""
risk_manager.py — Options Scalper Bot v5.0
Kelly Criterion + Daily Limits + Time Filter.
"""
import logging
from datetime import datetime, timezone, timedelta
from config import MAX_DAILY_SIGNALS, MAX_DAILY_LOSSES

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.daily_signals  = 0
        self.daily_losses   = 0
        self.daily_wins     = 0
        self.last_reset_day = datetime.now(timezone.utc).date()
        self.signal_history = []  # list of dicts

    def _reset_if_new_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_day:
            self.daily_signals  = 0
            self.daily_losses   = 0
            self.daily_wins     = 0
            self.last_reset_day = today
            logger.info("RiskManager: Daily counters reset.")

    def can_trade(self, signal) -> bool:
        self._reset_if_new_day()

        # Max daily signals
        if self.daily_signals >= MAX_DAILY_SIGNALS:
            logger.info(f"RiskManager: Max daily signals ({MAX_DAILY_SIGNALS}) reached.")
            return False

        # Max daily losses
        if self.daily_losses >= MAX_DAILY_LOSSES:
            logger.info(f"RiskManager: Max daily losses ({MAX_DAILY_LOSSES}) reached.")
            return False

        # Time filter: avoid first 5 min and last 15 min of session
        try:
            from zoneinfo import ZoneInfo
            now_ny = datetime.now(ZoneInfo("America/New_York"))
        except ImportError:
            import pytz
            now_ny = datetime.now(pytz.timezone("America/New_York"))

        market_open  = now_ny.replace(hour=9,  minute=30, second=0, microsecond=0)
        market_close = now_ny.replace(hour=16, minute=0,  second=0, microsecond=0)
        avoid_open   = market_open  + timedelta(minutes=5)
        avoid_close  = market_close - timedelta(minutes=15)

        if now_ny < avoid_open:
            logger.info("RiskManager: Too early — first 5 min avoided.")
            return False
        if now_ny > avoid_close:
            logger.info("RiskManager: Too late — last 15 min avoided.")
            return False

        return True

    def record_signal(self, signal):
        self._reset_if_new_day()
        self.daily_signals += 1
        self.signal_history.append({
            "symbol":    signal.symbol,
            "direction": signal.direction,
            "time":      datetime.now(timezone.utc).isoformat(),
            "strength":  signal.strength,
        })

    def record_result(self, win: bool):
        self._reset_if_new_day()
        if win:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

    def get_stats(self) -> dict:
        self._reset_if_new_day()
        return {
            "daily_signals": self.daily_signals,
            "daily_wins":    self.daily_wins,
            "daily_losses":  self.daily_losses,
        }
