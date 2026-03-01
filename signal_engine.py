"""
=============================================================
  Options Scalper Bot — signal_engine.py
      (Call / Put)
    :
    1.   (Clean Entry)
    2.   (Reversal)
    3.   (Breakout)
=========================================================="""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

import pandas as pd

from config import (
    SCALP_RSI_OVERBOUGHT, SCALP_RSI_OVERSOLD,
    SCALP_STOCH_OB, SCALP_STOCH_OS, SCALP_VOLUME_MULT,
    SCALP_TARGET_PCT, SCALP_STOP_PCT, SCALP_TIME_MIN, SCALP_TIME_MAX,
    DAY_RSI_OVERBOUGHT, DAY_RSI_OVERSOLD,
    DAY_TARGET_PCT, DAY_STOP_PCT, DAY_TIME_MIN, DAY_TIME_MAX,
    SIGNAL_MIN_STRENGTH,
)
from indicators import check_rsi_divergence
from market_regime import (
    detect_market_regime, get_liquidity_levels, detect_order_blocks,
    detect_fair_value_gaps, calculate_volume_profile, calculate_fibonacci_levels,
    detect_divergence, score_signal, calculate_tiered_targets
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 
# ─────────────────────────────────────────────

@dataclass
class Signal:
    symbol:        str
    direction:     str        # "CALL" or "PUT"
    signal_type:   str        # "Clean Entry" / "Reversal" / "Breakout"
    mode:          str        # "Scalping" or "Day Trading"
    entry_price:   float
    target_price:  float
    stop_price:    float
    target_pct:    float
    stop_pct:      float
    time_min:      int
    time_max:      int
    strength:      float      # 0-10
    reasons:       list = field(default_factory=list)
    option_symbol: Optional[str] = None
    option_strike: Optional[float] = None
    option_expiry: Optional[str] = None
    # ── حقول v4.0 الجديدة ──────────────────────
    grade:          str = "C"          # A+ / A / B / C
    accuracy_pct:   int = 52           # نسبة الدقة المتوقعة
    regime:         str = "UNKNOWN"    # TRENDING / SIDEWAYS / VOLATILE
    adx:            float = 0.0        # قيمة ADX
    tp1_price:      Optional[float] = None
    tp2_price:      Optional[float] = None
    tp3_price:      Optional[float] = None
    tp1_pct:        float = 15.0       # +15% على العقد
    tp2_pct:        float = 50.0       # +50% على العقد
    tp3_pct:        float = 100.0      # +100% على العقد
    sl_contract_pct: float = 30.0      # -30% على العقد
    near_liquidity: bool = False
    liquidity_type: str = ""
    order_block:    bool = False
    fvg_signal:     bool = False
    near_fib_025:   bool = False
    divergence_type: str = "none"
    pdh:            Optional[float] = None
    pdl:            Optional[float] = None
    poc:            Optional[float] = None
    scoring_details: list = field(default_factory=list)


# ─────────────────────────────────────────────
# 
# ─────────────────────────────────────────────

class SignalEngine:

    def __init__(self):
        pass

    # ──────────────────────────────────────────
 # Scalping Engine (1-5 )
    # ──────────────────────────────────────────

    def analyze_scalp(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """
            (Scalping).
                  1-5 .
        """
        if df is None or len(df) < 30:
            return None

        last = df.iloc[-1]

        close      = last["close"]
        ema_fast   = last.get("ema_fast", None)
        ema_slow   = last.get("ema_slow", None)
        rsi_val    = last.get("rsi", 50)
        stoch_k    = last.get("stoch_k", 50)
        stoch_d    = last.get("stoch_d", 50)
        vwap_val   = last.get("vwap", close)
        vol_ratio  = last.get("vol_ratio", 1.0)
        bb_upper   = last.get("bb_upper", close * 1.02)
        bb_lower   = last.get("bb_lower", close * 0.98)
        atr_val    = last.get("atr", close * 0.005)

        if any(v is None for v in [ema_fast, ema_slow]):
            return None

        signals_call = []
        signals_put  = []

 # ── 1. (Clean Entry) ──────────
 # CALL: EMA + VWAP + RSI 45-65
        if (close > ema_fast > ema_slow and
                close > vwap_val and
                45 < rsi_val < 65 and
                stoch_k > stoch_d):
            signals_call.append(("Clean Entry", 3.5,
                                  "  EMA  VWAP RSI    "))

 # PUT: EMA + VWAP + RSI 35-55
        if (close < ema_fast < ema_slow and
                close < vwap_val and
                35 < rsi_val < 55 and
                stoch_k < stoch_d):
            signals_put.append(("Clean Entry", 3.5,
                                  "  EMA  VWAP RSI    "))

 # ── 2. (Reversal) ───────────
        divergence = check_rsi_divergence(df, lookback=5)

 # CALL Reversal: RSI + + 
        if (rsi_val < SCALP_RSI_OVERSOLD and
                stoch_k < SCALP_STOCH_OS and
                stoch_k > stoch_d):
            score = 3.0
            if divergence == "bullish":
                score += 1.5
            signals_call.append(("Reversal", score,
                                  f"RSI={rsi_val:.1f}   ={stoch_k:.1f} ={divergence}"))

 # PUT Reversal: RSI + + 
        if (rsi_val > SCALP_RSI_OVERBOUGHT and
                stoch_k > SCALP_STOCH_OB and
                stoch_k < stoch_d):
            score = 3.0
            if divergence == "bearish":
                score += 1.5
            signals_put.append(("Reversal", score,
                                  f"RSI={rsi_val:.1f}   ={stoch_k:.1f} ={divergence}"))

 # ── 3. (Breakout) ────────────
 # CALL Breakout: + 
        if (close > bb_upper and
                vol_ratio > SCALP_VOLUME_MULT and
                close > vwap_val):
            signals_call.append(("Breakout", 4.0,
                                  f" BB ={bb_upper:.2f}  ={vol_ratio:.1f}x"))

 # PUT Breakout: + 
        if (close < bb_lower and
                vol_ratio > SCALP_VOLUME_MULT and
                close < vwap_val):
            signals_put.append(("Breakout", 4.0,
                                  f" BB ={bb_lower:.2f}  ={vol_ratio:.1f}x"))

 # ── ───────────────────
        best_signal = self._pick_best(signals_call, signals_put)
        if best_signal is None:
            return None

        direction, sig_type, base_score, reason = best_signal

 # ── ──────────
        bonus = 0.0
        bonus_reasons = []

        if vol_ratio > 2.0:
            bonus += 0.5
            bonus_reasons.append(f"   ({vol_ratio:.1f}x)")

        if direction == "CALL" and close > vwap_val:
            bonus += 0.3
            bonus_reasons.append("  VWAP")
        elif direction == "PUT" and close < vwap_val:
            bonus += 0.3
            bonus_reasons.append("  VWAP")

 # ATR
        atr_multiplier = 1.5
        target_pct = min(atr_val * atr_multiplier / close, SCALP_TARGET_PCT * 2)
        target_pct = max(target_pct, SCALP_TARGET_PCT * 0.5)

        final_strength = min(base_score + bonus, 10.0)

        if final_strength < SIGNAL_MIN_STRENGTH:
            return None

        entry  = close
        if direction == "CALL":
            target = round(entry * (1 + target_pct), 2)
            stop   = round(entry * (1 - SCALP_STOP_PCT), 2)
        else:
            target = round(entry * (1 - target_pct), 2)
            stop   = round(entry * (1 + SCALP_STOP_PCT), 2)

        # ── v4.0: تحليل السيولة والتصنيف ──────────────────
        regime_data  = {}
        liq_data     = {}
        ob_data      = {}
        fvg_data     = {}
        fib_data     = {}
        div_data     = {}
        scoring      = {"grade": "C", "accuracy_pct": 52, "reasons": []}
        tiered       = {}

        try:
            regime_data = detect_market_regime(df)
            # إذا كان السوق متذبذباً، نخفض قوة الإشارة
            if regime_data.get("regime") == "SIDEWAYS":
                final_strength = max(final_strength - 1.5, 0)
                bonus_reasons.append("⚠️ السوق متذبذب — إشارة بحذر")
            elif regime_data.get("regime") == "VOLATILE":
                final_strength = max(final_strength - 2.0, 0)
                bonus_reasons.append("⚠️ تذبذب عالٍ — خطر")

            ob_data  = detect_order_blocks(df)
            fvg_data = detect_fair_value_gaps(df)
            fib_data = calculate_fibonacci_levels(df)
            div_data = detect_divergence(df)

            # نمرر بيانات السيولة فارغة هنا (تُحسب في main.py من البيانات اليومية)
            scoring = score_signal(
                direction    = direction,
                regime       = regime_data,
                liquidity    = liq_data,
                order_blocks = ob_data,
                fvg          = fvg_data,
                fib          = fib_data,
                divergence   = div_data,
                current_price = close,
            )
            tiered = calculate_tiered_targets(direction, close, liq_data, fib_data, atr_val)
        except Exception as e:
            logger.warning(f"v4.0 analysis error for {symbol}: {e}")

        return Signal(
            symbol       = symbol,
            direction    = direction,
            signal_type  = sig_type,
            mode         = "Scalping",
            entry_price  = round(entry, 2),
            target_price = target,
            stop_price   = stop,
            target_pct   = round(target_pct * 100, 2),
            stop_pct     = round(SCALP_STOP_PCT * 100, 2),
            time_min     = SCALP_TIME_MIN,
            time_max     = SCALP_TIME_MAX,
            strength     = round(final_strength, 1),
            reasons      = [reason] + bonus_reasons + scoring.get("reasons", []),
            # حقول v4.0
            grade          = scoring.get("grade", "C"),
            accuracy_pct   = scoring.get("accuracy_pct", 52),
            regime         = regime_data.get("regime", "UNKNOWN"),
            adx            = regime_data.get("adx", 0.0),
            tp1_price      = tiered.get("tp1_price"),
            tp2_price      = tiered.get("tp2_price"),
            tp3_price      = tiered.get("tp3_price"),
            tp1_pct        = tiered.get("tp1_contract_pct", 15.0),
            tp2_pct        = tiered.get("tp2_contract_pct", 50.0),
            tp3_pct        = tiered.get("tp3_contract_pct", 100.0),
            sl_contract_pct = tiered.get("sl_contract_pct", 30.0),
            near_liquidity = liq_data.get("near_liquidity_level") is not None,
            liquidity_type = liq_data.get("near_liquidity_type", ""),
            order_block    = ob_data.get("price_in_bullish_ob") or ob_data.get("price_in_bearish_ob"),
            fvg_signal     = fvg_data.get("price_in_fvg", False),
            near_fib_025   = fib_data.get("near_golden_025", False),
            divergence_type = div_data.get("divergence_type", "none"),
            scoring_details = scoring.get("reasons", []),
        )

    # ──────────────────────────────────────────
 # Day Trading Engine (15-60 )
    # ──────────────────────────────────────────

    def analyze_day(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """
            (Day Trading).
             15  - .
        """
        if df is None or len(df) < 50:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close       = last["close"]
        ema_fast    = last.get("ema_fast", None)
        ema_slow    = last.get("ema_slow", None)
        ema_trend   = last.get("ema_trend", None)
        rsi_val     = last.get("rsi", 50)
        macd_hist   = last.get("macd_hist", 0)
        prev_hist   = prev.get("macd_hist", 0)
        vwap_val    = last.get("vwap", close)
        vol_ratio   = last.get("vol_ratio", 1.0)
        bb_upper    = last.get("bb_upper", close * 1.02)
        bb_lower    = last.get("bb_lower", close * 0.98)
        atr_val     = last.get("atr", close * 0.01)

        if any(v is None for v in [ema_fast, ema_slow]):
            return None

        signals_call = []
        signals_put  = []

        trend_up   = (ema_trend is None) or (close > ema_trend)
        trend_down = (ema_trend is None) or (close < ema_trend)

 # ── 1. (Clean Entry) ──────────
        if (trend_up and
                ema_fast > ema_slow and
                macd_hist > 0 and
                45 < rsi_val < 70 and
                close > vwap_val):
            score = 4.0
            if macd_hist > prev_hist:
                score += 0.5
            signals_call.append(("Clean Entry", score,
                                  f"  EMA   MACD={macd_hist:.3f}"))

        if (trend_down and
                ema_fast < ema_slow and
                macd_hist < 0 and
                30 < rsi_val < 55 and
                close < vwap_val):
            score = 4.0
            if macd_hist < prev_hist:
                score += 0.5
            signals_put.append(("Clean Entry", score,
                                  f"  EMA   MACD={macd_hist:.3f}"))

 # ── 2. (Reversal) ───────────
        divergence = check_rsi_divergence(df, lookback=8)

        if (rsi_val < DAY_RSI_OVERSOLD and
                macd_hist > prev_hist and
                close > prev["close"]):
            score = 3.5
            if divergence == "bullish":
                score += 1.5
            signals_call.append(("Reversal", score,
                                  f"RSI={rsi_val:.1f}   MACD  ={divergence}"))

        if (rsi_val > DAY_RSI_OVERBOUGHT and
                macd_hist < prev_hist and
                close < prev["close"]):
            score = 3.5
            if divergence == "bearish":
                score += 1.5
            signals_put.append(("Reversal", score,
                                  f"RSI={rsi_val:.1f}   MACD  ={divergence}"))

 # ── 3. (Breakout) ────────────
        if (close > bb_upper and
                vol_ratio > 1.8 and
                trend_up and
                macd_hist > 0):
            signals_call.append(("Breakout", 4.5,
                                  f" BB  ={vol_ratio:.1f}x  "))

        if (close < bb_lower and
                vol_ratio > 1.8 and
                trend_down and
                macd_hist < 0):
            signals_put.append(("Breakout", 4.5,
                                  f" BB  ={vol_ratio:.1f}x  "))

 # ── ───────────────────
        best_signal = self._pick_best(signals_call, signals_put)
        if best_signal is None:
            return None

        direction, sig_type, base_score, reason = best_signal

 # ── ────────────────────────
        bonus = 0.0
        bonus_reasons = []

        if vol_ratio > 2.5:
            bonus += 0.5
            bonus_reasons.append(f"   ({vol_ratio:.1f}x)")

        if ema_trend is not None:
            if direction == "CALL" and close > ema_trend:
                bonus += 0.5
                bonus_reasons.append("  EMA 200")
            elif direction == "PUT" and close < ema_trend:
                bonus += 0.5
                bonus_reasons.append("  EMA 200")

 # ATR
        atr_multiplier = 2.0
        target_pct = min(atr_val * atr_multiplier / close, DAY_TARGET_PCT * 2)
        target_pct = max(target_pct, DAY_TARGET_PCT * 0.5)

        final_strength = min(base_score + bonus, 10.0)

        if final_strength < SIGNAL_MIN_STRENGTH:
            return None

        entry = close
        if direction == "CALL":
            target = round(entry * (1 + target_pct), 2)
            stop   = round(entry * (1 - DAY_STOP_PCT), 2)
        else:
            target = round(entry * (1 - target_pct), 2)
            stop   = round(entry * (1 + DAY_STOP_PCT), 2)

        # ── v4.0: تحليل السيولة والتصنيف ──────────────────
        regime_data  = {}
        liq_data     = {}
        ob_data      = {}
        fvg_data     = {}
        fib_data     = {}
        div_data     = {}
        scoring      = {"grade": "C", "accuracy_pct": 52, "reasons": []}
        tiered       = {}

        try:
            regime_data = detect_market_regime(df)
            if regime_data.get("regime") == "SIDEWAYS":
                final_strength = max(final_strength - 1.0, 0)
                bonus_reasons.append("⚠️ السوق متذبذب")
            elif regime_data.get("regime") == "VOLATILE":
                final_strength = max(final_strength - 1.5, 0)
                bonus_reasons.append("⚠️ تذبذب عالٍ")

            ob_data  = detect_order_blocks(df)
            fvg_data = detect_fair_value_gaps(df)
            fib_data = calculate_fibonacci_levels(df)
            div_data = detect_divergence(df)

            scoring = score_signal(
                direction     = direction,
                regime        = regime_data,
                liquidity     = liq_data,
                order_blocks  = ob_data,
                fvg           = fvg_data,
                fib           = fib_data,
                divergence    = div_data,
                current_price = close,
            )
            tiered = calculate_tiered_targets(direction, close, liq_data, fib_data, atr_val)
        except Exception as e:
            logger.warning(f"v4.0 day analysis error for {symbol}: {e}")

        return Signal(
            symbol       = symbol,
            direction    = direction,
            signal_type  = sig_type,
            mode         = "Day Trading",
            entry_price  = round(entry, 2),
            target_price = target,
            stop_price   = stop,
            target_pct   = round(target_pct * 100, 2),
            stop_pct     = round(DAY_STOP_PCT * 100, 2),
            time_min     = DAY_TIME_MIN,
            time_max     = DAY_TIME_MAX,
            strength     = round(final_strength, 1),
            reasons      = [reason] + bonus_reasons + scoring.get("reasons", []),
            # حقول v4.0
            grade          = scoring.get("grade", "C"),
            accuracy_pct   = scoring.get("accuracy_pct", 52),
            regime         = regime_data.get("regime", "UNKNOWN"),
            adx            = regime_data.get("adx", 0.0),
            tp1_price      = tiered.get("tp1_price"),
            tp2_price      = tiered.get("tp2_price"),
            tp3_price      = tiered.get("tp3_price"),
            tp1_pct        = tiered.get("tp1_contract_pct", 15.0),
            tp2_pct        = tiered.get("tp2_contract_pct", 50.0),
            tp3_pct        = tiered.get("tp3_contract_pct", 100.0),
            sl_contract_pct = tiered.get("sl_contract_pct", 30.0),
            near_liquidity = liq_data.get("near_liquidity_level") is not None,
            liquidity_type = liq_data.get("near_liquidity_type", ""),
            order_block    = ob_data.get("price_in_bullish_ob") or ob_data.get("price_in_bearish_ob"),
            fvg_signal     = fvg_data.get("price_in_fvg", False),
            near_fib_025   = fib_data.get("near_golden_025", False),
            divergence_type = div_data.get("divergence_type", "none"),
            scoring_details = scoring.get("reasons", []),
        )

    # ──────────────────────────────────────────
 # 
    # ──────────────────────────────────────────

    def _pick_best(self, calls: list, puts: list):
        """     Call  Put."""
        all_signals = (
            [("CALL", *s) for s in calls] +
            [("PUT",  *s) for s in puts]
        )
        if not all_signals:
            return None
        all_signals.sort(key=lambda x: x[2], reverse=True)
        return all_signals[0]
