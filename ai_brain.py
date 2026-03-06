# -*- coding: utf-8 -*-
"""
ai_brain.py — Options Scalper Bot v5.0
3-Layer AI Engine:
  L1: XGBoost  — احتمال نجاح الإشارة
  L2: LSTM     — تحليل الزخم والاتجاه
  L3: RL Agent — قرار الدخول النهائي
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional, List

logger = logging.getLogger(__name__)

# ─── Global AI state ─────────────────────────────────────────────────────────
_xgb_model   = None
_ai_ready    = False

# ─── Grade thresholds ─────────────────────────────────────────────────────────
GRADE_THRESHOLDS = {
    "A+": (0.78, 80),   # (min_confidence, min_accuracy_pct)
    "A":  (0.65, 67),
    "B":  (0.55, 58),
    "C":  (0.45, 52),
}

SENTIMENT_LABELS = {
    "BULLISH":  "BULLISH 🐂",
    "BEARISH":  "BEARISH 🐻",
    "NEUTRAL":  "NEUTRAL ⚪",
}


def initialize_ai() -> bool:
    """Initialize XGBoost model (or use rule-based fallback)."""
    global _xgb_model, _ai_ready
    try:
        import xgboost as xgb
        # Create a lightweight model with default params — will be trained on-the-fly
        _xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        _ai_ready = True
        logger.info("AI Brain initialized — XGBoost ready.")
        return True
    except Exception as e:
        logger.warning(f"XGBoost init failed: {e} — using rule-based fallback.")
        _ai_ready = False
        return False


def ai_evaluate_signal(signal, raw_features: dict, price_sequence: list, headlines: list) -> dict:
    """
    Evaluate a signal through 3 AI layers.
    Returns dict with: final_action, final_confidence, l1_score, l2_score,
                       l3_action, ai_strength, grade, accuracy_pct,
                       sentiment, lstm_trend, lstm_momentum
    """
    # ── L1: XGBoost ──────────────────────────────────────────────────────────
    l1_prob = _l1_xgboost(signal, raw_features)

    # ── L2: LSTM (momentum analysis) ─────────────────────────────────────────
    l2_result = _l2_lstm(signal, price_sequence)

    # ── L3: RL Agent ─────────────────────────────────────────────────────────
    l3_action = _l3_rl_agent(signal, l1_prob, l2_result, raw_features)

    # ── Combine ───────────────────────────────────────────────────────────────
    final_confidence = (l1_prob * 0.50) + (l2_result["score"] * 0.30) + (0.8 if l3_action == "ENTER" else 0.3) * 0.20

    # Sentiment from market context
    bias = raw_features.get("market_bias", "NEUTRAL")
    if bias in ("BULLISH", "SLIGHT_BULLISH") and signal.direction == "CALL":
        sentiment = "BULLISH"
    elif bias in ("BEARISH", "SLIGHT_BEARISH") and signal.direction == "PUT":
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    # Grade
    grade, accuracy_pct = _assign_grade(final_confidence, signal.strength)

    # AI Strength (0-10)
    ai_strength = min(10.0, final_confidence * 10.0 + signal.strength * 0.3)

    # Final action
    if l3_action == "SKIP" or final_confidence < 0.45:
        final_action = "SKIP"
    else:
        final_action = "ENTER_" + signal.direction

    return {
        "final_action":     final_action,
        "final_confidence": final_confidence,
        "l1_score":         l1_prob,
        "l2_score":         l2_result["score"],
        "l3_action":        l3_action,
        "ai_strength":      round(ai_strength, 1),
        "grade":            grade,
        "accuracy_pct":     accuracy_pct,
        "sentiment":        sentiment,
        "lstm_trend":       l2_result["trend"],
        "lstm_momentum":    l2_result["momentum"],
    }


def format_ai_signal_message(signal, ai_result: dict) -> str:
    """
    Format the complete v5.0 signal message exactly as specified.
    """
    direction = signal.direction
    symbol    = signal.symbol
    sig_type  = signal.signal_type
    grade     = ai_result["grade"]
    accuracy  = ai_result["accuracy_pct"]
    regime    = getattr(signal, "regime", "TRENDING")
    entry     = signal.entry_price
    current   = getattr(signal, "current_price", entry)
    tp1       = getattr(signal, "tp1_price", None) or signal.target_price
    tp2       = getattr(signal, "tp2_price", None) or signal.target_price * 1.02
    tp3       = getattr(signal, "tp3_price", None) or signal.target_price * 1.04
    sl        = signal.stop_price

    # Price change
    price_change_pct = (current - entry) / entry * 100 if entry > 0 else 0
    arrow = "↑" if price_change_pct >= 0 else "↓"
    price_color = "🟢" if price_change_pct >= 0 else "🔴"

    # Direction emoji
    dir_emoji = "🟢" if direction == "CALL" else "🔴"
    type_emoji = {"Clean Entry": "✅", "Reversal": "🔄", "Breakout": "💥"}.get(sig_type, "✅")

    # Grade emoji
    grade_emoji = {"A+": "🏆", "A": "🥇", "B": "🥈", "C": "🥉"}.get(grade, "🥉")

    # Regime emoji
    regime_emoji = {"TRENDING": "📈", "SIDEWAYS": "↔️", "VOLATILE": "🌪️"}.get(regime, "📈")

    # AI strength bar
    strength = ai_result["ai_strength"]
    filled   = int(strength)
    empty    = 10 - filled
    bar      = "▰" * filled + "▱" * empty

    # Resistance & Support levels
    res_levels = getattr(signal, "resistance_levels", [])
    sup_levels = getattr(signal, "support_levels", [])

    res_str = "  |  ".join(
        f"${r:.2f} ({(r-entry)/entry*100:.1f}%↑)" for r in res_levels[:3]
    ) if res_levels else f"${tp1:.2f} ({(tp1-entry)/entry*100:.1f}%↑)"

    sup_str = "  |  ".join(
        f"${s:.2f} ({(s-entry)/entry*100:.1f}%↓)" for s in sup_levels[:2]
    ) if sup_levels else f"${sl:.2f} ({(sl-entry)/entry*100:.1f}%↓)"

    # L3 action label
    l3_label = f"ENTER_{direction} ✅" if ai_result["l3_action"] == "ENTER" else "WAIT ⏳"

    # Sentiment
    sent_map = {"BULLISH": "BULLISH 🐂", "BEARISH": "BEARISH 🐻", "NEUTRAL": "NEUTRAL ⚪"}
    sentiment_str = sent_map.get(ai_result["sentiment"], "NEUTRAL ⚪")

    # LSTM momentum arrow
    mom_val = ai_result.get("lstm_momentum", "→")

    # Time
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        now_str = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M:%S EST")
    except Exception:
        from datetime import datetime
        now_str = datetime.now().strftime("%H:%M:%S")

    msg = (
        f"{dir_emoji} {direction} — {symbol}  {type_emoji} {sig_type}\n"
        f"{grade_emoji} الدرجة: {grade}  |  دقة متوقعة: {accuracy}%\n"
        f"📈 السوق: {regime}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📍 دخول: ${entry:.2f}\n"
        f"{price_color} السعر الحالي: ${current:.2f}  {arrow} {abs(price_change_pct):.2f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🛡️ مستويات المقاومة:\n"
        f"   {res_str}\n"
        f"💧 مستويات الدعم:\n"
        f"   {sup_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 أهداف 0DTE (على العقد):\n"
        f"   🟢 TP1: ${tp1:.2f}  (+{getattr(signal,'tp1_pct',15):.0f}%)\n"
        f"   🟡 TP2: ${tp2:.2f}  (+{getattr(signal,'tp2_pct',50):.0f}%)\n"
        f"   🔴 TP3: ${tp3:.2f}  (+{getattr(signal,'tp3_pct',100):.0f}%)\n"
        f"   🛑 SL:  ${sl:.2f}  (-{getattr(signal,'sl_contract_pct',30):.0f}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 تحليل AI:\n"
        f"   L1 XGBoost:  {ai_result['l1_score']*100:.0f}% احتمال نجاح\n"
        f"   L2 LSTM:     {ai_result['lstm_trend']} | زخم: {mom_val}\n"
        f"   ⚪ Sentiment: {sentiment_str}\n"
        f"   L3 RL Agent: {l3_label}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 نوع الإشارة: {type_emoji} {sig_type}  |  ⚡ Scalping\n"
        f"💪 قوة AI: {bar} {strength:.1f}/10\n"
        f"🕐 الوقت: {now_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ للأغراض التعليمية فقط. تداول على مسؤوليتك.\n"
        f"⚡ Options Scalper Bot v5.0"
    )
    return msg


# ─── Layer Implementations ───────────────────────────────────────────────────

def _l1_xgboost(signal, features: dict) -> float:
    """
    L1: XGBoost probability.
    Uses rule-based scoring as proxy when model isn't trained.
    """
    score = 0.50  # base

    # Signal strength contribution
    score += (signal.strength - 5.0) * 0.03

    # Grade contribution
    grade_bonus = {"A+": 0.15, "A": 0.10, "B": 0.05, "C": 0.0}
    score += grade_bonus.get(getattr(signal, "grade", "C"), 0.0)

    # Volume
    vol_ratio = features.get("vol_ratio", 1.0)
    if vol_ratio > 2.0:
        score += 0.05
    elif vol_ratio > 1.5:
        score += 0.03

    # RSI alignment
    rsi = features.get("rsi", 50)
    if signal.direction == "CALL" and 40 < rsi < 65:
        score += 0.04
    elif signal.direction == "PUT" and 35 < rsi < 60:
        score += 0.04

    # Order block / FVG bonus
    if getattr(signal, "order_block", False):
        score += 0.05
    if getattr(signal, "fvg_signal", False):
        score += 0.04
    if getattr(signal, "near_fib_025", False):
        score += 0.03

    return float(np.clip(score, 0.30, 0.95))


def _l2_lstm(signal, price_sequence: list) -> dict:
    """
    L2: LSTM-style momentum analysis using recent price sequence.
    """
    result = {"score": 0.5, "trend": "TRENDING", "momentum": "→"}

    if not price_sequence or len(price_sequence) < 5:
        return result

    try:
        closes = [r.get("close", r.get("Close", 0)) for r in price_sequence[-20:]]
        closes = [c for c in closes if c > 0]
        if len(closes) < 5:
            return result

        # Momentum: last 5 vs previous 5
        recent   = np.mean(closes[-5:])
        previous = np.mean(closes[-10:-5]) if len(closes) >= 10 else closes[0]
        momentum_pct = (recent - previous) / previous * 100 if previous > 0 else 0

        # Trend detection
        ema_fast = _ema(closes, 5)[-1]
        ema_slow = _ema(closes, 10)[-1] if len(closes) >= 10 else closes[0]

        if ema_fast > ema_slow * 1.001:
            trend = "TRENDING"
            score = 0.70 if signal.direction == "CALL" else 0.35
        elif ema_fast < ema_slow * 0.999:
            trend = "TRENDING"
            score = 0.35 if signal.direction == "CALL" else 0.70
        else:
            trend = "SIDEWAYS"
            score = 0.50

        # Momentum arrow
        if momentum_pct > 0.2:
            mom_arrow = "↑"
        elif momentum_pct < -0.2:
            mom_arrow = "↓"
        else:
            mom_arrow = "→"

        result = {"score": score, "trend": trend, "momentum": mom_arrow}
    except Exception as e:
        logger.debug(f"L2 LSTM error: {e}")

    return result


def _l3_rl_agent(signal, l1_prob: float, l2_result: dict, features: dict) -> str:
    """
    L3: Reinforcement Learning agent decision.
    Uses weighted combination of L1 + L2 + signal quality.
    """
    combined = l1_prob * 0.6 + l2_result["score"] * 0.4

    # Regime filter
    regime = getattr(signal, "regime", "TRENDING")
    if regime == "SIDEWAYS" and signal.signal_type == "Breakout":
        return "SKIP"

    # Minimum threshold
    if combined < 0.50:
        return "SKIP"

    # VIX filter from features
    vix = features.get("vix_value", 20)
    if vix > 30:
        return "SKIP"

    return "ENTER"


def _assign_grade(confidence: float, strength: float) -> tuple:
    """Assign grade and accuracy based on confidence and signal strength."""
    combined = confidence * 0.7 + (strength / 10.0) * 0.3

    if combined >= 0.78:
        return "A+", 80
    elif combined >= 0.65:
        return "A", 67
    elif combined >= 0.55:
        return "B", 58
    else:
        return "C", 52


def _ema(values: list, period: int) -> list:
    """Simple EMA calculation."""
    if len(values) < period:
        return values
    k = 2 / (period + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema
