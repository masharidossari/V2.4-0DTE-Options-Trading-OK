"""
=============================================================
  Options Scalper Bot v4.0 - market_regime.py
  وحدة تحديد حالة السوق والسيولة
  ─────────────────────────────────────────────
  1. Sideways Detector    — هل السوق في اتجاه أم تذبذب؟
  2. Static Liquidity Levels — PDH/PDL/WH/WL/Round Numbers
  3. Order Blocks         — كتل الأوامر المؤسسية
  4. Fair Value Gaps      — فجوات القيمة العادلة
  5. Volume Profile       — POC / VAH / VAL
  6. Fibonacci 0.25       — نقطة الدخول الذهبية
  7. Divergence Detection — الدايفرجنس العادي والمخفي
=============================================================
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. SIDEWAYS DETECTOR
# ─────────────────────────────────────────────

def detect_market_regime(df: pd.DataFrame) -> Dict:
    """
    يحدد ما إذا كان السوق في حالة اتجاه (TRENDING) أم تذبذب (SIDEWAYS).
    يستخدم 5 مقاييس مختلفة ونظام نقاط.

    Returns:
        dict: {
            'regime': 'TRENDING' | 'SIDEWAYS' | 'VOLATILE',
            'score': int (0-5, كلما ارتفع كلما كان السوق أكثر تذبذباً),
            'adx': float,
            'details': list
        }
    """
    if df is None or len(df) < 30:
        return {"regime": "UNKNOWN", "score": 0, "adx": 0, "details": []}

    sideways_score = 0
    details = []

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── المقياس 1: تقارب المتوسطات المتحركة ──────────
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    ema_spread = abs(ema9.iloc[-1] - ema50.iloc[-1]) / close.iloc[-1] * 100
    if ema_spread < 0.3:  # المتوسطات متقاربة جداً
        sideways_score += 1
        details.append(f"تقارب المتوسطات: {ema_spread:.2f}% (< 0.3%)")

    # ── المقياس 2: ADX (مؤشر قوة الاتجاه) ──────────
    adx_val = _calculate_adx(df, period=14)
    if adx_val < 20:  # ADX أقل من 20 = سوق بدون اتجاه
        sideways_score += 1
        details.append(f"ADX ضعيف: {adx_val:.1f} (< 20)")

    # ── المقياس 3: RSI قرب المنتصف ──────────────────
    rsi_series = _calculate_rsi(close, 14)
    rsi_val = rsi_series.iloc[-1]
    if 45 <= rsi_val <= 55:  # RSI في المنطقة المحايدة
        sideways_score += 1
        details.append(f"RSI محايد: {rsi_val:.1f} (45-55)")

    # ── المقياس 4: ضيق نطاق بولينجر باند ───────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_width = (bb_std * 2 / bb_mid * 100).iloc[-1]
    if bb_width < 1.5:  # نطاق ضيق جداً
        sideways_score += 1
        details.append(f"بولينجر ضيق: {bb_width:.2f}% (< 1.5%)")

    # ── المقياس 5: تكرار تقاطعات EMA (علامة التذبذب) ─
    crossovers = 0
    for i in range(-20, -1):
        if (ema9.iloc[i-1] > ema21.iloc[i-1]) != (ema9.iloc[i] > ema21.iloc[i]):
            crossovers += 1
    if crossovers >= 3:  # 3 تقاطعات أو أكثر في آخر 20 شمعة
        sideways_score += 1
        details.append(f"تقاطعات EMA متكررة: {crossovers} مرة في 20 شمعة")

    # ── تحديد النظام ─────────────────────────────────
    if sideways_score >= 3:
        regime = "SIDEWAYS"
    elif adx_val > 30:
        regime = "TRENDING"
    elif adx_val > 20:
        regime = "TRENDING"
    else:
        regime = "SIDEWAYS"

    # التحقق من التذبذب العالي (Volatile)
    atr_val = _calculate_atr(df, 14).iloc[-1]
    atr_pct = atr_val / close.iloc[-1] * 100
    if atr_pct > 3.0:
        regime = "VOLATILE"
        details.append(f"تذبذب عالٍ: ATR = {atr_pct:.2f}%")

    return {
        "regime":  regime,
        "score":   sideways_score,
        "adx":     round(adx_val, 1),
        "atr_pct": round(atr_pct, 2),
        "details": details,
    }


# ─────────────────────────────────────────────
# 2. STATIC LIQUIDITY LEVELS
# ─────────────────────────────────────────────

def get_liquidity_levels(df_daily: pd.DataFrame, current_price: float) -> Dict:
    """
    يحسب مستويات السيولة الثابتة من البيانات اليومية.

    Args:
        df_daily: بيانات يومية (على الأقل 10 أيام)
        current_price: السعر الحالي

    Returns:
        dict: جميع مستويات السيولة مرتبة
    """
    if df_daily is None or len(df_daily) < 5:
        return {}

    levels = {}

    # ── اليوم السابق ─────────────────────────────────
    if len(df_daily) >= 2:
        prev = df_daily.iloc[-2]
        levels["PDH"] = round(float(prev["high"]),  2)   # Previous Day High
        levels["PDL"] = round(float(prev["low"]),   2)   # Previous Day Low
        levels["PDC"] = round(float(prev["close"]), 2)   # Previous Day Close

    # ── أسبوع حالي (آخر 5 أيام) ──────────────────────
    week_data = df_daily.iloc[-5:]
    levels["WH"]  = round(float(week_data["high"].max()),  2)  # Week High
    levels["WL"]  = round(float(week_data["low"].min()),   2)  # Week Low

    # ── أسبوع سابق (5 أيام قبل الأسبوع الحالي) ───────
    if len(df_daily) >= 10:
        prev_week = df_daily.iloc[-10:-5]
        levels["PWH"] = round(float(prev_week["high"].max()), 2)  # Prev Week High
        levels["PWL"] = round(float(prev_week["low"].min()),  2)  # Prev Week Low

    # ── آخر 14 يوم ───────────────────────────────────
    if len(df_daily) >= 14:
        two_weeks = df_daily.iloc[-14:]
        levels["14D_H"] = round(float(two_weeks["high"].max()), 2)
        levels["14D_L"] = round(float(two_weeks["low"].min()),  2)

    # ── الأرقام المستديرة (Round Numbers) ────────────
    levels["round_levels"] = _get_round_numbers(current_price)

    # ── تصنيف المستويات (دعم / مقاومة) ──────────────
    resistance = sorted([v for k, v in levels.items()
                         if isinstance(v, float) and v > current_price])
    support    = sorted([v for k, v in levels.items()
                         if isinstance(v, float) and v < current_price], reverse=True)

    # إضافة الأرقام المستديرة
    for r in levels.get("round_levels", []):
        if r > current_price and r not in resistance:
            resistance.append(r)
        elif r < current_price and r not in support:
            support.append(r)

    resistance.sort()
    support.sort(reverse=True)

    levels["nearest_resistance"] = resistance[0] if resistance else None
    levels["nearest_support"]    = support[0]    if support    else None
    levels["all_resistance"]     = resistance[:5]
    levels["all_support"]        = support[:5]

    # ── هل السعر قرب منطقة سيولة؟ (±0.3%) ──────────
    proximity_threshold = current_price * 0.003
    near_level = None
    near_type  = None
    min_dist   = float("inf")

    for k, v in levels.items():
        if isinstance(v, float) and k not in ["PDC"]:
            dist = abs(v - current_price)
            if dist < proximity_threshold and dist < min_dist:
                min_dist  = dist
                near_level = v
                near_type  = "resistance" if v > current_price else "support"

    levels["near_liquidity_level"] = near_level
    levels["near_liquidity_type"]  = near_type
    levels["near_liquidity_dist_pct"] = round(min_dist / current_price * 100, 3) if near_level else None

    return levels


def _get_round_numbers(price: float) -> List[float]:
    """يولد الأرقام المستديرة القريبة من السعر الحالي."""
    if price > 500:
        step = 50
    elif price > 100:
        step = 10
    elif price > 50:
        step = 5
    else:
        step = 1

    base = round(price / step) * step
    return [round(base + i * step, 2) for i in range(-4, 5)]


# ─────────────────────────────────────────────
# 3. ORDER BLOCKS
# ─────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame, lookback: int = 30) -> Dict:
    """
    يكتشف كتل الأوامر المؤسسية (Order Blocks).

    Order Block صاعد = آخر شمعة هابطة قبل حركة صاعدة قوية
    Order Block هابط = آخر شمعة صاعدة قبل حركة هابطة قوية

    Returns:
        dict: {
            'bullish_ob': {'high': float, 'low': float, 'index': int} | None,
            'bearish_ob': {'high': float, 'low': float, 'index': int} | None,
            'price_in_bullish_ob': bool,
            'price_in_bearish_ob': bool
        }
    """
    if df is None or len(df) < 10:
        return {"bullish_ob": None, "bearish_ob": None,
                "price_in_bullish_ob": False, "price_in_bearish_ob": False}

    current_price = df["close"].iloc[-1]
    avg_volume    = df["volume"].rolling(20).mean().iloc[-1] if "volume" in df.columns else 1
    recent        = df.iloc[-lookback:]

    bullish_ob = None
    bearish_ob = None

    for i in range(len(recent) - 4, 0, -1):
        candle     = recent.iloc[i]
        next_3     = recent.iloc[i+1:i+4]

        is_bearish = candle["close"] < candle["open"]
        is_bullish = candle["close"] > candle["open"]

        # حجم الشمعة الحالية مقارنة بالمتوسط
        candle_vol = candle.get("volume", avg_volume)
        strong_move = (next_3["close"].max() - candle["high"]) / candle["high"] > 0.002

        # Order Block صاعد: شمعة هابطة + حركة صاعدة قوية بعدها
        if is_bearish and strong_move and bullish_ob is None:
            bullish_ob = {
                "high":  round(float(candle["high"]),  4),
                "low":   round(float(candle["low"]),   4),
                "open":  round(float(candle["open"]),  4),
                "close": round(float(candle["close"]), 4),
                "index": i,
            }

        # Order Block هابط: شمعة صاعدة + حركة هابطة قوية بعدها
        strong_down = (candle["low"] - next_3["close"].min()) / candle["low"] > 0.002
        if is_bullish and strong_down and bearish_ob is None:
            bearish_ob = {
                "high":  round(float(candle["high"]),  4),
                "low":   round(float(candle["low"]),   4),
                "open":  round(float(candle["open"]),  4),
                "close": round(float(candle["close"]), 4),
                "index": i,
            }

        if bullish_ob and bearish_ob:
            break

    # هل السعر داخل Order Block؟
    price_in_bull = (bullish_ob is not None and
                     bullish_ob["low"] <= current_price <= bullish_ob["high"])
    price_in_bear = (bearish_ob is not None and
                     bearish_ob["low"] <= current_price <= bearish_ob["high"])

    return {
        "bullish_ob":          bullish_ob,
        "bearish_ob":          bearish_ob,
        "price_in_bullish_ob": price_in_bull,
        "price_in_bearish_ob": price_in_bear,
    }


# ─────────────────────────────────────────────
# 4. FAIR VALUE GAPS (FVG)
# ─────────────────────────────────────────────

def detect_fair_value_gaps(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    يكتشف فجوات القيمة العادلة (Fair Value Gaps / Imbalance).

    FVG صاعد: low[i] > high[i-2]  (فجوة صاعدة)
    FVG هابط: high[i] < low[i-2]  (فجوة هابطة)

    Returns:
        dict: قائمة بالـ FVGs الموجودة وأقربها للسعر الحالي
    """
    if df is None or len(df) < 5:
        return {"bullish_fvg": [], "bearish_fvg": [], "nearest_fvg": None}

    current_price = df["close"].iloc[-1]
    recent = df.iloc[-lookback:]

    bullish_fvgs = []
    bearish_fvgs = []

    for i in range(2, len(recent)):
        c0 = recent.iloc[i]    # الشمعة الحالية
        c2 = recent.iloc[i-2]  # الشمعة قبل السابقة

        # FVG صاعد: أدنى الشمعة الحالية أعلى من أعلى الشمعة قبل السابقة
        if c0["low"] > c2["high"]:
            fvg = {
                "top":    round(float(c0["low"]),  4),
                "bottom": round(float(c2["high"]), 4),
                "type":   "bullish",
                "size":   round(float(c0["low"] - c2["high"]), 4),
            }
            bullish_fvgs.append(fvg)

        # FVG هابط: أعلى الشمعة الحالية أقل من أدنى الشمعة قبل السابقة
        if c0["high"] < c2["low"]:
            fvg = {
                "top":    round(float(c2["low"]),  4),
                "bottom": round(float(c0["high"]), 4),
                "type":   "bearish",
                "size":   round(float(c2["low"] - c0["high"]), 4),
            }
            bearish_fvgs.append(fvg)

    # أقرب FVG للسعر الحالي
    nearest_fvg = None
    min_dist    = float("inf")

    for fvg in bullish_fvgs + bearish_fvgs:
        mid  = (fvg["top"] + fvg["bottom"]) / 2
        dist = abs(mid - current_price)
        if dist < min_dist:
            min_dist    = dist
            nearest_fvg = fvg

    # هل السعر داخل FVG؟
    price_in_fvg = False
    fvg_type     = None
    if nearest_fvg:
        if nearest_fvg["bottom"] <= current_price <= nearest_fvg["top"]:
            price_in_fvg = True
            fvg_type     = nearest_fvg["type"]

    return {
        "bullish_fvg":   bullish_fvgs[-3:] if bullish_fvgs else [],
        "bearish_fvg":   bearish_fvgs[-3:] if bearish_fvgs else [],
        "nearest_fvg":   nearest_fvg,
        "price_in_fvg":  price_in_fvg,
        "fvg_type":      fvg_type,
        "fvg_dist_pct":  round(min_dist / current_price * 100, 3) if nearest_fvg else None,
    }


# ─────────────────────────────────────────────
# 5. VOLUME PROFILE (POC / VAH / VAL)
# ─────────────────────────────────────────────

def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict:
    """
    يحسب ملف الحجم (Volume Profile) لتحديد POC و VAH و VAL.

    POC = Point of Control (السعر الذي تداول عنده أعلى حجم)
    VAH = Value Area High (الحد الأعلى لمنطقة 70% من التداول)
    VAL = Value Area Low  (الحد الأدنى لمنطقة 70% من التداول)
    """
    if df is None or len(df) < 10 or "volume" not in df.columns:
        return {"poc": None, "vah": None, "val": None}

    price_min = df["low"].min()
    price_max = df["high"].max()

    if price_max <= price_min:
        return {"poc": None, "vah": None, "val": None}

    # تقسيم النطاق السعري إلى bins
    bin_edges  = np.linspace(price_min, price_max, bins + 1)
    bin_volume = np.zeros(bins)

    for _, row in df.iterrows():
        # توزيع حجم الشمعة على الـ bins التي يمتد إليها نطاقها
        candle_low  = row["low"]
        candle_high = row["high"]
        candle_vol  = row["volume"]

        for b in range(bins):
            bin_low  = bin_edges[b]
            bin_high = bin_edges[b + 1]
            overlap  = max(0, min(candle_high, bin_high) - max(candle_low, bin_low))
            candle_range = candle_high - candle_low
            if candle_range > 0:
                bin_volume[b] += candle_vol * (overlap / candle_range)

    # POC = الـ bin بأعلى حجم
    poc_idx    = np.argmax(bin_volume)
    poc        = round((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2, 4)

    # Value Area = 70% من إجمالي الحجم حول الـ POC
    total_vol   = bin_volume.sum()
    target_vol  = total_vol * 0.70
    va_vol      = bin_volume[poc_idx]
    va_low_idx  = poc_idx
    va_high_idx = poc_idx

    while va_vol < target_vol:
        expand_low  = va_low_idx  > 0
        expand_high = va_high_idx < bins - 1

        if not expand_low and not expand_high:
            break

        add_low  = bin_volume[va_low_idx  - 1] if expand_low  else 0
        add_high = bin_volume[va_high_idx + 1] if expand_high else 0

        if add_high >= add_low:
            va_high_idx += 1
            va_vol += add_high
        else:
            va_low_idx -= 1
            va_vol += add_low

    vah = round(bin_edges[va_high_idx + 1], 4)
    val = round(bin_edges[va_low_idx],      4)

    current_price = df["close"].iloc[-1]

    return {
        "poc":              poc,
        "vah":              vah,
        "val":              val,
        "price_above_poc":  current_price > poc,
        "price_in_va":      val <= current_price <= vah,
        "dist_to_poc_pct":  round(abs(current_price - poc) / current_price * 100, 3),
    }


# ─────────────────────────────────────────────
# 6. FIBONACCI 0.25 (نقطة الدخول الذهبية)
# ─────────────────────────────────────────────

def calculate_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    يحسب مستويات فيبوناتشي بناءً على آخر حركة كبيرة.
    يستخدم Pivot Highs و Pivot Lows لتحديد بداية ونهاية الحركة.

    مستويات الارتداد: 0.114, 0.25, 0.38, 0.50, 0.618, 0.79, 0.88
    مستويات الامتداد: 1.25, 1.50, 2.00, 2.50, 3.00
    """
    if df is None or len(df) < 20:
        return {}

    recent = df.iloc[-lookback:]
    current_price = df["close"].iloc[-1]

    # تحديد أعلى قمة وأدنى قاع في الفترة
    swing_high = float(recent["high"].max())
    swing_low  = float(recent["low"].min())
    fib_range  = swing_high - swing_low

    if fib_range <= 0:
        return {}

    # تحديد اتجاه الحركة الأخيرة
    first_half_avg = recent.iloc[:len(recent)//2]["close"].mean()
    second_half_avg = recent.iloc[len(recent)//2:]["close"].mean()
    trend_up = second_half_avg > first_half_avg

    # ── مستويات الارتداد (Retracement) ───────────────
    retracement_ratios = [0.114, 0.25, 0.38, 0.50, 0.618, 0.70, 0.79, 0.88]
    extension_ratios   = [1.25, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00]

    retracements = {}
    extensions   = {}

    if trend_up:
        # الحركة صاعدة: الارتداد ينزل من القمة
        for r in retracement_ratios:
            level = swing_high - fib_range * r
            retracements[f"fib_{str(r).replace('.', '_')}"] = round(level, 4)
        # الامتداد يصعد فوق القمة
        for e in extension_ratios:
            level = swing_low + fib_range * e
            extensions[f"ext_{str(e).replace('.', '_')}"] = round(level, 4)
    else:
        # الحركة هابطة: الارتداد يصعد من القاع
        for r in retracement_ratios:
            level = swing_low + fib_range * r
            retracements[f"fib_{str(r).replace('.', '_')}"] = round(level, 4)
        # الامتداد ينزل تحت القاع
        for e in extension_ratios:
            level = swing_high - fib_range * e
            extensions[f"ext_{str(e).replace('.', '_')}"] = round(level, 4)

    # ── هل السعر قرب مستوى 0.25 الذهبي؟ ─────────────
    golden_entry = retracements.get("fib_0_25")
    near_golden  = False
    golden_dist  = None

    if golden_entry:
        golden_dist = abs(current_price - golden_entry) / current_price * 100
        near_golden = golden_dist < 0.3  # ±0.3% من مستوى 0.25

    # ── أقرب مستوى فيبو للسعر الحالي ────────────────
    all_levels = {**retracements, **extensions}
    nearest_fib = None
    nearest_fib_name = None
    min_dist = float("inf")

    for name, level in all_levels.items():
        dist = abs(level - current_price)
        if dist < min_dist:
            min_dist = dist
            nearest_fib = level
            nearest_fib_name = name

    return {
        "swing_high":      round(swing_high, 4),
        "swing_low":       round(swing_low,  4),
        "fib_range":       round(fib_range,  4),
        "trend_up":        trend_up,
        "retracements":    retracements,
        "extensions":      extensions,
        "golden_entry":    golden_entry,
        "near_golden_025": near_golden,
        "golden_dist_pct": round(golden_dist, 3) if golden_dist else None,
        "nearest_fib":     nearest_fib,
        "nearest_fib_name": nearest_fib_name,
        "nearest_fib_dist_pct": round(min_dist / current_price * 100, 3),
    }


# ─────────────────────────────────────────────
# 7. DIVERGENCE DETECTION (عادي ومخفي)
# ─────────────────────────────────────────────

def detect_divergence(df: pd.DataFrame, rsi_period: int = 14,
                       lb_left: int = 5, lb_right: int = 3) -> Dict:
    """
    يكتشف الدايفرجنس بنوعيه:
    - العادي (انعكاسي): إشارة انعكاس الاتجاه
    - المخفي (استمراري): إشارة استمرار الاتجاه

    Returns:
        dict: {
            'regular_bullish': bool,   # انعكاس صاعد
            'regular_bearish': bool,   # انعكاس هابط
            'hidden_bullish':  bool,   # استمرار صعود
            'hidden_bearish':  bool,   # استمرار هبوط
            'divergence_type': str,    # ملخص نوع الدايفرجنس
            'strength': float          # قوة الدايفرجنس (0-3)
        }
    """
    if df is None or len(df) < lb_left + lb_right + 10:
        return {
            "regular_bullish": False, "regular_bearish": False,
            "hidden_bullish":  False, "hidden_bearish":  False,
            "divergence_type": "none", "strength": 0.0
        }

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # حساب RSI
    rsi_series = _calculate_rsi(close, rsi_period)

    # تحديد Pivot Highs و Pivot Lows
    pivot_highs = _find_pivots(rsi_series, lb_left, lb_right, "high")
    pivot_lows  = _find_pivots(rsi_series, lb_left, lb_right, "low")

    regular_bullish = False
    regular_bearish = False
    hidden_bullish  = False
    hidden_bearish  = False

    # ── الدايفرجنس الصاعد العادي ──────────────────────
    # قاع سعر أقل + قاع RSI أعلى → انعكاس صاعد محتمل
    if len(pivot_lows) >= 2:
        idx1, idx2 = pivot_lows[-2], pivot_lows[-1]
        if (low.iloc[idx2] < low.iloc[idx1] and
                rsi_series.iloc[idx2] > rsi_series.iloc[idx1]):
            regular_bullish = True

    # ── الدايفرجنس الهابط العادي ──────────────────────
    # قمة سعر أعلى + قمة RSI أقل → انعكاس هابط محتمل
    if len(pivot_highs) >= 2:
        idx1, idx2 = pivot_highs[-2], pivot_highs[-1]
        if (high.iloc[idx2] > high.iloc[idx1] and
                rsi_series.iloc[idx2] < rsi_series.iloc[idx1]):
            regular_bearish = True

    # ── الدايفرجنس الصاعد المخفي ──────────────────────
    # قاع سعر أعلى + قاع RSI أقل → استمرار صعود
    if len(pivot_lows) >= 2:
        idx1, idx2 = pivot_lows[-2], pivot_lows[-1]
        if (low.iloc[idx2] > low.iloc[idx1] and
                rsi_series.iloc[idx2] < rsi_series.iloc[idx1]):
            hidden_bullish = True

    # ── الدايفرجنس الهابط المخفي ──────────────────────
    # قمة سعر أقل + قمة RSI أعلى → استمرار هبوط
    if len(pivot_highs) >= 2:
        idx1, idx2 = pivot_highs[-2], pivot_highs[-1]
        if (high.iloc[idx2] < high.iloc[idx1] and
                rsi_series.iloc[idx2] > rsi_series.iloc[idx1]):
            hidden_bearish = True

    # ── تحديد نوع الدايفرجنس وقوته ──────────────────
    strength = 0.0
    div_type = "none"

    if regular_bullish:
        div_type  = "regular_bullish"
        strength += 2.0
    elif regular_bearish:
        div_type  = "regular_bearish"
        strength += 2.0
    elif hidden_bullish:
        div_type  = "hidden_bullish"
        strength += 1.5
    elif hidden_bearish:
        div_type  = "hidden_bearish"
        strength += 1.5

    # تعزيز القوة إذا تزامن مع RSI في منطقة تشبع
    rsi_now = rsi_series.iloc[-1]
    if regular_bullish and rsi_now < 35:
        strength += 1.0
    elif regular_bearish and rsi_now > 65:
        strength += 1.0

    return {
        "regular_bullish": regular_bullish,
        "regular_bearish": regular_bearish,
        "hidden_bullish":  hidden_bullish,
        "hidden_bearish":  hidden_bearish,
        "divergence_type": div_type,
        "strength":        round(strength, 1),
        "rsi_current":     round(float(rsi_now), 1),
    }


# ─────────────────────────────────────────────
# 8. SIGNAL SCORING — نظام تصنيف A/B/C
# ─────────────────────────────────────────────

def score_signal(
    direction: str,
    regime: Dict,
    liquidity: Dict,
    order_blocks: Dict,
    fvg: Dict,
    fib: Dict,
    divergence: Dict,
    current_price: float,
) -> Dict:
    """
    يحسب نقاط الإشارة ويصنفها إلى A+ / A / B / C.

    كل شرط يعطي نقطة واحدة (0 أو 1).
    المجموع الأقصى: 5 نقاط.

    Returns:
        dict: {
            'grade': 'A+' | 'A' | 'B' | 'C',
            'score': int (0-5),
            'accuracy_pct': int,
            'conditions': dict,
            'reasons': list
        }
    """
    score = 0
    conditions = {}
    reasons    = []

    # ── الشرط 1: السوق في اتجاه (Trending) ──────────
    is_trending = regime.get("regime") == "TRENDING"
    conditions["trending"] = is_trending
    if is_trending:
        score += 1
        reasons.append(f"✅ السوق في اتجاه (ADX: {regime.get('adx', 0):.0f})")
    else:
        reasons.append(f"⚠️ السوق متذبذب (ADX: {regime.get('adx', 0):.0f})")

    # ── الشرط 2: السعر قرب منطقة سيولة ──────────────
    near_liq = liquidity.get("near_liquidity_level") is not None
    conditions["near_liquidity"] = near_liq
    if near_liq:
        score += 1
        liq_type = liquidity.get("near_liquidity_type", "")
        liq_dist = liquidity.get("near_liquidity_dist_pct", 0)
        reasons.append(f"✅ قرب منطقة سيولة ({liq_type}, {liq_dist:.2f}%)")

    # ── الشرط 3: Order Block أو FVG ──────────────────
    ob_confirmed = False
    if direction == "CALL":
        ob_confirmed = (order_blocks.get("price_in_bullish_ob") or
                        (fvg.get("price_in_fvg") and fvg.get("fvg_type") == "bullish"))
    else:
        ob_confirmed = (order_blocks.get("price_in_bearish_ob") or
                        (fvg.get("price_in_fvg") and fvg.get("fvg_type") == "bearish"))

    conditions["ob_or_fvg"] = ob_confirmed
    if ob_confirmed:
        score += 1
        if order_blocks.get("price_in_bullish_ob") or order_blocks.get("price_in_bearish_ob"):
            reasons.append("✅ السعر داخل Order Block مؤسسي")
        else:
            reasons.append("✅ السعر داخل Fair Value Gap")

    # ── الشرط 4: مستوى فيبوناتشي 0.25 أو 0.38 ───────
    near_fib = False
    fib_name = ""
    if fib.get("near_golden_025"):
        near_fib = True
        fib_name = "0.25 الذهبي"
    elif fib.get("nearest_fib_dist_pct", 999) < 0.4:
        nearest_name = fib.get("nearest_fib_name", "")
        if "0_38" in nearest_name or "0_50" in nearest_name:
            near_fib = True
            fib_name = nearest_name.replace("fib_", "").replace("_", ".")

    conditions["near_fib"] = near_fib
    if near_fib:
        score += 1
        reasons.append(f"✅ قرب مستوى فيبو {fib_name}")

    # ── الشرط 5: دايفرجنس مؤكد ───────────────────────
    div_confirmed = False
    if direction == "CALL":
        div_confirmed = (divergence.get("regular_bullish") or
                         divergence.get("hidden_bullish"))
    else:
        div_confirmed = (divergence.get("regular_bearish") or
                         divergence.get("hidden_bearish"))

    conditions["divergence"] = div_confirmed
    if div_confirmed:
        score += 1
        div_type = divergence.get("divergence_type", "")
        if "regular" in div_type:
            reasons.append(f"✅ دايفرجنس انعكاسي ({div_type})")
        else:
            reasons.append(f"✅ دايفرجنس استمراري ({div_type})")

    # ── تحديد الدرجة ─────────────────────────────────
    if score == 5:
        grade        = "A+"
        accuracy_pct = 82
    elif score == 4:
        grade        = "A"
        accuracy_pct = 72
    elif score == 3:
        grade        = "B"
        accuracy_pct = 62
    else:
        grade        = "C"
        accuracy_pct = 52

    return {
        "grade":        grade,
        "score":        score,
        "accuracy_pct": accuracy_pct,
        "conditions":   conditions,
        "reasons":      reasons,
    }


# ─────────────────────────────────────────────
# 9. نظام الأهداف المتدرج TP1/TP2/TP3
# ─────────────────────────────────────────────

def calculate_tiered_targets(
    direction: str,
    entry_price: float,
    liquidity: Dict,
    fib: Dict,
    atr_value: float = None,
) -> Dict:
    """
    يحسب الأهداف المتدرجة ووقف الخسارة بناءً على مستويات السيولة والفيبو.

    TP1: +15% على العقد (هدف محافظ)
    TP2: +50% على العقد (هدف متوسط)
    TP3: +100% على العقد (هدف طموح)
    SL:  -30% على العقد (وقف الخسارة)
    """
    # نسب الأهداف على سعر العقد
    TP1_PCT = 0.15   # +15%
    TP2_PCT = 0.50   # +50%
    TP3_PCT = 1.00   # +100%
    SL_PCT  = 0.30   # -30%

    # أهداف السعر الأساسية (على سعر الأصل)
    if direction == "CALL":
        # استخدام مستويات المقاومة كأهداف
        resistances = liquidity.get("all_resistance", [])
        tp1_price = resistances[0] if len(resistances) > 0 else round(entry_price * 1.005, 2)
        tp2_price = resistances[1] if len(resistances) > 1 else round(entry_price * 1.010, 2)
        tp3_price = resistances[2] if len(resistances) > 2 else round(entry_price * 1.020, 2)

        # استخدام مستويات امتداد فيبو كأهداف بديلة
        ext_125 = fib.get("extensions", {}).get("ext_1_25")
        ext_150 = fib.get("extensions", {}).get("ext_1_50")
        ext_200 = fib.get("extensions", {}).get("ext_2_0")

        if ext_125 and ext_125 > entry_price:
            tp1_price = min(tp1_price, ext_125) if tp1_price else ext_125
        if ext_150 and ext_150 > entry_price:
            tp2_price = min(tp2_price, ext_150) if tp2_price else ext_150
        if ext_200 and ext_200 > entry_price:
            tp3_price = min(tp3_price, ext_200) if tp3_price else ext_200

        sl_price = liquidity.get("nearest_support") or round(entry_price * 0.995, 2)

    else:  # PUT
        # استخدام مستويات الدعم كأهداف
        supports = liquidity.get("all_support", [])
        tp1_price = supports[0] if len(supports) > 0 else round(entry_price * 0.995, 2)
        tp2_price = supports[1] if len(supports) > 1 else round(entry_price * 0.990, 2)
        tp3_price = supports[2] if len(supports) > 2 else round(entry_price * 0.980, 2)

        sl_price = liquidity.get("nearest_resistance") or round(entry_price * 1.005, 2)

    return {
        # أهداف على سعر الأصل
        "tp1_price": round(tp1_price, 2) if tp1_price else None,
        "tp2_price": round(tp2_price, 2) if tp2_price else None,
        "tp3_price": round(tp3_price, 2) if tp3_price else None,
        "sl_price":  round(sl_price,  2) if sl_price  else None,

        # أهداف على سعر العقد (0DTE)
        "tp1_contract_pct": TP1_PCT * 100,   # +15%
        "tp2_contract_pct": TP2_PCT * 100,   # +50%
        "tp3_contract_pct": TP3_PCT * 100,   # +100%
        "sl_contract_pct":  SL_PCT  * 100,   # -30%
    }


# ─────────────────────────────────────────────
# دوال مساعدة (Private)
# ─────────────────────────────────────────────

def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """يحسب مؤشر ADX لقياس قوة الاتجاه."""
    try:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr_s     = tr.ewm(com=period - 1, min_periods=period).mean()
        plus_di   = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_s
        minus_di  = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_s
        dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx       = dx.ewm(com=period - 1, min_periods=period).mean()

        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0
    except Exception:
        return 0.0


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _find_pivots(series: pd.Series, lb_left: int, lb_right: int,
                  pivot_type: str = "high") -> List[int]:
    """يجد مواضع Pivot Highs أو Pivot Lows في السلسلة."""
    pivots = []
    for i in range(lb_left, len(series) - lb_right):
        window = series.iloc[i - lb_left: i + lb_right + 1]
        center = series.iloc[i]
        if pivot_type == "high" and center == window.max():
            pivots.append(i)
        elif pivot_type == "low" and center == window.min():
            pivots.append(i)
    return pivots
