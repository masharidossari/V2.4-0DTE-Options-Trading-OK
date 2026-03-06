"""
=============================================================
  Options Scalper Bot v5.0 - notifier.py
  Professional Telegram Notification System
  - Rich signal messages with all layers
  - Market context summary
  - Options flow data
  - Risk management info
  - Performance stats
=============================================================
"""

import logging
from datetime import datetime

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS
from signal_engine import Signal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Emoji Maps
# ─────────────────────────────────────────────
DIRECTION_EMOJI = {"CALL": "🟢", "PUT": "🔴"}
TYPE_EMOJI      = {"Clean Entry": "✅", "Reversal": "🔄", "Breakout": "💥"}
MODE_EMOJI      = {"Scalping": "⚡", "Day Trading": "📊"}
BIAS_EMOJI      = {
    "BULLISH":        "🐂",
    "SLIGHT_BULLISH": "📈",
    "NEUTRAL":        "➡️",
    "SLIGHT_BEARISH": "📉",
    "BEARISH":        "🐻",
    "AVOID":          "🚫",
}
VIX_EMOJI = {
    "LOW":      "😴",
    "NORMAL":   "✅",
    "ELEVATED": "⚠️",
    "HIGH":     "🔥",
    "EXTREME":  "💀",
}

STRENGTH_BARS = [
    (9.0, "▰▰▰▰▰ ELITE"),
    (8.0, "▰▰▰▰▱ STRONG"),
    (7.0, "▰▰▰▱▱ GOOD"),
    (6.0, "▰▰▱▱▱ FAIR"),
    (5.0, "▰▱▱▱▱ WEAK"),
    (0.0, "▱▱▱▱▱ MINIMAL"),
]


def _strength_bar(s: float) -> str:
    for threshold, label in STRENGTH_BARS:
        if s >= threshold:
            return label
    return "▱▱▱▱▱"


def _escape(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r'_*[]()~`>#+-=|{}.!'
    for ch in special:
        text = text.replace(ch, f'\\{ch}')
    return text


# تصنيف الدرجة مع الإيموجي
GRADE_EMOJI = {
    "A+": "🏆",
    "A":  "🥇",
    "B":  "🥈",
    "C":  "🥉",
}
REGIME_EMOJI = {
    "TRENDING":  "📈",
    "SIDEWAYS":  "↔️",
    "VOLATILE":  "🌪️",
    "UNKNOWN":   "❓",
}


def format_signal_message(signal: Signal) -> str:
    """
    Format a rich professional signal message for Telegram — v5.0.
    يتضمن: تصنيف A/B/C + أهداف TP1/TP2/TP3 + حالة السوق + مستويات السيولة
    """
    dir_emoji   = DIRECTION_EMOJI.get(signal.direction, "⚪")
    type_emoji  = TYPE_EMOJI.get(signal.signal_type, "📊")
    mode_emoji  = MODE_EMOJI.get(signal.mode, "📊")
    grade_emoji = GRADE_EMOJI.get(signal.grade, "📊")
    regime_emoji = REGIME_EMOJI.get(signal.regime, "❓")
    bar         = _strength_bar(signal.strength)
    now         = datetime.now().strftime("%H:%M:%S")

    up_arrow   = "⬆️" if signal.direction == "CALL" else "⬇️"
    down_arrow = "⬇️" if signal.direction == "CALL" else "⬆️"

    # ── تصنيف الإشارة ─────────────────────────────
    grade_line = (
        f"{grade_emoji} *الدرجة: {signal.grade}* "
        f"\| دقة متوقعة: *{signal.accuracy_pct}%*\n"
    )

    # ── حالة السوق ───────────────────────────────
    regime_line = (
        f"{regime_emoji} *حالة السوق:* {_escape(signal.regime)} "
        f"\(ADX: {signal.adx:.0f}\)\n"
    )

    # ── مؤشرات السيولة ────────────────────────────
    liquidity_parts = []
    if signal.near_fib_025:
        liquidity_parts.append("🎯 قرب فيبو 0\.25 الذهبي")
    if signal.order_block:
        liquidity_parts.append("🏦 Order Block مؤسسي")
    if signal.fvg_signal:
        liquidity_parts.append("📐 Fair Value Gap")
    if signal.near_liquidity:
        liq_type = signal.liquidity_type or "منطقة"
        liquidity_parts.append(f"💧 سيولة \({_escape(liq_type)}\)")
    if signal.divergence_type and signal.divergence_type != "none":
        div_map = {
            "regular_bullish": "🔁 دايفرجنس انعكاسي صاعد",
            "regular_bearish": "🔁 دايفرجنس انعكاسي هابط",
            "hidden_bullish":  "↩️ دايفرجنس استمراري صاعد",
            "hidden_bearish":  "↩️ دايفرجنس استمراري هابط",
        }
        liquidity_parts.append(div_map.get(signal.divergence_type, ""))

    liquidity_block = ""
    if liquidity_parts:
        liq_lines = "\n".join([f"   {p}" for p in liquidity_parts if p])
        liquidity_block = f"💎 *تأكيدات إضافية:*\n{liq_lines}\n{'─' * 28}\n"

    # ── الأهداف المتدرجة TP1/TP2/TP3 ─────────────
    tp_block = ""
    if signal.tp1_price or signal.tp2_price or signal.tp3_price:
        tp_lines = []
        if signal.tp1_price:
            tp_lines.append(f"   TP1: \\${signal.tp1_price:,.2f} \(\+{signal.tp1_pct:.0f}% على العقد\)")
        if signal.tp2_price:
            tp_lines.append(f"   TP2: \\${signal.tp2_price:,.2f} \(\+{signal.tp2_pct:.0f}% على العقد\)")
        if signal.tp3_price:
            tp_lines.append(f"   TP3: \\${signal.tp3_price:,.2f} \(\+{signal.tp3_pct:.0f}% على العقد\)")
        tp_block = (
            f"🎯 *أهداف 0DTE \(على العقد\):*\n"
            + "\n".join(tp_lines)
            + f"\n   🛑 SL: \-{signal.sl_contract_pct:.0f}% على العقد\n"
            + f"{'─' * 28}\n"
        )
    else:
        # الأهداف الافتراضية
        tp_block = (
            f"🎯 *أهداف 0DTE \(على العقد\):*\n"
            f"   TP1: \+{signal.tp1_pct:.0f}% \| TP2: \+{signal.tp2_pct:.0f}% \| TP3: \+{signal.tp3_pct:.0f}%\n"
            f"   🛑 SL: \-{signal.sl_contract_pct:.0f}%\n"
            f"{'─' * 28}\n"
        )

    # ── أسباب الإشارة ─────────────────────────────
    reasons_lines = ""
    for r in signal.reasons[:8]:
        reasons_lines += f"   • {_escape(str(r))}\n"

    # ── معلومات العقد ─────────────────────────────
    option_block = ""
    if signal.option_symbol:
        option_block = (
            f"\n📋 *العقد المقترح:*\n"
            f"   الرمز: `{_escape(str(signal.option_symbol))}`\n"
            f"   سعر التنفيذ: \\${signal.option_strike:,.0f}\n"
            f"   الانتهاء: {_escape(str(signal.option_expiry))}\n"
        )

    message = (
        f"{dir_emoji} *{signal.direction} — {_escape(signal.symbol)}*  "
        f"{type_emoji} {_escape(signal.signal_type)}\n"
        f"{grade_line}"
        f"{regime_line}"
        f"{'─' * 28}\n"
        f"📍 *دخول:* \\${signal.entry_price:,.2f}\n"
        f"{up_arrow} *هدف سعر الأصل:* \\${signal.target_price:,.2f} \(\+{_escape(str(signal.target_pct))}%\)\n"
        f"{down_arrow} *وقف خسارة:* \\${signal.stop_price:,.2f} \(\-{_escape(str(signal.stop_pct))}%\)\n"
        f"{'─' * 28}\n"
        f"{tp_block}"
        f"{liquidity_block}"
        f"⏳ *الوقت المتوقع:* {signal.time_min}–{signal.time_max} دقيقة\n"
        f"💪 *قوة الإشارة:* {signal.strength}/10  {_escape(bar)}\n"
        f"{mode_emoji} *الوضع:* {_escape(signal.mode)}\n"
        f"🕐 *الوقت:* {_escape(now)}\n"
        f"{'─' * 28}\n"
        f"📊 *أسباب الإشارة:*\n{reasons_lines}"
        f"{option_block}"
        f"{'─' * 28}\n"
        f"⚠️ _للأغراض التعليمية فقط\\. تداول على مسؤوليتك\\._"
    )
    return message


def format_market_summary(market_ctx: dict) -> str:
    """
    Format a market context summary message.
    Sent at market open and every 2 hours.
    """
    bias = market_ctx.get("market_bias", "NEUTRAL")
    spy  = market_ctx.get("spy_trend", {})
    qqq  = market_ctx.get("qqq_trend", {})
    vix  = market_ctx.get("vix_regime", {})

    bias_emoji = BIAS_EMOJI.get(bias, "➡️")
    vix_emoji  = VIX_EMOJI.get(vix.get("level", "NORMAL"), "✅")
    now = datetime.now().strftime("%H:%M EST")

    spy_dir = _escape(spy.get("direction", "?"))
    qqq_dir = _escape(qqq.get("direction", "?"))
    spy_p   = spy.get("price", 0)
    qqq_p   = qqq.get("price", 0)
    vix_val = vix.get("value", 0)
    vix_lvl = _escape(vix.get("level", "?"))
    vix_note = _escape(vix.get("note", ""))
    bias_esc = _escape(bias)

    message = (
        f"📡 *Market Context Update — {_escape(now)}*\n"
        f"{'─' * 30}\n"
        f"{bias_emoji} *Overall Bias:* {bias_esc}\n"
        f"{'─' * 30}\n"
        f"📈 *SPY:* \\${spy_p:,.2f} — {spy_dir}\n"
        f"📈 *QQQ:* \\${qqq_p:,.2f} — {qqq_dir}\n"
        f"{'─' * 30}\n"
        f"{vix_emoji} *Volatility \\(VIX proxy\\):* {vix_val}% — {vix_lvl}\n"
        f"   _{vix_note}_\n"
        f"{'─' * 30}\n"
        f"_Signals will be filtered based on this context_"
    )
    return message


def format_performance_message(perf: dict) -> str:
    """Format a performance stats message."""
    today = perf.get("today", {})
    all_t = perf.get("all_time", {})

    today_signals = today.get("signals", 0)
    today_wins    = today.get("wins", 0)
    today_losses  = today.get("losses", 0)
    win_rate      = all_t.get("win_rate", 0)
    total_pnl     = all_t.get("total_pnl", 0)
    avg_win       = all_t.get("avg_win", 0)
    avg_loss      = all_t.get("avg_loss", 0)

    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
    now = datetime.now().strftime("%Y-%m-%d")

    message = (
        f"📊 *Performance Report — {_escape(now)}*\n"
        f"{'─' * 30}\n"
        f"*Today:*\n"
        f"   Signals: {today_signals}\n"
        f"   Wins: {today_wins} | Losses: {today_losses}\n"
        f"{'─' * 30}\n"
        f"*All Time:*\n"
        f"   Win Rate: {win_rate}%\n"
        f"   Avg Win: \\+{avg_win}% | Avg Loss: \\-{avg_loss}%\n"
        f"   {pnl_emoji} Total P&L: {_escape(str(total_pnl))}%\n"
    )
    return message


def send_telegram_message(text: str, parse_mode: str = "MarkdownV2") -> bool:
    """Send message to all configured chat IDs."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    all_ok = True
    for chat_id in TELEGRAM_CHAT_IDS:
        payload = {
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": parse_mode,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info(f"Telegram OK -> chat_id={chat_id}")
            else:
                logger.error(f"Telegram FAIL -> chat_id={chat_id}: {resp.text[:200]}")
                # Retry with plain text if markdown fails
                if "can't parse" in resp.text.lower():
                    plain_payload = {"chat_id": chat_id, "text": text.replace("\\", ""), "parse_mode": ""}
                    try:
                        requests.post(url, json=plain_payload, timeout=10)
                    except Exception:
                        pass
                all_ok = False
        except Exception as e:
            logger.error(f"Telegram ERROR -> chat_id={chat_id}: {e}")
            all_ok = False
    return all_ok


def send_signal(signal: Signal, custom_message: str = None) -> bool:
    """Send a formatted signal message. If custom_message provided, use it directly."""
    message = custom_message if custom_message else format_signal_message(signal)
    return send_telegram_message(message)


def send_market_context(market_ctx: dict) -> bool:
    """Send market context summary."""
    message = format_market_summary(market_ctx)
    return send_telegram_message(message)


def send_performance_report(perf: dict) -> bool:
    """Send performance stats."""
    message = format_performance_message(perf)
    return send_telegram_message(message)


def send_message(text: str) -> bool:
    """Send a plain message to all chat IDs."""
    return send_telegram_message(text)


def send_startup_message(ai_ready: bool = False) -> bool:
    """Send bot startup notification — v5.0."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    symbols_str = _escape("AAPL \u2022 MSFT \u2022 NVDA \u2022 GOOGL \u2022 AMZN \u2022 META \u2022 TSLA \u2022 QQQ \u2022 SPY")
    ai_status = "3\-Layer AI Active \(XGBoost \+ LSTM \+ RL\)" if ai_ready else "Rule\-Based Mode"
    msg = (
        f"\ud83e\udd16 *Options Scalper Bot v4\.0 \u2014 ONLINE*\n\n"
        f"\ud83e\udde0 *AI Engine:* {ai_status}\n\n"
        f"\ud83d\udcc5 Started: {_escape(now)}\n\n"
        f"\ud83d\udcc8 *Watching:*\n"
        f"   {symbols_str}\n\n"
        f"\ud83d\udd0d *Signal Types:*\n"
        f"   \u2705 Clean Entry\n"
        f"   \ud83d\udd04 Reversal\n"
        f"   \ud83d\udca5 Breakout\n\n"
        f"\ud83c\udfc6 *\u062a\u0635\u0646\u064a\u0641 \u0627\u0644\u0625\u0634\u0627\u0631\u0627\u062a:* A\+ / A / B / C \u0645\u0639 \u0646\u0633\u0628\u0629 \u0627\u0644\u062f\u0642\u0629\n\n"
        f"\ud83d\udca7 *\u0637\u0628\u0642\u0627\u062a v4\.0 \u0627\u0644\u062c\u062f\u064a\u062f\u0629:*\n"
        f"   \ud83d\udcc8 Sideways Detector \\(ADX \+ BB Width \+ ATR\\)\n"
        f"   \ud83c\udfe6 Order Blocks \u0645\u0624\u0633\u0633\u064a\n"
        f"   \ud83d\udcd0 Fair Value Gaps \\(FVG\\)\n"
        f"   \ud83c\udfaf \u0641\u064a\u0628\u0648\u0646\u0627\u062a\u0634\u064a 0\.25 \u0627\u0644\u0630\u0647\u0628\u064a\n"
        f"   \ud83d\udd01 Divergence \u0639\u0627\u062f\u064a \u0648 \u0645\u062e\u0641\u064a\n"
        f"   \ud83c\udfaf TP1 / TP2 / TP3 \u0645\u062a\u062f\u0631\u062c\u0629 \\(15% \u2192 50% \u2192 100%\\)\n"
        f"   \ud83d\uded1 SL \u062b\u0627\u0628\u062a \-30% \u0639\u0644\u0649 \u0627\u0644\u0639\u0642\u062f\n\n"
        f"\ud83e\udde0 *Active Layers:*\n"
        f"   \ud83d\udce1 Market Context \\(SPY/QQQ/VIX\\)\n"
        f"   \ud83d\udcca Technical Analysis \\(EMA/RSI/MACD/BB/VWAP\\)\n"
        f"   \ud83c\udf0a Options Flow \\(PCR/OI/GEX/UOA/MaxPain\\)\n"
        f"   \ud83d\udee1\ufe0f Risk Manager \\(Kelly/DailyLimit/TimeFilter\\)\n\n"
        f"\u23f1\ufe0f *Modes:*\n"
        f"   \u26a1 Scalping: 1\\-5 min\n"
        f"   \ud83d\udcca Day Trading: 15\\-60 min\n\n"
        f"\ud83d\udce1 Feed: Alpaca SIP \\(Real\\-time\\)\n\n"
        f"\u2705 _Bot v4\.0 is running 24/7 on ForexVPS Edge_"
    )
    return send_telegram_message(msg)



def send_error_alert(symbol: str, error: str) -> bool:
    """Send error alert."""
    msg = (
        f"⚠️ *Error Alert*\n\n"
        f"Symbol: {_escape(symbol)}\n"
        f"Error: `{_escape(str(error)[:200])}`"
    )
    return send_telegram_message(msg)


def send_market_closed_reminder() -> bool:
    """Send a reminder that market is closed."""
    msg = (
        f"🌙 *Market Closed*\n\n"
        f"US market is closed\\. Bot is on standby\\.\n"
        f"Will resume at next market open \\(Mon\\-Fri 9:30 AM EST\\)\\."
    )
    return send_telegram_message(msg)
