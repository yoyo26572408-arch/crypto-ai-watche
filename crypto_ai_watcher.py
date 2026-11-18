"""
Crypto AI Vision Market Watcher (Gemini 2.5 Flash 看圖盯盤版)
流程：
  1. 每 30 分鐘由 cron-job.org 準時觸發，抓取 30m / 1h / 4h / 1d 剛收盤的 K 線
  2. 直接繪製高清晰度 BingX 風格 K 線圖（含均線與高低點）
  3. 將 K 線圖傳給 Gemini Vision 直接「看圖辨識」三角收斂、旗形壓縮、M頭W底、頭肩、突破與假跌破
  4. 若有明確型態，Gemini 直接給出入場 / 止損 / 止盈，畫上三條線後推播至 Telegram
"""

import argparse
import base64
import io
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any

import ccxt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests


# ==================== 使用者設定區 ====================
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "bingx")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,SUI/USDT:USDT,ZEC/USDT:USDT").split(",")
TIMEFRAMES = os.getenv("TIMEFRAMES", "30m,1h,4h,1d").split(",")
CANDLE_LIMIT = 200
MIN_AI_CONFIDENCE = int(os.getenv("MIN_AI_CONFIDENCE", "7"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# ======================================================


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    """抓取 K 線資料並排除尚未收盤的最後一根 K 線"""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    if len(df) > 2:
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算 EMA20、EMA50 與 ATR(14)"""
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


def _fmt_price(val: float) -> str:
    """依幣種價位自動決定小數位數"""
    if val >= 1000:
        return f"{val:,.1f}"
    if val >= 10:
        return f"{val:.2f}"
    if val >= 1:
        return f"{val:.3f}"
    return f"{val:.5f}"


def render_chart_png(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    entry: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> bytes:
    """繪製仿 BingX App 風格的高清晰度 K 線圖（支援純看圖模式與加上入場/止損/止盈線）"""
    if "ema20" not in df.columns:
        df = add_indicators(df.copy())

    plot_df = df.tail(85).copy().reset_index(drop=True)
    plot_df["dt_tw"] = plot_df["dt"] + pd.Timedelta(hours=8)
    plot_df_idx = plot_df.set_index("dt_tw")

    ohlcv_df = plot_df_idx[["open", "high", "low", "close", "volume"]].copy()
    ohlcv_df.columns = ["Open", "High", "Low", "Close", "Volume"]

    mc = mpf.make_marketcolors(
        up="#0ecb81",
        down="#f6465d",
        edge="inherit",
        wick="inherit",
        volume={"up": "#0ecb81", "down": "#f6465d"},
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor="#0b0e11",
        figcolor="#0b0e11",
        gridcolor="#1a1e26",
        gridstyle="-",
        y_on_right=True,
        rc={
            "axes.edgecolor": "#262b36",
            "xtick.color": "#848e9c",
            "ytick.color": "#848e9c",
            "axes.labelcolor": "#848e9c",
            "text.color": "#eaecef",
        },
    )

    addplots = [
        mpf.make_addplot(plot_df_idx["ema20"], color="#29b6f6", width=0.9, panel=0),
        mpf.make_addplot(plot_df_idx["ema50"], color="#ffa726", width=0.9, panel=0),
    ]

    last_price = float(ohlcv_df["Close"].iloc[-1])
    all_y = [float(plot_df["low"].min()), float(plot_df["high"].max())]
    for v in (entry, stop_loss, take_profit):
        if v is not None and v > 0:
            all_y.append(float(v))
    y_min, y_max = min(all_y), max(all_y)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else y_max * 0.02

    buf = io.BytesIO()
    fig, axes = mpf.plot(
        ohlcv_df,
        type="candle",
        volume=True,
        panel_ratios=(5, 1.2),
        addplot=addplots,
        style=style,
        title=f"{symbol}  ·  {timeframe}  (UTC+8)",
        figsize=(12.5, 7),
        xlim=(-1, len(ohlcv_df) + 18),
        ylim=(y_min - y_pad, y_max + y_pad),
        returnfig=True,
        tight_layout=True,
    )

    ax_main = axes[0]
    n = len(ohlcv_df)

    # 標註最高價與最低價
    hi_idx = int(plot_df["high"].idxmax())
    hi_val = float(plot_df["high"].iloc[hi_idx])
    lo_idx = int(plot_df["low"].idxmin())
    lo_val = float(plot_df["low"].iloc[lo_idx])

    hi_offset = 6 if hi_idx < n - 12 else -14
    ax_main.plot([hi_idx, hi_idx + hi_offset], [hi_val, hi_val], color="#eaecef", linewidth=0.9)
    ax_main.text(
        hi_idx + hi_offset + (0.6 if hi_offset > 0 else -0.6),
        hi_val,
        _fmt_price(hi_val),
        color="#eaecef",
        fontsize=9,
        va="center",
        ha="left" if hi_offset > 0 else "right",
        fontweight="bold",
    )

    lo_offset = 6 if lo_idx < n - 12 else -14
    ax_main.plot([lo_idx, lo_idx + lo_offset], [lo_val, lo_val], color="#eaecef", linewidth=0.9)
    ax_main.text(
        lo_idx + lo_offset + (0.6 if lo_offset > 0 else -0.6),
        lo_val,
        _fmt_price(lo_val),
        color="#eaecef",
        fontsize=9,
        va="center",
        ha="left" if lo_offset > 0 else "right",
        fontweight="bold",
    )

    right_x = n + 1.5
    lines_to_draw: list[tuple[str, float, str, str, str]] = []
    if take_profit:
        lines_to_draw.append(("TP", float(take_profit), "#0ecb81", "#0c261d", "--"))
    if entry:
        lines_to_draw.append(("Entry", float(entry), "#eaecef", "#1e2329", "-."))
    if stop_loss:
        lines_to_draw.append(("SL", float(stop_loss), "#f6465d", "#2c1117", "--"))

    for label, price_val, edge_col, bg_col, ls in lines_to_draw:
        ax_main.axhline(price_val, color=edge_col, linestyle=ls, linewidth=1.2, alpha=0.95)
        ax_main.text(
            right_x,
            price_val,
            f" {label} {_fmt_price(price_val)} ",
            color=edge_col,
            fontsize=9,
            va="center",
            ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=bg_col, edgecolor=edge_col, linewidth=1),
        )

    if not entry:
        ax_main.plot([n - 1, n + 1.2], [last_price, last_price], color="#eaecef", linestyle=":", linewidth=1.0)
        ax_main.text(
            right_x,
            last_price,
            f" {_fmt_price(last_price)} ",
            color="#eaecef",
            fontsize=9,
            va="center",
            ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#1e2329", edgecolor="#eaecef", linewidth=1),
        )

    fig.savefig(buf, dpi=220, bbox_inches="tight", facecolor="#0b0e11")
    buf.seek(0)
    return buf.read()


def analyze_chart_with_gemini_vision(
    symbol: str, timeframe: str, df: pd.DataFrame, raw_chart_bytes: bytes
) -> dict[str, Any]:
    """直接讓 Gemini Vision 看 K 線圖，辨識三角收斂、旗形、M頭/W底、頭肩、突破與假跌破"""
    if not GEMINI_API_KEY:
        raise RuntimeError("請設定 GEMINI_API_KEY 才能使用 AI 看圖盯盤")

    last_price = float(df["close"].iloc[-1])
    recent_high = float(df["high"].tail(20).max())
    recent_low = float(df["low"].tail(20).min())
    img_b64 = base64.b64encode(raw_chart_bytes).decode("utf-8")

    prompt = f"""你是專業的加密貨幣技術分析師。請直接觀察這張 {symbol} ({timeframe} 週期) 的最新 K 線圖。
目前最新收盤價為 {_fmt_price(last_price)}，近 20 根最高 {_fmt_price(recent_high)}、最低 {_fmt_price(recent_low)}。

請檢查圖中「最右側（最新一段走勢）」是否正在形成或剛觸發以下任何清晰型態或結構：
1. 三角收斂 / 末端緊密壓縮收斂（包含對稱三角、上升/下降三角、旗形收斂，哪怕只有 10-15 根 K 線的小收斂也算）
2. M 頭 / W 底 / 頭肩頂 / 頭肩底
3. 關鍵支撐壓力帶量突破，或假跌破 / 假突破獵取 (Liquidity Sweep)

如果沒有明確型態（只是無序震盪），請將 has_pattern 設為 false。
如果有明確型態，請給出具體的入場點、止損點與止盈點（風險報酬比建議至少 1:1.8），只回傳純 JSON：
{{
  "has_pattern": true 或 false,
  "pattern_name": "型態名稱（例如：末端三角收斂、W底突破、假跌破反彈）",
  "confidence": 1 到 10 的整數,
  "bias": "偏多" 或 "偏空" 或 "待突破方向確認",
  "entry": 數字（建議入場或突破掛單價位）,
  "stop_loss": 數字（結構失效止損價位）,
  "take_profit": 數字（目標止盈價位）,
  "summary": "用 2 句繁體中文說明你從圖上看到什麼結構，以及進出場觀察重點"
}}"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/png", "data": img_b64}},
            ]
        }],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    resp = requests.post(url, json=payload, timeout=40)
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def send_telegram_photo(image_bytes: bytes, caption: str) -> None:
    """透過 Telegram Bot 發送 K 線圖與分析說明"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[模擬推播] 未設定 Telegram Token/Chat ID，訊息內容如下：\n" + caption)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    resp = requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024], "parse_mode": "HTML"},
        files={"photo": ("chart.png", image_bytes, "image/png")},
        timeout=20,
    )
    resp.raise_for_status()


def is_recently_closed_for_tf(df: pd.DataFrame, timeframe: str) -> bool:
    """配合每 30 分鐘準時觸發，確保各週期只在剛收盤時檢查一次、不重複推播"""
    last_open_utc = df["dt"].iloc[-1].to_pydatetime()
    tf_minutes = {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}.get(timeframe, 30)
    last_close_utc = last_open_utc + timedelta(minutes=tf_minutes)
    age_minutes = (datetime.now(timezone.utc) - last_close_utc).total_seconds() / 60
    return age_minutes <= 35


def run(ignore_time_window: bool = False) -> None:
    exchange_cls = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_cls({"enableRateLimit": True})
    tz_tw = timezone(timedelta(hours=8))

    for raw_symbol in SYMBOLS:
        symbol = raw_symbol.strip()
        if not symbol:
            continue
        for raw_tf in TIMEFRAMES:
            tf = raw_tf.strip()
            if not tf:
                continue
            try:
                df = fetch_ohlcv(exchange, symbol, tf, limit=CANDLE_LIMIT)
                if not ignore_time_window and not is_recently_closed_for_tf(df, tf):
                    print(f"[跳過] {symbol} {tf} 非剛收盤的 K 線，避免重複通知")
                    continue

                df = add_indicators(df)
                raw_chart = render_chart_png(df, symbol, tf)
                ai_result = analyze_chart_with_gemini_vision(symbol, tf, df, raw_chart)

                if ai_result.get("has_pattern") and int(ai_result.get("confidence", 0)) >= MIN_AI_CONFIDENCE:
                    last_price = float(df["close"].iloc[-1])
                    candle_tw = df["dt"].iloc[-1].astimezone(tz_tw).strftime("%Y-%m-%d %H:%M")

                    entry_p = float(ai_result.get("entry") or last_price)
                    sl_p = float(ai_result.get("stop_loss") or last_price * 0.98)
                    tp_p = float(ai_result.get("take_profit") or last_price * 1.04)
                    risk = abs(entry_p - sl_p)
                    rr = abs(tp_p - entry_p) / risk if risk > 0 else 0

                    marked_chart = render_chart_png(
                        df, symbol, tf,
                        entry=entry_p, stop_loss=sl_p, take_profit=tp_p,
                    )
                    msg = (
                        f"🔔 <b>AI 看圖警報｜{symbol} ({tf})</b>\n"
                        f"• <b>型態結構</b>：{ai_result.get('pattern_name')}\n"
                        f"• <b>AI 傾向</b>：{ai_result.get('bias')}（信心 {ai_result.get('confidence')}/10）\n"
                        f"• <b>現價</b>：{_fmt_price(last_price)}（{candle_tw} UTC+8）\n"
                        f"• <b>參考規劃</b>：入場 <code>{_fmt_price(entry_p)}</code> ｜ 止損 <code>{_fmt_price(sl_p)}</code> ｜ 止盈 <code>{_fmt_price(tp_p)}</code>（R:R ≈ 1:{rr:.1f}）\n"
                        f"• <b>AI 分析</b>：{ai_result.get('summary')}"
                    )
                    send_telegram_photo(marked_chart, msg)
                    print(f"[已推播] {symbol} {tf}: {ai_result.get('pattern_name')}")
                else:
                    print(f"[無明確型態] {symbol} {tf} (has_pattern={ai_result.get('has_pattern')}, conf={ai_result.get('confidence')})")

            except Exception as exc:
                print(f"[錯誤] {symbol} {tf} 處理失敗: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto AI Vision Market Watcher")
    parser.add_argument("--test", action="store_true", help="忽略時間窗口限制，強制掃描一次")
    args = parser.parse_args()
    run(ignore_time_window=args.test)
