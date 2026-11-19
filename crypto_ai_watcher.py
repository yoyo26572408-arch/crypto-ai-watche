"""
Crypto AI Vision Market Watcher（兩段式 AI + Telegram 互動體檢報告 + 抗限流版）
流程：
  1. 檢查 Telegram 是否有你的新提問（例如傳「狀態」「有沒有出錯」），若有則標記本輪結束後回報完整體檢清單
  2. 從交易所 (BingX 優先 + Gate 備援) 抓取 30m / 1h / 4h / 1d K 線並繪製圖表
  3. 【AI 第一關】看圖辨識：抓出清晰型態與趨勢折線，過濾純震盪雜訊
  4. 【AI 第二關】風控評分：評估風險與量價，通過門檻即畫上趨勢線與入場/止損/止盈推播
  5. 若你有發問或本輪出現系統錯誤，自動回傳各幣種狀態報告到 Telegram
"""

import argparse
import base64
import io
import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any

import ccxt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests


# ==================== 使用者設定區 ====================
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "bingx")
FALLBACK_EXCHANGE_ID = os.getenv("FALLBACK_EXCHANGE_ID", "gate")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,SUI/USDT:USDT,ZEC/USDT:USDT").split(",")
TIMEFRAMES = os.getenv("TIMEFRAMES", "30m,1h,4h,1d").split(",")
CANDLE_LIMIT = 200
MIN_AI_CONFIDENCE = int(os.getenv("MIN_AI_CONFIDENCE", "6"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# ======================================================


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = 200,
    fallback_exchange: ccxt.Exchange | None = None,
) -> pd.DataFrame:
    """抓取 K 線資料，遇限流自動重試 3 次，仍失敗則自動切換備援交易所"""
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            if len(df) > 2:
                df = df.iloc[:-1].reset_index(drop=True)
            return df
        except Exception as exc:
            last_err = exc
            print(f"[重試 {attempt}/3] {exchange.id} 抓取 {symbol} {timeframe} 失敗: {exc}")
            time.sleep(2 * attempt)

    if fallback_exchange is not None:
        print(f"[切換備援] 改用 {fallback_exchange.id} 抓取 {symbol} {timeframe}")
        ohlcv = fallback_exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        if len(df) > 2:
            df = df.iloc[:-1].reset_index(drop=True)
        return df

    raise RuntimeError(f"無法取得 {symbol} {timeframe} 資料: {last_err}")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算 EMA20、EMA50、RSI(14)、ATR(14) 與 20 週期均量"""
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
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


def _extract_chart_pivots(plot_df: pd.DataFrame, window: int = 3) -> list[dict[str, Any]]:
    """擷取最近 85 根 K 線內的轉折高低點索引與價位，供 AI 畫出趨勢折線"""
    pivots: list[dict[str, Any]] = []
    for i in range(window, len(plot_df) - window):
        h_slice = plot_df["high"].iloc[i - window : i + window + 1]
        l_slice = plot_df["low"].iloc[i - window : i + window + 1]
        if plot_df["high"].iloc[i] == h_slice.max():
            pivots.append({"bar": int(i), "type": "high", "price": round(float(plot_df["high"].iloc[i]), 5)})
        if plot_df["low"].iloc[i] == l_slice.min():
            pivots.append({"bar": int(i), "type": "low", "price": round(float(plot_df["low"].iloc[i]), 5)})
    return pivots[-16:]


def _call_gemini(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """統一呼叫 Gemini API 並解析 JSON 回傳"""
    if not GEMINI_API_KEY:
        raise RuntimeError("請設定 GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    resp = requests.post(url, json=payload, timeout=45)
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def render_chart_png(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    entry: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trend_lines: list[list[list[float]]] | None = None,
) -> bytes:
    """繪製仿 BingX / TradingView 風格 K 線圖（含趨勢折線、目標預測箭頭與入場/止損/止盈線）"""
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

    if trend_lines:
        for line in trend_lines:
            if not isinstance(line, list) or len(line) < 2:
                continue
            xs, ys = [], []
            for pt in line:
                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                    bx = max(0.0, min(float(n - 1), float(pt[0])))
                    py = float(pt[1])
                    if py > 0:
                        xs.append(bx)
                        ys.append(py)
            if len(xs) >= 2:
                ax_main.plot(xs, ys, color="#b2b5be", linewidth=2.0, alpha=0.9, zorder=5)

    if take_profit and take_profit > 0:
        start_y = float(entry) if entry and entry > 0 else last_price
        arrow_color = "#0ecb81" if float(take_profit) >= start_y else "#f6465d"
        ax_main.annotate(
            "",
            xy=(n + 7, float(take_profit)),
            xytext=(n - 1, start_y),
            arrowprops=dict(arrowstyle="->,head_width=0.45,head_length=0.6", color=arrow_color, lw=2.0, alpha=0.95),
            zorder=6,
        )

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


def stage1_detect_pattern_from_chart(
    symbol: str, timeframe: str, df: pd.DataFrame, raw_chart_bytes: bytes
) -> dict[str, Any]:
    """【AI 第一關】看圖辨識：抓出圖上的清晰型態與趨勢折線，過濾純震盪雜訊"""
    plot_df = df.tail(85).copy().reset_index(drop=True)
    pivots = _extract_chart_pivots(plot_df, window=3)
    last_price = float(plot_df["close"].iloc[-1])
    img_b64 = base64.b64encode(raw_chart_bytes).decode("utf-8")

    prompt = f"""你是專業的圖表型態辨識師。請觀察這張 {symbol} ({timeframe} 週期、共 85 根 K 線，最左 bar 0、最右最新收盤 bar 84) 的 K 線圖。
目前收盤價 {_fmt_price(last_price)}，主要轉折點參考：
{json.dumps(pivots, ensure_ascii=False)}

請檢查圖中「最右側（最新走勢）」是否出現以下任何型態或結構：
- 三角收斂 / 末端緊密壓縮 / 旗形 / 楔形（包含近 8-15 根 K 線的小型收斂）
- M 頭 / W 底 / 頭肩頂 / 頭肩底
- 通道上下緣測試 / 關鍵支撐壓力突破 / 長影線假跌破或假突破

只有在圖表完全無序震盪時才設 has_pattern = false。若有型態，請挑出 `trend_lines` 座標點畫出結構線（M/W 頭肩回傳 1 條折線，三角或通道回傳 2 條邊緣線）。只回傳純 JSON：
{{
  "has_pattern": true 或 false,
  "pattern_name": "型態名稱",
  "bias": "偏多" 或 "偏空" 或 "待突破確認",
  "trend_lines": [[[bar, price], [bar, price], ...]],
  "pattern_note": "簡短描述型態所在位置與高低點"
}}"""

    return _call_gemini([
        {"text": prompt},
        {"inlineData": {"mimeType": "image/png", "data": img_b64}},
    ])


def stage2_score_risk_and_plan(
    symbol: str, timeframe: str, df: pd.DataFrame, raw_chart_bytes: bytes, stage1_result: dict[str, Any]
) -> dict[str, Any]:
    """【AI 第二關】風控評分：評估結構風險、量價配合與風報比，決定是否放行並輸出入場/止損/止盈"""
    recent = df.tail(25)[["dt", "open", "high", "low", "close", "volume", "ema20", "ema50", "rsi", "atr"]].copy()
    recent["dt"] = recent["dt"].dt.strftime("%m-%d %H:%M")
    candles_text = recent.round(4).to_string(index=False)
    img_b64 = base64.b64encode(raw_chart_bytes).decode("utf-8")

    prompt = f"""你是嚴謹的加密貨幣交易風控官。第一關看圖辨識在 {symbol} ({timeframe}) 發現了型態：
{json.dumps(stage1_result, ensure_ascii=False)}

以下為最近 25 根 K 線與均線/RSI/ATR 數據：
{candles_text}

請同時對照 K 線圖與量價數據進行風險評估：
1. 評估是否有逆勢撞均線、量能背離、止損過遠或假訊號風險，給出 `risk_score`（1 到 10 的整數，分數越高代表風險越低、交易品質越好）。
2. 給出具體的 `entry`（入場點）、`stop_loss`（止損點）、`take_profit`（止盈點，風報比至少 1:1.8）。

只回傳純 JSON：
{{
  "risk_score": 1 到 10 的整數,
  "risk_level": "低風險" 或 "中風險" 或 "高風險",
  "entry": 數字,
  "stop_loss": 數字,
  "take_profit": 數字,
  "summary": "用 2 句繁體中文說明型態重點、風險評估理由與操作建議"
}}"""

    return _call_gemini([
        {"text": prompt},
        {"inlineData": {"mimeType": "image/png", "data": img_b64}},
    ])


def check_telegram_user_query() -> str | None:
    """檢查你在 Telegram 是否傳了新訊息詢問狀態，並自動標記已讀避免重複回覆"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return None
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        resp = requests.get(url, params={"timeout": 2}, timeout=10)
        resp.raise_for_status()
        updates = resp.json().get("result", [])
        if not updates:
            return None

        max_id = max(u["update_id"] for u in updates)
        user_msg = None
        for u in updates:
            msg = u.get("message") or {}
            if str(msg.get("chat", {}).get("id", "")) == str(TELEGRAM_CHAT_ID) and msg.get("text"):
                user_msg = msg["text"].strip()

        # 標記這批訊息已讀，避免下次重複回覆
        requests.get(url, params={"offset": max_id + 1, "timeout": 1}, timeout=10)
        return user_msg
    except Exception as exc:
        print(f"[Telegram 收訊檢查失敗] {exc}")
        return None


def send_telegram_text(text: str) -> None:
    """透過 Telegram Bot 發送純文字狀態報告"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[模擬文字推播]\n" + text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text[:4096], "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=15,
    )
    resp.raise_for_status()


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
    user_query = check_telegram_user_query()
    if user_query:
        print(f"[收到 Telegram 提問] {user_query}")

    exchange_cls = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_cls({"enableRateLimit": True})
    fallback_cls = getattr(ccxt, FALLBACK_EXCHANGE_ID, None)
    fallback_exchange = fallback_cls({"enableRateLimit": True}) if fallback_cls else None
    tz_tw = timezone(timedelta(hours=8))

    status_lines: list[str] = []
    error_lines: list[str] = []

    for raw_symbol in SYMBOLS:
        symbol = raw_symbol.strip()
        if not symbol:
            continue
        short_sym = symbol.split("/")[0]
        for raw_tf in TIMEFRAMES:
            tf = raw_tf.strip()
            if not tf:
                continue
            try:
                time.sleep(0.4)
                df = fetch_ohlcv(exchange, symbol, tf, limit=CANDLE_LIMIT, fallback_exchange=fallback_exchange)
                # 若使用者主動在 Telegram 詢問狀態，強制掃描所有週期以便回報最新情況
                if not ignore_time_window and not user_query and not is_recently_closed_for_tf(df, tf):
                    continue

                df = add_indicators(df)
                raw_chart = render_chart_png(df, symbol, tf)

                s1 = stage1_detect_pattern_from_chart(symbol, tf, df, raw_chart)
                if not s1.get("has_pattern"):
                    msg_line = f"• <b>{short_sym} ({tf})</b>：無明確型態"
                    status_lines.append(msg_line)
                    print(f"[第一關過濾] {symbol} {tf} 無明確型態")
                    continue

                print(f"[第一關命中] {symbol} {tf}: {s1.get('pattern_name')} ({s1.get('bias')})")
                s2 = stage2_score_risk_and_plan(symbol, tf, df, raw_chart, s1)
                risk_score = int(s2.get("risk_score", 0))

                if risk_score >= MIN_AI_CONFIDENCE:
                    last_price = float(df["close"].iloc[-1])
                    candle_tw = df["dt"].iloc[-1].astimezone(tz_tw).strftime("%Y-%m-%d %H:%M")

                    entry_p = float(s2.get("entry") or last_price)
                    sl_p = float(s2.get("stop_loss") or last_price * 0.98)
                    tp_p = float(s2.get("take_profit") or last_price * 1.04)
                    risk = abs(entry_p - sl_p)
                    rr = abs(tp_p - entry_p) / risk if risk > 0 else 0

                    marked_chart = render_chart_png(
                        df, symbol, tf,
                        entry=entry_p, stop_loss=sl_p, take_profit=tp_p,
                        trend_lines=s1.get("trend_lines"),
                    )
                    msg = (
                        f"🔔 <b>AI 盯盤警報｜{symbol} ({tf})</b>\n"
                        f"• <b>型態結構</b>：{s1.get('pattern_name')}（{s1.get('bias')}）\n"
                        f"• <b>風險評分</b>：{risk_score}/10（{s2.get('risk_level')}）\n"
                        f"• <b>現價</b>：{_fmt_price(last_price)}（{candle_tw} UTC+8）\n"
                        f"• <b>參考規劃</b>：入場 <code>{_fmt_price(entry_p)}</code> ｜ 止損 <code>{_fmt_price(sl_p)}</code> ｜ 止盈 <code>{_fmt_price(tp_p)}</code>（R:R ≈ 1:{rr:.1f}）\n"
                        f"• <b>AI 分析</b>：{s2.get('summary')}"
                    )
                    send_telegram_photo(marked_chart, msg)
                    status_lines.append(f"• <b>{short_sym} ({tf})</b>：✅ 已發圖（{s1.get('pattern_name')}，{risk_score}/10）")
                    print(f"[第二關通過已推播] {symbol} {tf}: 評分 {risk_score}/10")
                else:
                    status_lines.append(
                        f"• <b>{short_sym} ({tf})</b>：🟡 有「{s1.get('pattern_name')}」但風險評分 {risk_score}/10（未達 {MIN_AI_CONFIDENCE} 分）"
                    )
                    print(f"[第二關風險過濾] {symbol} {tf} {s1.get('pattern_name')} 評分僅 {risk_score}/10")

            except Exception as exc:
                err_msg = f"• <b>{short_sym} ({tf})</b>：❌ 錯誤（{str(exc)[:80]}）"
                status_lines.append(err_msg)
                error_lines.append(err_msg)
                print(f"[錯誤] {symbol} {tf} 處理失敗: {exc}")

    # 若你在 Telegram 傳訊詢問，或本輪有發生系統錯誤，主動回傳一份體檢總結
    if user_query or error_lines:
        now_tw = datetime.now(tz_tw).strftime("%m-%d %H:%M")
        header = (
            f"📋 <b>盯盤系統體檢回報（{now_tw} UTC+8）</b>\n"
            f"目前門檻：風險評分 ≥ <b>{MIN_AI_CONFIDENCE}/10</b>\n"
        )
        if user_query:
            header += f"收到你的訊息：「<i>{user_query[:30]}</i>」\n\n"
        else:
            header += "⚠️ <b>偵測到部分項目異常：</b>\n\n"

        body = "\n".join(status_lines) if status_lines else "本輪無剛收盤的週期。"
        send_telegram_text(header + body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto AI Two-Stage Vision Watcher")
    parser.add_argument("--test", action="store_true", help="忽略時間窗口限制，強制掃描一次")
    args = parser.parse_args()
    run(ignore_time_window=args.test)
