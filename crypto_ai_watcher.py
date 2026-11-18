"""
Crypto AI Market Watcher (with BingX-style Chart + Entry / SL / TP Lines)
Detects:
  - Double Top / Double Bottom (M頭 / W底)
  - Head & Shoulders / Inverse H&S (頭肩頂 / 頭肩底)
  - Triangle Convergence via Trendline Regression (三角收斂，允許小幅插針誤差)
  - Key S/R Breakout & Liquidity Sweep / Fakeout (關鍵支撐壓力突破 / 假跌破假突破)
Then uses Gemini AI to filter noise, compute Entry / SL / TP, and sends a BingX-style chart to Telegram.
"""

import argparse
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
TIMEFRAMES = os.getenv("TIMEFRAMES", "30m,1h,4h").split(",")
CANDLE_LIMIT = 720
MIN_AI_CONFIDENCE = int(os.getenv("MIN_AI_CONFIDENCE", "6"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# ======================================================


def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 720) -> pd.DataFrame:
    """抓取 K 線資料並排除尚未收盤的最後一根 K 線"""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    if len(df) > 2:
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算輔助指標：EMA20、EMA50、RSI(14)、ATR(14) 與 20 週期均量"""
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


def find_pivots(df: pd.DataFrame, window: int = 4) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """找出波段高點 (Swing High) 與波段低點 (Swing Low)"""
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    for i in range(window, len(df) - window):
        h_slice = df["high"].iloc[i - window : i + window + 1]
        l_slice = df["low"].iloc[i - window : i + window + 1]
        if df["high"].iloc[i] == h_slice.max():
            highs.append((i, float(df["high"].iloc[i])))
        if df["low"].iloc[i] == l_slice.min():
            lows.append((i, float(df["low"].iloc[i])))
    return highs, lows


def detect_double_top_bottom(
    df: pd.DataFrame, highs: list[tuple[int, float]], lows: list[tuple[int, float]]
) -> list[dict[str, Any]]:
    """偵測 M 頭 (Double Top) 與 W 底 (Double Bottom)"""
    signals: list[dict[str, Any]] = []
    last_close = float(df["close"].iloc[-1])
    last_idx = len(df) - 1

    if len(highs) >= 2:
        (i1, p1), (i2, p2) = highs[-2], highs[-1]
        if 6 <= (i2 - i1) <= 60 and (last_idx - i2) <= 15:
            diff_pct = abs(p1 - p2) / max(p1, p2)
            if diff_pct <= 0.012:
                neckline = float(df["low"].iloc[i1:i2 + 1].min())
                if last_close <= neckline * 1.005 and last_close >= neckline * 0.97:
                    status = "已跌破頸線" if last_close < neckline else "正在測試頸線支撐"
                    signals.append({
                        "pattern": "M頭 (Double Top)",
                        "direction": "偏空",
                        "detail": f"雙頂高點 {p1:.2f} / {p2:.2f}，頸線 {neckline:.2f}（{status}）",
                        "key_level": round(neckline, 5),
                    })

    if len(lows) >= 2:
        (i1, p1), (i2, p2) = lows[-2], lows[-1]
        if 6 <= (i2 - i1) <= 60 and (last_idx - i2) <= 15:
            diff_pct = abs(p1 - p2) / max(p1, p2)
            if diff_pct <= 0.012:
                neckline = float(df["high"].iloc[i1:i2 + 1].max())
                if last_close >= neckline * 0.995 and last_close <= neckline * 1.03:
                    status = "已突破頸線" if last_close > neckline else "正在測試頸線壓力"
                    signals.append({
                        "pattern": "W底 (Double Bottom)",
                        "direction": "偏多",
                        "detail": f"雙底低點 {p1:.2f} / {p2:.2f}，頸線 {neckline:.2f}（{status}）",
                        "key_level": round(neckline, 5),
                    })
    return signals


def detect_head_and_shoulders(
    df: pd.DataFrame, highs: list[tuple[int, float]], lows: list[tuple[int, float]]
) -> list[dict[str, Any]]:
    """偵測 頭肩頂 與 頭肩底"""
    signals: list[dict[str, Any]] = []
    last_close = float(df["close"].iloc[-1])
    last_idx = len(df) - 1

    if len(highs) >= 3:
        (i1, ls), (i2, head), (i3, rs) = highs[-3], highs[-2], highs[-1]
        if (last_idx - i3) <= 15 and head > ls * 1.005 and head > rs * 1.005:
            shoulder_diff = abs(ls - rs) / max(ls, rs)
            if shoulder_diff <= 0.02:
                neck1 = float(df["low"].iloc[i1:i2 + 1].min())
                neck2 = float(df["low"].iloc[i2:i3 + 1].min())
                neckline = (neck1 + neck2) / 2
                if last_close <= neckline * 1.01 and last_close >= neckline * 0.97:
                    signals.append({
                        "pattern": "頭肩頂 (Head & Shoulders)",
                        "direction": "偏空",
                        "detail": f"左肩 {ls:.2f} / 頭部 {head:.2f} / 右肩 {rs:.2f}，頸線約 {neckline:.2f}",
                        "key_level": round(neckline, 5),
                    })

    if len(lows) >= 3:
        (i1, ls), (i2, head), (i3, rs) = lows[-3], lows[-2], lows[-1]
        if (last_idx - i3) <= 15 and head < ls * 0.995 and head < rs * 0.995:
            shoulder_diff = abs(ls - rs) / max(ls, rs)
            if shoulder_diff <= 0.02:
                neck1 = float(df["high"].iloc[i1:i2 + 1].max())
                neck2 = float(df["high"].iloc[i2:i3 + 1].max())
                neckline = (neck1 + neck2) / 2
                if last_close >= neckline * 0.99 and last_close <= neckline * 1.03:
                    signals.append({
                        "pattern": "頭肩底 (Inverse H&S)",
                        "direction": "偏多",
                        "detail": f"左肩 {ls:.2f} / 頭部 {head:.2f} / 右肩 {rs:.2f}，頸線約 {neckline:.2f}",
                        "key_level": round(neckline, 5),
                    })
    return signals


def detect_triangle_convergence(
    df: pd.DataFrame, highs: list[tuple[int, float]], lows: list[tuple[int, float]]
) -> list[dict[str, Any]]:
    """使用線性回歸趨勢線偵測三角收斂，允許小幅插針誤差"""
    signals: list[dict[str, Any]] = []
    if len(highs) < 3 or len(lows) < 3:
        return signals

    recent_h = highs[-4:] if len(highs) >= 4 else highs[-3:]
    recent_l = lows[-4:] if len(lows) >= 4 else lows[-3:]
    last_idx = len(df) - 1

    # 確保收斂結構仍在近期，不是太久以前的舊型態
    if (last_idx - max(recent_h[-1][0], recent_l[-1][0])) > 25:
        return signals

    hx = np.array([i for i, _ in recent_h], dtype=float)
    hy = np.array([p for _, p in recent_h], dtype=float)
    lx = np.array([i for i, _ in recent_l], dtype=float)
    ly = np.array([p for _, p in recent_l], dtype=float)

    if hx[-1] <= hx[0] or lx[-1] <= lx[0]:
        return signals

    slope_h, intercept_h = np.polyfit(hx, hy, 1)
    slope_l, intercept_l = np.polyfit(lx, ly, 1)

    start_x = min(hx[0], lx[0])
    end_x = float(last_idx)
    upper_start = slope_h * start_x + intercept_h
    lower_start = slope_l * start_x + intercept_l
    upper_now = slope_h * end_x + intercept_h
    lower_now = slope_l * end_x + intercept_l

    initial_range = upper_start - lower_start
    current_range = upper_now - lower_now
    if initial_range <= 0 or current_range <= 0:
        return signals

    # 允許小幅插針：只要整體振幅收縮至少 25% (current_range / initial_range <= 0.75)
    if (current_range / initial_range) > 0.75:
        return signals

    mid_price = (upper_now + lower_now) / 2
    flat_thresh = mid_price * 0.00015  # 每根 K 線變動小於萬分之 1.5 視為水平線

    falling_h = slope_h < -flat_thresh
    rising_l = slope_l > flat_thresh
    flat_h = abs(slope_h) <= flat_thresh
    flat_l = abs(slope_l) <= flat_thresh

    tri_type = None
    if falling_h and rising_l:
        tri_type = "對稱三角收斂"
    elif flat_h and rising_l:
        tri_type = "上升三角收斂"
    elif falling_h and flat_l:
        tri_type = "下降三角收斂"

    if tri_type:
        last_close = float(df["close"].iloc[-1])
        breakout_note = "處於末端收斂區間"
        direction = "待方向確認"
        key_lvl = upper_now
        if last_close > upper_now * 1.001:
            breakout_note = "剛向上突破上緣趨勢線"
            direction = "偏多"
            key_lvl = upper_now
        elif last_close < lower_now * 0.999:
            breakout_note = "剛向下跌破下緣趨勢線"
            direction = "偏空"
            key_lvl = lower_now

        signals.append({
            "pattern": tri_type,
            "direction": direction,
            "detail": f"上緣約 {upper_now:.2f} / 下緣約 {lower_now:.2f}（{breakout_note}）",
            "key_level": round(float(key_lvl), 5),
        })
    return signals


def detect_sr_breakout_and_sweep(
    df: pd.DataFrame, highs: list[tuple[int, float]], lows: list[tuple[int, float]]
) -> list[dict[str, Any]]:
    """偵測整個回溯區間內的關鍵支撐壓力突破，以及假跌破/假突破獵取"""
    signals: list[dict[str, Any]] = []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    last_idx = len(df) - 1
    vol_ratio = float(last["volume"] / last["vol_ma20"]) if pd.notna(last["vol_ma20"]) and last["vol_ma20"] > 0 else 1.0

    ref_highs = [p for i, p in highs if (last_idx - i) >= 10]
    ref_lows = [p for i, p in lows if (last_idx - i) >= 10]

    if ref_highs:
        major_res = max(ref_highs[-3:]) if len(ref_highs) >= 3 else max(ref_highs)
        if float(last["high"]) > major_res and float(last["close"]) < major_res:
            signals.append({
                "pattern": "假突破獵取 (Liquidity Sweep)",
                "direction": "偏空",
                "detail": f"最高刺破前高壓力 {major_res:.2f} 至 {last['high']:.2f} 後收回 {last['close']:.2f}（量能 {vol_ratio:.1f}x）",
                "key_level": round(major_res, 5),
            })
        elif float(prev["close"]) <= major_res and float(last["close"]) > major_res and vol_ratio >= 1.4:
            signals.append({
                "pattern": "關鍵壓力帶量突破",
                "direction": "偏多",
                "detail": f"放量 ({vol_ratio:.1f}x) 站上關鍵壓力 {major_res:.2f}，收在 {last['close']:.2f}",
                "key_level": round(major_res, 5),
            })

    if ref_lows:
        major_sup = min(ref_lows[-3:]) if len(ref_lows) >= 3 else min(ref_lows)
        if float(last["low"]) < major_sup and float(last["close"]) > major_sup:
            signals.append({
                "pattern": "假跌破獵取 (Liquidity Sweep)",
                "direction": "偏多",
                "detail": f"最低插破前低支撐 {major_sup:.2f} 至 {last['low']:.2f} 後收回 {last['close']:.2f}（量能 {vol_ratio:.1f}x）",
                "key_level": round(major_sup, 5),
            })
        elif float(prev["close"]) >= major_sup and float(last["close"]) < major_sup and vol_ratio >= 1.4:
            signals.append({
                "pattern": "關鍵支撐帶量跌破",
                "direction": "偏空",
                "detail": f"放量 ({vol_ratio:.1f}x) 跌破關鍵支撐 {major_sup:.2f}，收在 {last['close']:.2f}",
                "key_level": round(major_sup, 5),
            })

    return signals


def _fallback_trade_plan(df: pd.DataFrame, candidates: list[dict[str, Any]]) -> dict[str, float]:
    """當未提供 AI 價位時的結構化保底入場／止損／止盈計算"""
    entry = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1]) if "atr" in df.columns and pd.notna(df["atr"].iloc[-1]) else entry * 0.012
    bias = candidates[0].get("direction", "偏多")
    if bias == "偏空":
        sl = float(df["high"].tail(12).max()) + atr * 0.3
        risk = max(sl - entry, atr)
        sl = entry + risk
        tp = entry - risk * 2.0
    else:
        sl = float(df["low"].tail(12).min()) - atr * 0.3
        risk = max(entry - sl, atr)
        sl = entry - risk
        tp = entry + risk * 2.0
    return {"entry": entry, "stop_loss": sl, "take_profit": tp}


def verify_with_gemini(
    symbol: str, timeframe: str, df: pd.DataFrame, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """呼叫 Gemini API 進行結構複判，並輸出建議入場、止損與止盈價位"""
    fallback = _fallback_trade_plan(df, candidates)
    if not GEMINI_API_KEY:
        return {
            "is_valid": True,
            "confidence": 7,
            "bias": candidates[0]["direction"],
            "entry": fallback["entry"],
            "stop_loss": fallback["stop_loss"],
            "take_profit": fallback["take_profit"],
            "summary": "（未設定 GEMINI_API_KEY）" + "；".join(c["detail"] for c in candidates),
        }

    recent_candles = df.tail(25)[["dt", "open", "high", "low", "close", "volume", "rsi"]].copy()
    recent_candles["dt"] = recent_candles["dt"].dt.strftime("%m-%d %H:%M")
    candles_text = recent_candles.round(4).to_string(index=False)

    prompt = f"""你是專業的加密貨幣技術分析師。演算法在 {symbol} ({timeframe} 週期) 偵測到以下候選型態/結構：
{json.dumps(candidates, ensure_ascii=False, indent=2)}

以下為最近 25 根已收盤 K 線數據 (UTC 時間)：
{candles_text}

請嚴格評估這些型態或流動性結構是否具備交易參考價值（過濾掉震盪雜訊），並根據結構高低點與頸線給出具體的入場點、止損點與止盈點（風險報酬比建議至少 1:2），只回傳純 JSON（不要加 markdown 標記），格式如下：
{{
  "is_valid": true 或 false,
  "confidence": 1 到 10 的整數,
  "bias": "偏多" 或 "偏空" 或 "待確認",
  "entry": 數字（建議入場或回測接單價位）,
  "stop_loss": 數字（結構失效止損價位）,
  "take_profit": 數字（目標止盈價位）,
  "summary": "用 2 句繁體中文說明結構重點與操作邏輯"
}}"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    result = json.loads(text)
    for k in ("entry", "stop_loss", "take_profit"):
        if not result.get(k):
            result[k] = fallback[k]
    return result


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
    key_levels: list[float],
    entry: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> bytes:
    """繪製仿 BingX App 風格的高清晰度 K 線圖（含入場／止盈／止損線與右側標籤框）"""
    if "ema20" not in df.columns:
        df = add_indicators(df.copy())

    plot_df = df.tail(110).copy().reset_index(drop=True)
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

    unique_levels = sorted({round(lvl, 5) for lvl in key_levels if lvl > 0})
    last_price = float(ohlcv_df["Close"].iloc[-1])

    all_y = [float(plot_df["low"].min()), float(plot_df["high"].max())]
    for v in (entry, stop_loss, take_profit, *unique_levels):
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

    for lvl in unique_levels:
        ax_main.axhline(lvl, color="#f0b90b", linestyle="--", linewidth=1.0, alpha=0.85)
        ax_main.text(
            0.5,
            lvl,
            f" Key {_fmt_price(lvl)} ",
            color="#f0b90b",
            fontsize=8.5,
            va="bottom",
            ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#1e1b10", edgecolor="#f0b90b", linewidth=0.9),
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
    """避免同一根 K 線重複推播"""
    last_open_utc = df["dt"].iloc[-1].to_pydatetime()
    tf_minutes = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}.get(timeframe, 30)
    last_close_utc = last_open_utc + timedelta(minutes=tf_minutes)
    age_minutes = (datetime.now(timezone.utc) - last_close_utc).total_seconds() / 60
    return age_minutes <= 65


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
                highs, lows = find_pivots(df, window=4)

                candidates: list[dict[str, Any]] = []
                candidates.extend(detect_double_top_bottom(df, highs, lows))
                candidates.extend(detect_head_and_shoulders(df, highs, lows))
                candidates.extend(detect_triangle_convergence(df, highs, lows))
                candidates.extend(detect_sr_breakout_and_sweep(df, highs, lows))

                if not candidates:
                    print(f"[無訊號] {symbol} {tf} 目前無符合的結構或型態")
                    continue

                print(f"[初篩命中] {symbol} {tf}: {[c['pattern'] for c in candidates]}")
                ai_result = verify_with_gemini(symbol, tf, df, candidates)

                if ai_result.get("is_valid") and int(ai_result.get("confidence", 0)) >= MIN_AI_CONFIDENCE:
                    last_price = float(df["close"].iloc[-1])
                    candle_tw = df["dt"].iloc[-1].astimezone(tz_tw).strftime("%Y-%m-%d %H:%M")
                    patterns_str = "、".join(c["pattern"] for c in candidates)
                    key_levels = [float(c["key_level"]) for c in candidates if "key_level" in c]

                    entry_p = float(ai_result["entry"])
                    sl_p = float(ai_result["stop_loss"])
                    tp_p = float(ai_result["take_profit"])
                    risk = abs(entry_p - sl_p)
                    rr = abs(tp_p - entry_p) / risk if risk > 0 else 0

                    chart_bytes = render_chart_png(
                        df, symbol, tf, key_levels,
                        entry=entry_p, stop_loss=sl_p, take_profit=tp_p,
                    )
                    msg = (
                        f"🔔 <b>AI 盯盤警報｜{symbol} ({tf})</b>\n"
                        f"• <b>偵測結構</b>：{patterns_str}\n"
                        f"• <b>AI 傾向</b>：{ai_result.get('bias')}（信心 {ai_result.get('confidence')}/10）\n"
                        f"• <b>現價</b>：{_fmt_price(last_price)}（{candle_tw} UTC+8）\n"
                        f"• <b>參考規劃</b>：入場 <code>{_fmt_price(entry_p)}</code> ｜ 止損 <code>{_fmt_price(sl_p)}</code> ｜ 止盈 <code>{_fmt_price(tp_p)}</code>（R:R ≈ 1:{rr:.1f}）\n"
                        f"• <b>AI 分析</b>：{ai_result.get('summary')}"
                    )
                    send_telegram_photo(chart_bytes, msg)
                else:
                    print(f"[AI 過濾] {symbol} {tf} 信心不足 ({ai_result.get('confidence')}/10)")

            except Exception as exc:
                print(f"[錯誤] {symbol} {tf} 處理失敗: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto AI Market Watcher")
    parser.add_argument("--test", action="store_true", help="忽略時間窗口限制，強制掃描一次")
    args = parser.parse_args()
    run(ignore_time_window=args.test)
