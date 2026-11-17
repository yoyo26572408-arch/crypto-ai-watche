"""
Crypto AI Market Watcher (with BingX-style Candlestick Chart Alert)
Detects:
  - Double Top / Double Bottom (M頭 / W底)
  - Head & Shoulders / Inverse H&S (頭肩頂 / 頭肩底)
  - Triangle Convergence (三角收斂)
  - Key S/R Breakout & Liquidity Sweep / Fakeout (關鍵支撐壓力突破 / 假跌破假突破)
Then uses Gemini AI to filter noise and sends a crisp BingX-style candlestick chart + analysis to Telegram.
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
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT").split(",")
TIMEFRAMES = os.getenv("TIMEFRAMES", "1h,4h").split(",")
CANDLE_LIMIT = 720
MIN_AI_CONFIDENCE = int(os.getenv("MIN_AI_CONFIDENCE", "7"))

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
    """計算輔助指標：EMA20、EMA50、RSI(14) 與 20 週期均量"""
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
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
                        "key_level": round(neckline, 4),
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
                        "key_level": round(neckline, 4),
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
                        "key_level": round(neckline, 4),
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
                        "key_level": round(neckline, 4),
                    })
    return signals


def detect_triangle_convergence(
    df: pd.DataFrame, highs: list[tuple[int, float]], lows: list[tuple[int, float]]
) -> list[dict[str, Any]]:
    """偵測三角收斂"""
    signals: list[dict[str, Any]] = []
    if len(highs) < 3 or len(lows) < 3:
        return signals

    recent_h = highs[-3:]
    recent_l = lows[-3:]
    h_vals = [p for _, p in recent_h]
    l_vals = [p for _, p in recent_l]

    falling_highs = h_vals[0] > h_vals[1] > h_vals[2]
    rising_lows = l_vals[0] < l_vals[1] < l_vals[2]
    flat_highs = abs(h_vals[0] - h_vals[2]) / h_vals[0] <= 0.006
    flat_lows = abs(l_vals[0] - l_vals[2]) / l_vals[0] <= 0.006

    tri_type = None
    if falling_highs and rising_lows:
        tri_type = "對稱三角收斂"
    elif flat_highs and rising_lows:
        tri_type = "上升三角收斂"
    elif falling_highs and flat_lows:
        tri_type = "下降三角收斂"

    if tri_type:
        initial_range = h_vals[0] - l_vals[0]
        current_range = h_vals[-1] - l_vals[-1]
        if initial_range > 0 and (current_range / initial_range) <= 0.65:
            last_close = float(df["close"].iloc[-1])
            breakout_note = "處於末端收斂區間"
            if last_close > h_vals[-1]:
                breakout_note = "剛向上突破上緣"
            elif last_close < l_vals[-1]:
                breakout_note = "剛向下跌破下緣"
            signals.append({
                "pattern": tri_type,
                "direction": "待方向確認" if "末端" in breakout_note else ("偏多" if "上" in breakout_note else "偏空"),
                "detail": f"上緣 {h_vals[-1]:.2f} / 下緣 {l_vals[-1]:.2f}（{breakout_note}）",
                "key_level": round(h_vals[-1] if "上" in breakout_note else l_vals[-1], 4),
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
                "key_level": round(major_res, 4),
            })
        elif float(prev["close"]) <= major_res and float(last["close"]) > major_res and vol_ratio >= 1.4:
            signals.append({
                "pattern": "關鍵壓力帶量突破",
                "direction": "偏多",
                "detail": f"放量 ({vol_ratio:.1f}x) 站上關鍵壓力 {major_res:.2f}，收在 {last['close']:.2f}",
                "key_level": round(major_res, 4),
            })

    if ref_lows:
        major_sup = min(ref_lows[-3:]) if len(ref_lows) >= 3 else min(ref_lows)
        if float(last["low"]) < major_sup and float(last["close"]) > major_sup:
            signals.append({
                "pattern": "假跌破獵取 (Liquidity Sweep)",
                "direction": "偏多",
                "detail": f"最低插破前低支撐 {major_sup:.2f} 至 {last['low']:.2f} 後收回 {last['close']:.2f}（量能 {vol_ratio:.1f}x）",
                "key_level": round(major_sup, 4),
            })
        elif float(prev["close"]) >= major_sup and float(last["close"]) < major_sup and vol_ratio >= 1.4:
            signals.append({
                "pattern": "關鍵支撐帶量跌破",
                "direction": "偏空",
                "detail": f"放量 ({vol_ratio:.1f}x) 跌破關鍵支撐 {major_sup:.2f}，收在 {last['close']:.2f}",
                "key_level": round(major_sup, 4),
            })

    return signals


def verify_with_gemini(
    symbol: str, timeframe: str, df: pd.DataFrame, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """呼叫 Gemini API 進行結構複判與雜訊過濾"""
    if not GEMINI_API_KEY:
        return {
            "is_valid": True,
            "confidence": 7,
            "bias": candidates[0]["direction"],
            "summary": "（未設定 GEMINI_API_KEY）" + "；".join(c["detail"] for c in candidates),
        }

    recent_candles = df.tail(20)[["dt", "open", "high", "low", "close", "volume", "rsi"]].copy()
    recent_candles["dt"] = recent_candles["dt"].dt.strftime("%m-%d %H:%M")
    candles_text = recent_candles.round(2).to_string(index=False)

    prompt = f"""你是專業的加密貨幣技術分析師。演算法在 {symbol} ({timeframe} 週期) 偵測到以下候選型態/結構：
{json.dumps(candidates, ensure_ascii=False, indent=2)}

以下為最近 20 根已收盤 K 線數據 (UTC 時間)：
{candles_text}

請嚴格評估這些型態或流動性結構是否具備交易參考價值（過濾掉震盪雜訊或缺乏量價配合的假訊號），並只回傳純 JSON（不要加 markdown 標記），格式如下：
{{
  "is_valid": true 或 false,
  "confidence": 1 到 10 的整數,
  "bias": "偏多" 或 "偏空" 或 "待確認",
  "summary": "用 2-3 句繁體中文說明結構重點、關鍵觀察價位與失效位置（止損參考）"
}}"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


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
    df: pd.DataFrame, symbol: str, timeframe: str, key_levels: list[float]
) -> bytes:
    """繪製仿 BingX App 風格的高清晰度 K 線圖（右側價位軸、最高/最低點標註、現價與關鍵位標籤）"""
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

    buf = io.BytesIO()
    fig, axes = mpf.plot(
        ohlcv_df,
        type="candle",
        volume=True,
        panel_ratios=(5, 1.2),
        addplot=addplots,
        style=style,
        title=f"{symbol}  ·  {timeframe}  (UTC+8)",
        figsize=(12, 6.8),
        xlim=(-1, len(ohlcv_df) + 12),
        returnfig=True,
        tight_layout=True,
    )

    ax_main = axes[0]
    n = len(ohlcv_df)

    # 1. 標註最高價與最低價（仿 BingX 白色細指引線）
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
        fontsize=9.5,
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
        fontsize=9.5,
        va="center",
        ha="left" if lo_offset > 0 else "right",
        fontweight="bold",
    )

    # 2. 畫出關鍵頸線／支撐壓力虛線與右側外框標籤
    for lvl in unique_levels:
        ax_main.axhline(lvl, color="#f0b90b", linestyle="--", linewidth=1.1, alpha=0.9)
        ax_main.text(
            n + 1.5,
            lvl,
            f" {_fmt_price(lvl)} ",
            color="#f0b90b",
            fontsize=9,
            va="center",
            ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#1e1b10", edgecolor="#f0b90b", linewidth=1),
        )

    # 3. 畫出現價虛線與右側白色外框標籤
    ax_main.plot([n - 1, n + 1.2], [last_price, last_price], color="#eaecef", linestyle=":", linewidth=1.0)
    ax_main.text(
        n + 1.5,
        last_price,
        f" {_fmt_price(last_price)} ",
        color="#eaecef",
        fontsize=9.5,
        va="center",
        ha="left",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e2329", edgecolor="#eaecef", linewidth=1),
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
                    chart_bytes = render_chart_png(df, symbol, tf, key_levels)
                    msg = (
                        f"🔔 <b>AI 盯盤警報｜{symbol} ({tf})</b>\n"
                        f"• <b>偵測結構</b>：{patterns_str}\n"
                        f"• <b>AI 傾向</b>：{ai_result.get('bias')}（信心 {ai_result.get('confidence')}/10）\n"
                        f"• <b>現價</b>：{last_price:.2f}（收盤時間 {candle_tw} UTC+8）\n"
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
