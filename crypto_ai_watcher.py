"""
Crypto AI Vision Market Watcher（極速省時 + 8/10 高勝率標記 + 風險對齊建議槓桿倍率版）
"""

import argparse
import base64
import html
import io
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import ccxt
import matplotlib.pyplot as plt
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
MAX_MARGIN_LOSS_PCT = float(os.getenv("MAX_MARGIN_LOSS_PCT", "0.15"))  # 單筆止損預計虧損保證金比例 (預設 15%)
HISTORY_FILE = Path("sent_history.json")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# ======================================================

TF_MINUTES_MAP = {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
COOLDOWN_HOURS_MAP = {"15m": 2, "30m": 3, "1h": 6, "4h": 16, "1d": 48}


def load_history() -> dict[str, Any]:
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_history(history: dict[str, Any]) -> None:
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def is_in_cooldown_precheck(
    history: dict[str, Any], symbol: str, tf: str, current_price: float | None = None
) -> tuple[bool, str]:
    """在呼叫 AI 之前先檢查冷卻狀態，若價格偏離上次進場點 > 2% 則提早解除冷卻"""
    key = f"{symbol}|{tf}"
    rec = history.get(key)
    if not isinstance(rec, dict):
        return False, ""
    elapsed_hours = (time.time() - float(rec.get("ts", 0))) / 3600
    cooldown_h = COOLDOWN_HOURS_MAP.get(tf, 3)
    if elapsed_hours >= cooldown_h:
        return False, ""
    last_entry = float(rec.get("entry", 0))
    if current_price and last_entry > 0 and abs(current_price - last_entry) / last_entry > 0.02:
        return False, ""
    return True, str(rec.get("pattern") or "已推播結構")


def _validate_trade_plan(bias: str, entry: float, stop_loss: float, take_profit: float) -> tuple[bool, str]:
    """嚴格校驗 AI 給出的多空點位邏輯與止損距離"""
    if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
        return False, "點位數值異常 (<= 0)"

    sl_pct = abs(entry - stop_loss) / entry
    if sl_pct <= 0.001:
        return False, f"止損距離過窄 ({sl_pct * 100:.3f}%)"

    if "偏多" in bias:
        if not (stop_loss < entry < take_profit):
            return False, f"偏多點位邏輯錯誤 (SL={stop_loss}, Entry={entry}, TP={take_profit})"
    elif "偏空" in bias:
        if not (take_profit < entry < stop_loss):
            return False, f"偏空點位邏輯錯誤 (TP={take_profit}, Entry={entry}, SL={stop_loss})"
    else:
        # 待突破確認：至少確保 SL 與 TP 位於 Entry 兩側
        if (entry - stop_loss) * (take_profit - entry) <= 0:
            return False, "SL 與 TP 位於 Entry 同側"

    return True, ""


def _suggest_leverage(entry: float, stop_loss: float) -> tuple[int, float, float]:
    """依止損距離計算建議槓桿倍率（控制打損約虧保證金 15%，限制在 2x ~ 20x 之間）"""
    sl_pct = abs(entry - stop_loss) / entry
    raw_lev = int(MAX_MARGIN_LOSS_PCT / sl_pct)
    lev = max(2, min(20, raw_lev))
    est_margin_loss_pct = sl_pct * lev * 100
    return lev, sl_pct * 100, est_margin_loss_pct


def fetch_ticker_price(
    exchange: ccxt.Exchange, symbol: str, fallback_exchange: ccxt.Exchange | None = None
) -> float | None:
    """輕量取得最新成交價，供冷卻期波動檢查使用"""
    for ex in (exchange, fallback_exchange):
        if ex is None:
            continue
        try:
            ticker = ex.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            if price:
                return float(price)
        except Exception:
            continue
    return None


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = 200,
    fallback_exchange: ccxt.Exchange | None = None,
) -> pd.DataFrame:
    """抓取 K 線資料，遇限流自動重試 2 次，仍失敗則切換備援交易所"""
    last_err: Exception | None = None
    for attempt in range(1, 3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            if len(df) > 2:
                df = df.iloc[:-1].reset_index(drop=True)
            return df
        except Exception as exc:
            last_err = exc
            print(f"[重試 {attempt}/2] {exchange.id} 抓取 {symbol} {timeframe} 失敗: {exc}")
            time.sleep(1.5 * attempt)

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
    """計算 EMA20、EMA50、RSI(14)、ATR(14)（採 Wilder's Smoothing 對齊 TradingView）與 20 週期均量"""
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df


def _fmt_price(val: float) -> str:
    if val >= 1000:
        return f"{val:,.1f}"
    if val >= 10:
        return f"{val:.2f}"
    if val >= 1:
        return f"{val:.3f}"
    return f"{val:.5f}"


def _extract_chart_pivots(plot_df: pd.DataFrame, window: int = 3) -> list[dict[str, Any]]:
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
    """抗塞車版：自動切換多組備援模型 + 拉長退避等待時間"""
    if not GEMINI_API_KEY:
        raise RuntimeError("請設定 GEMINI_API_KEY")

    models_to_try = [
        GEMINI_MODEL,
        "gemini-3.5-flash-lite",
    ]
    last_exc: Exception | None = None
    for model_name in dict.fromkeys(models_to_try):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        for attempt in range(1, 3):
            time.sleep(2.0)  # 每次請求前先停 2 秒，避免連打被限流
            try:
                resp = requests.post(url, json=payload, timeout=(10, 60))
                if resp.status_code == 404:
                    last_exc = requests.HTTPError(f"404 Not Found for {model_name}", response=resp)
                    break
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = requests.HTTPError(f"{resp.status_code} for {model_name}", response=resp)
                    wait_sec = 5 * attempt  # 遇塞車等 5 秒、10 秒再試
                    print(f"[Gemini {resp.status_code}] 等待 {wait_sec}s 重試或切換備用模型...")
                    time.sleep(wait_sec)
                    continue
                resp.raise_for_status()
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(text)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return parsed[0]
                if isinstance(parsed, dict):
                    return parsed
                raise ValueError("Gemini 回傳非預期的 JSON 格式")
            except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
                last_exc = exc
                time.sleep(5 * attempt)

    raise last_exc or RuntimeError("Gemini API 呼叫失敗")


def render_chart_png(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    entry: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trend_lines: list[list[list[float]]] | None = None,
    high_prob_score: int | None = None,
    dpi: int = 180,
) -> bytes:
    """繪製仿 BingX / TradingView 風格 K 線圖（含 8/10+ 高勝率金色徽章）"""
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

    chart_min = float(plot_df["low"].min())
    chart_max = float(plot_df["high"].max())
    last_price = float(ohlcv_df["Close"].iloc[-1])

    def _valid_price(v: float | None) -> float | None:
        if v is None or v <= 0:
            return None
        if chart_min * 0.8 <= float(v) <= chart_max * 1.2:
            return float(v)
        return None

    entry_v = _valid_price(entry)
    sl_v = _valid_price(stop_loss)
    tp_v = _valid_price(take_profit)

    all_y = [chart_min, chart_max]
    for v in (entry_v, sl_v, tp_v):
        if v is not None:
            all_y.append(v)
    y_min, y_max = min(all_y), max(all_y)
    y_pad = (y_max - y_min) * 0.09 if y_max > y_min else y_max * 0.02
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    is_high_prob = high_prob_score is not None and high_prob_score >= 8
    title_str = (
        f"★ HIGH PROBABILITY ({high_prob_score}/10)  |  {symbol} · {timeframe} (UTC+8)"
        if is_high_prob
        else f"{symbol}  ·  {timeframe}  (UTC+8)"
    )

    buf = io.BytesIO()
    fig, axes = mpf.plot(
        ohlcv_df,
        type="candle",
        volume=True,
        panel_ratios=(5, 1.2),
        addplot=addplots,
        style=style,
        title=title_str,
        figsize=(12.5, 7),
        xlim=(-1, len(ohlcv_df) + 18),
        ylim=(y_lo, y_hi),
        returnfig=True,
        tight_layout=True,
    )

    try:
        ax_main = axes[0]
        n = len(ohlcv_df)

        if is_high_prob:
            ax_main.text(
                0.015,
                0.96,
                f" ★ HIGH WIN-RATE {high_prob_score}0%+ ({high_prob_score}/10) ★ ",
                transform=ax_main.transAxes,
                color="#0b0e11",
                fontsize=10.5,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0b90b", edgecolor="#fff3cd", linewidth=1.5),
                zorder=10,
            )

        if isinstance(trend_lines, list):
            for line in trend_lines:
                if not isinstance(line, list) or len(line) < 2:
                    continue
                xs, ys = [], []
                for pt in line:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        bx = max(0.0, min(float(n - 1), float(pt[0])))
                        py = float(pt[1])
                        if chart_min * 0.85 <= py <= chart_max * 1.15:
                            xs.append(bx)
                            ys.append(py)
                if len(xs) >= 2:
                    ax_main.plot(xs, ys, color="#b2b5be", linewidth=2.0, alpha=0.9, zorder=5)

        if tp_v is not None:
            start_y = entry_v if entry_v is not None else last_price
            arrow_color = "#0ecb81" if tp_v >= start_y else "#f6465d"
            ax_main.annotate(
                "",
                xy=(n + 7, tp_v),
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
        if tp_v is not None:
            lines_to_draw.append(("TP", tp_v, "#0ecb81", "#0c261d", "--"))
        if entry_v is not None:
            lines_to_draw.append(("Entry", entry_v, "#eaecef", "#1e2329", "-."))
        if sl_v is not None:
            lines_to_draw.append(("SL", sl_v, "#f6465d", "#2c1117", "--"))

        if lines_to_draw:
            lines_to_draw.sort(key=lambda item: item[1])
            min_gap = (y_hi - y_lo) * 0.06
            placed_ys: list[float] = []
            for label, price_val, edge_col, bg_col, ls in lines_to_draw:
                ax_main.axhline(price_val, color=edge_col, linestyle=ls, linewidth=1.2, alpha=0.95)
                label_y = price_val
                if placed_ys and (label_y - placed_ys[-1]) < min_gap:
                    label_y = placed_ys[-1] + min_gap
                placed_ys.append(label_y)
                ax_main.text(
                    right_x,
                    label_y,
                    f" {label} {_fmt_price(price_val)} ",
                    color=edge_col,
                    fontsize=9,
                    va="center",
                    ha="left",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor=bg_col, edgecolor=edge_col, linewidth=1),
                )
        else:
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

        fig.savefig(buf, dpi=dpi, bbox_inches="tight", facecolor="#0b0e11")
    finally:
        plt.close(fig)

    buf.seek(0)
    return buf.read()


def stage1_detect_pattern_from_chart(
    symbol: str, timeframe: str, df: pd.DataFrame, raw_chart_bytes: bytes
) -> dict[str, Any]:
    """【AI 第一關】嚴格幾何校驗所有型態，防止把 V 型反轉或單邊回檔誤認成頭肩底/三角"""
    plot_df = df.tail(85).copy().reset_index(drop=True)
    pivots = _extract_chart_pivots(plot_df, window=3)
    last_price = float(plot_df["close"].iloc[-1])
    img_b64 = base64.b64encode(raw_chart_bytes).decode("utf-8")

    prompt = f"""你是極度嚴謹的技術分析師。請觀察這張 {symbol} ({timeframe} 週期、共 85 根 K 線，最左 bar 0、最右最新收盤 bar 84) 的 K 線圖。
目前收盤價 {_fmt_price(last_price)}，轉折點座標參考：
{json.dumps(pivots, ensure_ascii=False)}

請依序檢查「最右側（最新走勢）」是否**真正符合**以下任一幾何定義（必須先確認轉折點數量與位置吻合，不吻合就一律回傳 false，寧缺勿濫）：
1. **頭肩頂**：必須有 3 個清楚波峰（左肩、明顯更高的頭部、右肩低於頭部），且中間有頸線；單純震盪不算。
2. **頭肩底**：必須有 3 個清楚波谷（左肩、明顯更低的頭部、右肩高於頭部）。**注意：單純的 V 型反轉或跌深後一路階梯墊高，沒有清楚獨立左肩與右肩波谷，絕對不算頭肩底！**
3. **M 頭 (Double Top)**：必須有 2 個高度相近的波段高點（誤差約 1.5% 內）與中間頸線。
4. **W 底 (Double Bottom)**：必須有 2 個深度相近的波段低點（誤差約 1.5% 內）與中間頸線。
5. **三角收斂 / 楔形**：上緣必須至少有 2 個高點、下緣必須至少有 2 個低點，兩條邊線明顯向右側收攏；單一尖頭回檔不算。
6. **上升 / 下降通道或旗形**：上下緣各有至少 2 個觸及點且大致平行。
7. **關鍵支撐壓力突破 / 假跌破假突破獵取 (Liquidity Sweep)**：最新幾根 K 線明確放量突破或長影線刺破前期顯著高低點後收回。

若沒有明確符合上述定義的結構，請務必設 `has_pattern = false`。
若確認符合，請先在 `pattern_note` 寫出具體各轉折點 bar 位置，再回傳 `trend_lines`：
- 三角 / 楔形 / 通道：回傳 **2 條獨立直線**（第 1 條連上緣高點、第 2 條連下緣低點）。
- M 頭 / W 底 / 頭肩：回傳 **1 條嚴格交替轉折的折線**（例如頭肩底為：高-左肩低-高-頭低-高-右肩低-高）。

只回傳純 JSON：
{{
  "has_pattern": true 或 false,
  "pattern_name": "型態名稱",
  "bias": "偏多" 或 "偏空" 或 "待突破確認",
  "trend_lines": [[[bar, price], [bar, price], ...]],
  "pattern_note": "寫出具體轉折點 bar 位置與結構說明"
}}"""

    return _call_gemini([
        {"text": prompt},
        {"inlineData": {"mimeType": "image/png", "data": img_b64}},
    ])


def stage2_score_risk_and_plan(
    symbol: str, timeframe: str, df: pd.DataFrame, raw_chart_bytes: bytes, stage1_result: dict[str, Any]
) -> dict[str, Any]:
    """【AI 第二關】風控評分"""
    recent = df.tail(25)[["dt", "open", "high", "low", "close", "volume", "ema20", "ema50", "rsi", "atr"]].copy()
    recent["dt"] = recent["dt"].dt.strftime("%m-%d %H:%M")
    candles_text = recent.round(4).to_string(index=False)
    img_b64 = base64.b64encode(raw_chart_bytes).decode("utf-8")

    prompt = f"""你是嚴謹的加密貨幣交易風控官。第一關看圖辨識在 {symbol} ({timeframe}) 發現了型態：
{json.dumps(stage1_result, ensure_ascii=False)}

以下為最近 25 根 K 線與均線/RSI/ATR 數據：
{candles_text}

請同時對照 K 線圖與量價數據進行嚴格風險評估：
1. 評估是否有逆勢撞均線、量能背離、止損位置合理性，給出 `risk_score`（1 到 10 的整數，8 分以上代表極高勝率 80%+ 優質機會）。
2. 給出具體的 `entry`（入場點）、`stop_loss`（止損點）、`take_profit`（止盈點，風報比至少 1:1.8）。

只回傳純 JSON：
{{
  "risk_score": 1 到 10 的整數,
  "risk_level": "高勝率精選" 或 "低風險" 或 "中風險" 或 "高風險",
  "entry": 數字,
  "stop_loss": 數字,
  "take_profit": 數字,
  "summary": "用 2 句繁體中文說明型態重點、風險評估理由與操作建議"
}}"""

    return _call_gemini([
        {"text": prompt},
        {"inlineData": {"mimeType": "image/png", "data": img_b64}},
    ])


def check_telegram_check_command() -> str | None:
    """僅在收到 /check 指令時才觸發全盤體檢，避免一般訊息誤觸發"""
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
        trigger_msg = None
        for u in updates:
            msg = u.get("message") or {}
            if str(msg.get("chat", {}).get("id", "")) == str(TELEGRAM_CHAT_ID):
                text = (msg.get("text") or "").strip()
                if text.lower().startswith("/check"):
                    trigger_msg = text

        requests.get(url, params={"offset": max_id + 1, "timeout": 1}, timeout=10)
        return trigger_msg
    except Exception as exc:
        print(f"[Telegram 收訊檢查失敗] {exc}")
        return None


def _safe_join_lines(header: str, lines: list[str], max_len: int = 3900) -> str:
    """按行安全組合訊息，避免切破 HTML 標籤"""
    out = header
    for line in lines:
        if len(out) + len(line) + 1 > max_len:
            out += "\n...(訊息過長，其餘省略)"
            break
        out += line + "\n"
    return out.rstrip()


def send_telegram_text(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[模擬文字推播]\n" + text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=15,
    )
    resp.raise_for_status()


def send_telegram_photo(image_bytes: bytes, caption: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[模擬推播] 未設定 Telegram Token/Chat ID，訊息內容如下：\n" + caption)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    resp = requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
        files={"photo": ("chart.png", image_bytes, "image/png")},
        timeout=20,
    )
    resp.raise_for_status()


def is_recently_closed_for_tf(df: pd.DataFrame, timeframe: str) -> bool:
    """收盤後 28 分鐘內才掃描，避免每 30 分鐘排程重複掃描同一根 K 線"""
    last_open_utc = df["dt"].iloc[-1].to_pydatetime()
    tf_minutes = TF_MINUTES_MAP.get(timeframe, 30)
    last_close_utc = last_open_utc + timedelta(minutes=tf_minutes)
    age_minutes = (datetime.now(timezone.utc) - last_close_utc).total_seconds() / 60
    return -2 <= age_minutes <= 28


def run(ignore_time_window: bool = False) -> None:
    user_query = check_telegram_check_command()
    if user_query:
        print(f"[收到 Telegram 指令] {user_query}")

    history = load_history()
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
        current_price = fetch_ticker_price(exchange, symbol, fallback_exchange)

        for raw_tf in TIMEFRAMES:
            tf = raw_tf.strip()
            if not tf:
                continue
            try:
                in_cd_pre, prev_pat = is_in_cooldown_precheck(history, symbol, tf, current_price)
                if in_cd_pre:
                    status_lines.append(f"• <b>{short_sym} ({tf})</b>：⏳ 冷卻中（{html.escape(prev_pat)}，略過掃描）")
                    print(f"[冷卻跳過] {symbol} {tf} 直接跳過 AI 掃描")
                    continue

                df = fetch_ohlcv(exchange, symbol, tf, limit=CANDLE_LIMIT, fallback_exchange=fallback_exchange)
                if not ignore_time_window and not user_query and not is_recently_closed_for_tf(df, tf):
                    continue

                df = add_indicators(df)
                raw_chart = render_chart_png(df, symbol, tf, dpi=110)

                s1 = stage1_detect_pattern_from_chart(symbol, tf, df, raw_chart)
                if not s1.get("has_pattern"):
                    status_lines.append(f"• <b>{short_sym} ({tf})</b>：無明確型態")
                    print(f"[第一關過濾] {symbol} {tf} 無明確型態")
                    continue

                pat_name = html.escape(str(s1.get("pattern_name") or "未命名型態"))
                bias_raw = str(s1.get("bias") or "待確認")
                bias_str = html.escape(bias_raw)
                print(f"[第一關命中] {symbol} {tf}: {pat_name} ({bias_str})")

                s2 = stage2_score_risk_and_plan(symbol, tf, df, raw_chart, s1)
                risk_score = int(s2.get("risk_score", 0))

                if risk_score >= MIN_AI_CONFIDENCE:
                    last_price = float(df["close"].iloc[-1])
                    tf_mins = TF_MINUTES_MAP.get(tf, 30)
                    close_dt_tw = (df["dt"].iloc[-1] + timedelta(minutes=tf_mins)).astimezone(tz_tw)
                    candle_tw = close_dt_tw.strftime("%Y-%m-%d %H:%M")

                    entry_p = float(s2.get("entry") or last_price)
                    is_short = "偏空" in bias_raw
                    sl_p = float(s2.get("stop_loss") or (last_price * 1.02 if is_short else last_price * 0.98))
                    tp_p = float(s2.get("take_profit") or (last_price * 0.96 if is_short else last_price * 1.04))

                    is_valid, invalid_reason = _validate_trade_plan(bias_raw, entry_p, sl_p, tp_p)
                    if not is_valid:
                        status_lines.append(f"• <b>{short_sym} ({tf})</b>：⚠️ 點位校驗未通過（{html.escape(invalid_reason)}）")
                        print(f"[點位校驗過濾] {symbol} {tf}: {invalid_reason}")
                        continue

                    risk = abs(entry_p - sl_p)
                    rr = abs(tp_p - entry_p) / risk
                    lev, sl_dist_pct, margin_loss_pct = _suggest_leverage(entry_p, sl_p)
                    risk_lvl = html.escape(str(s2.get("risk_level") or ""))
                    summary_safe = html.escape(str(s2.get("summary") or "")[:240])

                    marked_chart = render_chart_png(
                        df, symbol, tf,
                        entry=entry_p, stop_loss=sl_p, take_profit=tp_p,
                        trend_lines=s1.get("trend_lines"),
                        high_prob_score=risk_score if risk_score >= 8 else None,
                    )
                    if risk_score >= 8:
                        title_line = f"🚨🔥 <b>【高勝率 80%+ 精選】{html.escape(symbol)} ({tf})</b> 🔥🚨\n"
                        score_line = f"• <b>勝率評分</b>：⭐ <b>{risk_score}/10（{risk_score}0% 高勝率重磅訊號）</b>\n"
                        status_badge = f"🔥✅ 已發圖【高勝率 {risk_score}/10】"
                    else:
                        title_line = f"🔔 <b>AI 盯盤警報｜{html.escape(symbol)} ({tf})</b>\n"
                        score_line = f"• <b>風險評分</b>：{risk_score}/10（{risk_lvl}）\n"
                        status_badge = f"✅ 已發圖（{pat_name}，{risk_score}/10）"

                    msg = (
                        title_line
                        + f"• <b>型態結構</b>：{pat_name}（{bias_str}）\n"
                        + score_line
                        + f"• <b>現價</b>：{_fmt_price(last_price)}（收盤 {candle_tw} UTC+8）\n"
                        + f"• <b>參考規劃</b>：入場 <code>{_fmt_price(entry_p)}</code> ｜ 止損 <code>{_fmt_price(sl_p)}</code> ｜ 止盈 <code>{_fmt_price(tp_p)}</code>（R:R ≈ 1:{rr:.1f}）\n"
                        + f"• <b>建議倍率</b>：<code>{lev}x</code>（止損距離 {sl_dist_pct:.2f}%，打損約虧保證金 {margin_loss_pct:.0f}%）\n"
                        + f"• <b>AI 分析</b>：{summary_safe}"
                    )
                    send_telegram_photo(marked_chart, msg)
                    history[f"{symbol}|{tf}"] = {
                        "pattern": s1.get("pattern_name"),
                        "bias": bias_raw,
                        "entry": entry_p,
                        "ts": time.time(),
                    }
                    save_history(history)
                    status_lines.append(f"• <b>{short_sym} ({tf})</b>：{status_badge}")
                    print(f"[第二關通過已推播] {symbol} {tf}: 評分 {risk_score}/10, 建議 {lev}x")
                else:
                    status_lines.append(
                        f"• <b>{short_sym} ({tf})</b>：🟡 有「{pat_name}」但風險評分 {risk_score}/10（未達 {MIN_AI_CONFIDENCE} 分）"
                    )
                    print(f"[第二關風險過濾] {symbol} {tf} {pat_name} 評分僅 {risk_score}/10")

            except Exception as exc:
                err_safe = html.escape(str(exc)[:80])
                err_msg = f"• <b>{short_sym} ({tf})</b>：❌ 錯誤（{err_safe}）"
                status_lines.append(err_msg)
                error_lines.append(err_msg)
                print(f"[錯誤] {symbol} {tf} 處理失敗: {exc}")

    if user_query or error_lines:
        now_tw = datetime.now(tz_tw).strftime("%m-%d %H:%M")
        header = (
            f"📋 <b>盯盤系統體檢回報（{now_tw} UTC+8）</b>\n"
            f"目前門檻：風險評分 ≥ <b>{MIN_AI_CONFIDENCE}/10</b>\n"
        )
        if user_query:
            header += f"收到指令：「<i>{html.escape(user_query[:30])}</i>」\n\n"
        else:
            header += "⚠️ <b>偵測到部分項目異常：</b>\n\n"

        full_text = _safe_join_lines(header, status_lines) if status_lines else header + "本輪無剛收盤的週期。"
        send_telegram_text(full_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto AI Two-Stage Vision Watcher")
    parser.add_argument("--test", action="store_true", help="忽略時間窗口限制，強制掃描一次")
    args = parser.parse_args()
    run(ignore_time_window=args.test)
