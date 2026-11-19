"""BingX / Binance Gemini 型態觀察器 v2：AI 評分與統計勝率分開。"""

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


# 公開 USDT 永續合約行情；金鑰由 watcher_runtime 載入本機 .env。
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "bingx")
FALLBACK_EXCHANGE_ID = os.getenv("FALLBACK_EXCHANGE_ID", "binanceusdm")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,SUI/USDT:USDT,ZEC/USDT:USDT").split(",")
TIMEFRAMES = os.getenv("TIMEFRAMES", "30m,1h,4h,1d").split(",")
CANDLE_LIMIT = 200
TF_MINUTES_MAP = {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算 EMA20、EMA50、RSI(14)、ATR(14)（EMA 平滑近似，非保證與交易平台逐值一致）與 20 週期均量"""
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df.loc[(loss == 0) & (gain > 0), "rsi"] = 100
    df.loc[(loss == 0) & (gain == 0), "rsi"] = 50

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
    from watcher_runtime import call_gemini, settings
    return call_gemini(settings(), parts)


def render_chart_png(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    entry: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trend_lines: list[list[list[float]]] | None = None,
    dpi: int = 180,
) -> bytes:
    """繪製仿 BingX / TradingView 風格 K 線圖（型態與參考點位）"""
    if "ema20" not in df.columns:
        df = add_indicators(df.copy())

    plot_df = df.tail(85).copy().reset_index(drop=True)
    plot_df["dt_tw"] = plot_df["dt"].dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
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

    title_str = f"{symbol}  ·  {timeframe}  (UTC+8)"

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

若沒有明確符合上述定義的結構，或仍待突破確認，請務必設 `has_pattern = false`。
只看已提供資料，不猜測圖外走勢。不因辨識出型態就宣稱高勝率。
pattern_code 必須選固定代碼：head_shoulders_top、head_shoulders_bottom、double_top、double_bottom、triangle、wedge、channel、flag、breakout、liquidity_sweep。
pattern_start_bar 為型態最早結構轉折點的 bar 整數索引（0–81），每次檢視同一型態應錨定同一根歷史 K 線。
若確認符合，請先在 `pattern_note` 寫出具體各轉折點 bar 位置，再回傳 `trend_lines`：
- 三角 / 楔形 / 通道：回傳 **2 條獨立直線**（第 1 條連上緣高點、第 2 條連下緣低點）。
- M 頭 / W 底 / 頭肩：回傳 **1 條嚴格交替轉折的折線**（例如頭肩底為：高-左肩低-高-頭低-高-右肩低-高）。

只回傳純 JSON：
{{
  "has_pattern": true 或 false,
  "pattern_name": "型態名稱",
  "pattern_code": "固定英文代碼",
  "pattern_start_bar": 0,
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
1. 評估是否有逆勢撞均線、量能背離、止損位置合理性，給出 `risk_score`（1 到 10 的整數，只代表主觀品質，絕對不是成功機率）。
2. 若尚未確認突破／回踩或量價有疑慮，confirmed=false，禁止把未確認條件當成成立。
3. 本版只做回踩限價入場：偏多 entry 不高於最新收盤，偏空 entry 不低於最新收盤；入場距收盤最多 5%。
4. 給出具體的 `entry`（入場點）、`stop_loss`（止損點）、`take_profit`（止盈點，風報比至少 1:1.8）。

只回傳純 JSON：
{{
  "confirmed": true 或 false,
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


def download_market_data(output_dir: str) -> None:
    """下載公開 USDT 永續合約 K 線，不呼叫 AI 或 Telegram。"""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    exchanges = [getattr(ccxt, name)({"enableRateLimit": True, "timeout": 15000})
                 for name in dict.fromkeys([EXCHANGE_ID, FALLBACK_EXCHANGE_ID])]
    manifest = {"downloaded_at": datetime.now(timezone.utc).isoformat(), "files": [], "errors": []}
    for symbol in map(str.strip, SYMBOLS):
        if not symbol:
            continue
        for tf in map(str.strip, TIMEFRAMES):
            if not tf:
                continue
            for exchange in exchanges:
                try:
                    time.sleep(0.6)
                    rows = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=CANDLE_LIMIT + 1)
                    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df = df.drop_duplicates("timestamp").sort_values("timestamp")
                    now_ms = int(time.time() * 1000)
                    df = df[df["timestamp"] + TF_MINUTES_MAP[tf] * 60000 <= now_ms].tail(CANDLE_LIMIT).copy()
                    if df.empty:
                        raise ValueError("未取得已收盤 K 線")
                    if not np.isfinite(df.to_numpy(dtype=float)).all():
                        raise ValueError("行情包含非有限數值")
                    if ((df["high"] < df[["open", "close", "low"]].max(axis=1)) |
                        (df["low"] > df[["open", "close", "high"]].min(axis=1))).any():
                        raise ValueError("OHLC 價格關係異常")
                    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df["datetime_taipei"] = df["datetime_utc"].dt.tz_convert("Asia/Taipei")
                    df["exchange"] = exchange.id
                    df["symbol"] = symbol
                    filename = f"{exchange.id}_{symbol.split('/')[0]}_USDT_{tf}.csv"
                    df.to_csv(target / filename, index=False, encoding="utf-8-sig")
                    manifest["files"].append({"file": filename, "rows": len(df), "exchange": exchange.id,
                                              "symbol": symbol, "timeframe": tf,
                                              "last_open_utc": str(df["datetime_utc"].iloc[-1])})
                    print(f"[OK] {filename}: {len(df)} candles", flush=True)
                    break
                except Exception as exc:
                    print(f"[FAILED] {exchange.id} {symbol} {tf}: {exc}", flush=True)
            else:
                manifest["errors"].append({"symbol": symbol, "timeframe": tf})
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest['files'])} files to {target.resolve()}", flush=True)
    if manifest["errors"]:
        raise SystemExit(1)


def main():
    from watcher_runtime import load_env, settings, list_models, run_watcher, show_status
    load_env()
    # Download mode also respects local .env settings.
    global EXCHANGE_ID, FALLBACK_EXCHANGE_ID, SYMBOLS, TIMEFRAMES
    cfg = settings()
    EXCHANGE_ID, FALLBACK_EXCHANGE_ID = cfg["exchange"], cfg["fallback"]
    SYMBOLS, TIMEFRAMES = cfg["symbols"], cfg["timeframes"]
    parser = argparse.ArgumentParser(description="BingX / Binance Gemini 型態觀察器（模擬追蹤，不下單）")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--watch", action="store_true", help="持續輪詢；Ctrl+C 停止")
    modes.add_argument("--status", action="store_true", help="顯示設定及模擬紀錄統計")
    modes.add_argument("--list-models", action="store_true", help="查詢此 API Key 可用模型")
    modes.add_argument("--download-only", action="store_true", help="只下載行情 CSV")
    parser.add_argument("--track-only", action="store_true", help="只結算已有模擬訊號，不呼叫 Gemini")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--output", default="market_data")
    args = parser.parse_args()
    if args.status:
        show_status()
    elif args.list_models:
        print("\n".join(list_models(cfg)))
    elif args.download_only:
        download_market_data(args.output)
    else:
        import sys
        return run_watcher(sys.modules[__name__], args.watch, args.interval, args.track_only)
    return 0


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已停止；模擬紀錄已保存在本機。")
    except Exception as exc:
        from watcher_runtime import safe_error
        print("啟動失敗：" + safe_error(exc, {"key": os.getenv("GEMINI_API_KEY"), "token": os.getenv("TELEGRAM_BOT_TOKEN")}))
        raise SystemExit(1)
