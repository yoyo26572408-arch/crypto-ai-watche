"""Gemini K-line watcher v2.2. Standalone; auto model selection; no Telegram connection."""
from __future__ import annotations

# --- watcher_stats.py ---

import hashlib
import json
import math
import sqlite3
from pathlib import Path

VERSION = "forward-v2-limit-3-hold-24"
PATTERNS = {"head_shoulders_top", "head_shoulders_bottom", "double_top", "double_bottom",
            "triangle", "wedge", "channel", "flag", "breakout", "liquidity_sweep"}
TERMINAL = {"win", "loss", "timeout", "unfilled", "data_gap"}


def validate_plan(plan, min_rr=1.8):
    if plan.get("confirmed") is not True:
        raise ValueError("AI 未確認可執行")
    if plan.get("bias") not in {"偏多", "偏空"}:
        raise ValueError("多空方向未確認")
    for name in ("entry", "stop_loss", "take_profit"):
        value = plan.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必須為有限正數")
    entry, sl, tp = (plan[k] for k in ("entry", "stop_loss", "take_profit"))
    if not ((sl < entry < tp) if plan["bias"] == "偏多" else (tp < entry < sl)):
        raise ValueError("入場／止損／止盈方向不合理")
    if abs(entry - sl) / entry <= 0.001:
        raise ValueError("止損距離過窄")
    rr = abs(tp - entry) / abs(entry - sl)
    if rr < min_rr:
        raise ValueError("風報比未達門檻")
    score = plan.get("risk_score")
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 10:
        raise ValueError("AI 評分應為 1–10 整數")
    return rr


def wilson_lower(wins, total, z=1.959963984540054):
    """Lower endpoint of a two-sided 95% Wilson interval; not a guarantee."""
    if total == 0:
        return 0.0
    p = wins / total
    return (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total)


def cohort_key(exchange, symbol, tf, pattern, plan, model, policy):
    rr = validate_plan(plan, policy["min_rr"])
    # Keep instrument, exchange, timeframe, model and rules separate. Avoid mixing easy TP
    # with difficult TP: use narrow score, R:R and percentage-stop buckets.
    stop_pct = abs(plan["entry"] - plan["stop_loss"]) / plan["entry"] * 100
    values = [VERSION, exchange, symbol, tf, pattern, plan["bias"], plan["risk_score"],
              math.floor(rr * 2), math.floor(stop_pct * 2), model, policy]
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def fingerprint(pattern, bias, anchor_ms):
    return f"{pattern}|{bias}|{anchor_ms}"


def evaluate(signal, candles):
    """Consume contiguous CLOSED OHLCV bars strictly after signal creation.

    Entry is a limit order: long entry <= signal close, short entry >= signal close.
    Gap-through fills use the specified limit price (conservative, no improvement).
    Entry candle SL always loses; TP alone on entry candle only wins if entry was
    marketable at the OPEN, otherwise ordering is unknown and TP is ignored.
    """
    s = dict(signal)
    for bar in candles:
        ts = int(bar[0])
        if s["status"] in TERMINAL:
            break
        if ts < s["next_ms"]:
            continue
        if ts != s["next_ms"]:
            s.update(status="data_gap", resolved_ms=ts + s["tf_ms"])
            break
        op, hi, lo = map(float, bar[1:4])
        long = s["bias"] == "偏多"
        s["next_ms"] = ts + s["tf_ms"]
        just_entered = False
        if s["status"] == "pending":
            s["waited"] += 1
            filled = lo <= s["entry"] if long else hi >= s["entry"]
            if not filled:
                if s["waited"] >= s["entry_bars"]:
                    s.update(status="unfilled", resolved_ms=ts + s["tf_ms"])
                continue
            just_entered = True
            s.update(status="open", entered_ms=ts)
        s["held"] += 1
        sl_hit = lo <= s["stop_loss"] if long else hi >= s["stop_loss"]
        tp_hit = hi >= s["take_profit"] if long else lo <= s["take_profit"]
        entry_at_open = op <= s["entry"] if long else op >= s["entry"]
        if sl_hit:  # SL first, including ambiguous bars containing both.
            s.update(status="loss", resolved_ms=ts + s["tf_ms"])
        elif tp_hit and (not just_entered or entry_at_open):
            s.update(status="win", resolved_ms=ts + s["tf_ms"])
        elif s["held"] >= s["hold_bars"]:
            s.update(status="timeout", resolved_ms=ts + s["tf_ms"])
    return s


class Ledger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY, market TEXT NOT NULL, cohort TEXT NOT NULL,
                created_ms INTEGER NOT NULL, resolved_ms INTEGER,
                status TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS signals_cohort ON signals(cohort, resolved_ms);
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))

    def save(self, signal):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?)", (
                signal["id"], signal["market"], signal["cohort"], signal["created_ms"],
                signal.get("resolved_ms"), signal["status"], json.dumps(signal, ensure_ascii=False)))

    def active(self, market=None):
        sql = "SELECT payload FROM signals WHERE status IN ('pending','open')"
        rows = self.db.execute(sql + (" AND market=?" if market else ""), (market,) if market else ())
        return [json.loads(r[0]) for r in rows]

    def add(self, market, cohort, signal_ms, tf_ms, plan, policy, metadata=None):
        # One pending/open signal per exchange+symbol+timeframe; overlapping labels
        # otherwise exaggerate the effective sample count.
        if self.active(market):
            return None
        sid = hashlib.sha256(f"{market}|{signal_ms}|{cohort}".encode()).hexdigest()[:24]
        if self.db.execute("SELECT 1 FROM signals WHERE id=?", (sid,)).fetchone():
            return None
        signal = dict(plan, id=sid, market=market, cohort=cohort, created_ms=signal_ms,
                      tf_ms=tf_ms, next_ms=signal_ms, waited=0, held=0, status="pending",
                      entry_bars=policy["entry_bars"], hold_bars=policy["hold_bars"],
                      metadata=metadata or {}, exported=False)
        self.save(signal)
        return signal

    def update(self, market, candles):
        for signal in self.active(market):
            self.save(evaluate(signal, candles))

    def stats(self, cohort, asof_ms, min_samples=100, threshold=0.8):
        # Only previously resolved outcomes, never current/future labels. Unfilled
        # orders do not count as trades; expired filled orders count against success.
        rows = self.db.execute("""SELECT status, COUNT(*) AS n FROM signals
            WHERE cohort=? AND resolved_ms<=? GROUP BY status""", (cohort, asof_ms))
        counts = {r["status"]: r["n"] for r in rows}
        wins = counts.get("win", 0)
        total = sum(counts.get(k, 0) for k in ("win", "loss", "timeout"))
        lower = wilson_lower(wins, total)
        return dict(wins=wins, total=total, rate=wins/total if total else 0,
                    lower95=lower, unfilled=counts.get("unfilled", 0),
                    data_gap=counts.get("data_gap", 0),
                    eligible=total >= min_samples and lower >= threshold and not counts.get("data_gap", 0))

    def summary(self):
        return {r[0]: r[1] for r in self.db.execute("SELECT status, COUNT(*) FROM signals GROUP BY status")}


# --- watcher_runtime.py ---

import contextlib
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent


def load_env():
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def settings():
    load_env()
    cfg = dict(exchange=os.getenv("EXCHANGE_ID", "bingx"),
               fallback=os.getenv("FALLBACK_EXCHANGE_ID", "binanceusdm"),
               symbols=[s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,SUI/USDT:USDT,ZEC/USDT:USDT").split(",") if s.strip()],
               timeframes=[s.strip() for s in os.getenv("TIMEFRAMES", "30m,1h,4h,1d").split(",") if s.strip()],
               policy=dict(entry_bars=int(os.getenv("ENTRY_WAIT_BARS", "3")),
                           hold_bars=int(os.getenv("HOLD_BARS", "24")),
                           min_rr=float(os.getenv("MIN_RR", "1.8"))),
               min_score=int(os.getenv("MIN_AI_CONFIDENCE", "8")),
               min_samples=int(os.getenv("MIN_SAMPLES", "100")),
               threshold=float(os.getenv("SUCCESS_THRESHOLD", "0.80")),
               key=(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()),
               model=os.getenv("GEMINI_MODEL", "").strip().removeprefix("models/"),
               state_dir=ROOT / os.getenv("STATE_DIR", "state"))
    if cfg["exchange"] not in {"bingx", "binanceusdm"} or cfg["fallback"] not in {"bingx", "binanceusdm"}:
        raise ValueError("EXCHANGE_ID / FALLBACK_EXCHANGE_ID 只支援 bingx 或 binanceusdm")
    if not cfg["symbols"] or not cfg["timeframes"] or not set(cfg["timeframes"]) <= {"15m", "30m", "1h", "4h", "1d"}:
        raise ValueError("幣種或週期設定不正確")
    if (not 1 <= cfg["min_score"] <= 10 or cfg["min_samples"] < 1 or
        not 0.8 <= cfg["threshold"] < 1 or not 1 <= cfg["policy"]["entry_bars"] <= 100 or
        not 1 <= cfg["policy"]["hold_bars"] <= 1000 or
        not math.isfinite(cfg["policy"]["min_rr"]) or cfg["policy"]["min_rr"] < 1.8):
        raise ValueError("門檻設定不正確（成功率至少 0.80、R:R 至少 1.8）")
    return cfg


def safe_error(exc, cfg):
    text = str(exc)
    for secret in (cfg.get("key"), cfg.get("token")):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text[:300]


@contextlib.contextmanager
def single_instance(folder):
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / "watcher.lock").open("a+b")
    handle.seek(0)
    if not handle.read(1):
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("已有另一個盯盤程序在使用此資料庫") from None
    try:
        yield
    finally:
        handle.close()


def list_models(cfg):
    if not cfg["key"]:
        raise ValueError("未收到 GEMINI_API_KEY（或 GOOGLE_API_KEY）。GitHub Actions 必須用 env 將 secrets 注入執行步驟；只有建立 Secret 不會自動成為環境變數。")
    models, page = [], None
    while True:
        response = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                                headers={"x-goog-api-key": cfg["key"]},
                                params={"pageSize": 1000, **({"pageToken": page} if page else {})}, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Gemini 模型清單 HTTP {response.status_code}，請檢查 API Key／權限")
        body = response.json()
        models += [m["name"].removeprefix("models/") for m in body.get("models", [])
                   if "generateContent" in m.get("supportedGenerationMethods", [])]
        page = body.get("nextPageToken")
        if not page:
            return sorted(models)


class GeminiRequestError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"Gemini HTTP {status}；請檢查模型／權限／用量。")


def model_candidates(models):
    # The listing is authoritative for existence, but it does not promise vision
    # or JSON support. A small image+JSON preflight below verifies both.
    excluded = ("image", "audio", "tts", "live", "native", "embedding", "robotics", "computer-use")
    candidates = [m for m in models if m.startswith("gemini-") and "flash" in m
                  and not any(word in m.lower() for word in excluded)]
    def rank(name):
        match = re.match(r"gemini-(\d+)(?:\.(\d+))?", name)
        version = (int(match[1]), int(match[2] or 0)) if match else (0, 0)
        return ("lite" in name, any(x in name for x in ("preview", "exp")),
                -version[0], -version[1], name)
    return sorted(candidates, key=rank)


def verify_model(cfg):
    models = list_models(cfg)
    explicit = cfg["model"].lower() not in {"", "auto"}
    if explicit and not re.fullmatch(r"[A-Za-z0-9._-]+", cfg["model"]):
        raise ValueError("GEMINI_MODEL 格式不正確")
    if explicit and cfg["model"] not in models:
        raise ValueError("指定的 GEMINI_MODEL 不在此 Key 的可用清單；可改成 auto 由程式選擇")
    pin_path = cfg["state_dir"] / "model_choice.json"
    account = hashlib.sha256(cfg["key"].encode()).hexdigest()
    try:
        pinned = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pinned = {}
    candidates = [cfg["model"]] if explicit else model_candidates(models)
    if (pinned.get("account") == account and pinned.get("model") in models and
        pinned.get("vision_json_verified") is True and
        (not explicit or pinned.get("model") == cfg["model"])):
        cfg["model"] = pinned["model"]
        print(f"[Gemini] 使用已驗證模型：{cfg['model']}", flush=True)
        return cfg["model"]
    if not candidates:
        raise ValueError("此 Key 沒有可自動選用的 Flash 模型；請用 --list-models 查看並明確設定 GEMINI_MODEL")
    # A valid 16x16 PNG; the preflight consumes a small API request once per
    # newly selected model/account. Do not silently try other accounts or quota pools.
    parts = [{"text": 'Check image input and JSON output. Reply with exactly {"ready":true}.'},
             {"inlineData": {"mimeType": "image/png", "data":
              "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAI0lEQVR4nGP8//8/AymAiSTVDKMaiANMRKqDg1ENxACSQwkAVW0DHeN02ZEAAAAASUVORK5CYII="}}]
    for candidate in candidates[:5]:
        cfg["model"] = candidate
        try:
            result = call_gemini(cfg, parts)
            if result.get("ready") is not True:
                raise ValueError("看圖／JSON 測試未通過")
        except GeminiRequestError as exc:
            if not explicit and exc.status in {400, 404}:
                continue
            raise
        except ValueError:
            if not explicit:
                continue
            raise
        cfg["state_dir"].mkdir(parents=True, exist_ok=True)
        temporary = pin_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(dict(model=candidate, account=account, vision_json_verified=True)), encoding="utf-8")
        temporary.replace(pin_path)
        print(f"[Gemini] 已選定並驗證看圖／JSON：{candidate}", flush=True)
        return candidate
    raise ValueError("可用 Flash 模型均未通過看圖／JSON 測試，請查看模型權限或明確設定 GEMINI_MODEL")


def call_gemini(cfg, parts):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg['model']}:generateContent"
    for attempt in range(3):
        try:
            response = requests.post(url, headers={"x-goog-api-key": cfg["key"]},
                                     json={"contents": [{"parts": parts}],
                                           "generationConfig": {"responseMimeType": "application/json"}}, timeout=(10, 90))
        except requests.RequestException:
            if attempt == 2:
                raise RuntimeError("Gemini 連線逾時／失敗") from None
            time.sleep(2 ** (attempt + 1))
            continue
        if response.status_code in {429, 500, 502, 503, 504}:
            time.sleep(min(30, 5 * (attempt + 1)))
            continue
        if response.status_code != 200:
            raise GeminiRequestError(response.status_code)
        try:
            candidate = response.json()["candidates"][0]
            if candidate.get("finishReason") != "STOP":
                raise ValueError("回應未完整結束")
            text = "".join(p.get("text", "") for p in candidate["content"]["parts"] if not p.get("thought"))
            result = json.loads(text)
            if not isinstance(result, dict):
                raise ValueError("不是 JSON 物件")
            return result
        except (ValueError, KeyError, IndexError, TypeError):
            raise ValueError("Gemini 回應不完整或 JSON 格式不正確；略過此次訊號") from None
    raise RuntimeError("Gemini 忙碌，已停止本次請求")


def clean_candles(rows, tf_ms, now_ms):
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"] + tf_ms <= now_ms].reset_index(drop=True)
    if df.empty:
        raise ValueError("沒有已收盤 K 線")
    numeric = df.to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or (df[["open", "high", "low", "close"]] <= 0).any().any()
        or (df["volume"] < 0).any() or (df["timestamp"] % tf_ms != 0).any()
        or (df["high"] < df[["open", "close", "low"]].max(axis=1)).any()
        or (df["low"] > df[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("交易所 OHLCV 資料不合理")
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def fetch_frame(exchange, symbol, tf, tf_ms, now_ms, since=None):
    rows = []
    cursor = since
    # Paginate open paper trades after downtime; never fill BingX gaps with Binance.
    for _ in range(100):
        time.sleep(0.65)
        batch = None
        for attempt in range(3):
            try:
                batch = exchange.fetch_ohlcv(symbol, tf, since=cursor, limit=500 if cursor is not None else 201)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        if not batch:
            break
        rows.extend(batch)
        last = max(int(b[0]) for b in batch)
        if cursor is None or last + tf_ms >= now_ms or last < cursor:
            break
        new_cursor = last + tf_ms
        if cursor is not None and new_cursor <= cursor:
            break
        cursor = new_cursor
    return clean_candles(rows, tf_ms, now_ms)


def validate_shape(shape, df):
    if shape.get("has_pattern") is not True:
        if shape.get("has_pattern") is not False:
            raise ValueError("has_pattern 應為布林值")
        return None
    if shape.get("pattern_code") not in PATTERNS or shape.get("bias") not in {"偏多", "偏空"}:
        raise ValueError("型態尚未確認或型態代碼不合法")
    anchor = shape.get("pattern_start_bar")
    if isinstance(anchor, bool) or not isinstance(anchor, int) or not 0 <= anchor < min(85, len(df)) - 3:
        raise ValueError("型態起點座標不合法")
    lines = shape.get("trend_lines")
    if not isinstance(lines, list) or not 1 <= len(lines) <= 3:
        raise ValueError("缺少型態線")
    window = df.tail(85).reset_index(drop=True)
    for line in lines:
        if not isinstance(line, list) or not 2 <= len(line) <= 12:
            raise ValueError("型態線座標不正確")
        prev = -1
        for point in line:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("型態點座標不正確")
            x, y = point
            if (isinstance(x, bool) or not isinstance(x, int) or not prev < x < len(window)
                or isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y)
                or not float(window["low"].min()) * 0.85 <= y <= float(window["high"].max()) * 1.15):
                raise ValueError("型態點超出圖表範圍")
            prev = x
    return int(window["timestamp"].iloc[anchor])


def make_signal_text(signal, stats):
    m = signal["metadata"]
    start = pd.Timestamp(signal["created_ms"], unit="ms", tz="UTC").tz_convert("Asia/Taipei")
    return (f"型態觀察｜{m['symbol']} {m['timeframe']}｜{m['exchange']}\n"
            f"{m['pattern']}／{signal['bias']} · AI 評分 {signal['risk_score']}/10\n"
            f"同條件紙上紀錄：{stats['wins']}/{stats['total']}，成功率 {stats['rate']:.1%}\n"
            f"95% Wilson 下限 {stats['lower95']:.1%}（歷史統計，非本次保證）\n"
            f"限價入場 {signal['entry']:g}｜止損 {signal['stop_loss']:g}｜止盈 {signal['take_profit']:g}\n"
            f"追蹤自 {start:%m/%d %H:%M} UTC+8 起，等待 {signal['entry_bars']} 根；進場後 {signal['hold_bars']} 根內有效\n"
            f"{str(m.get('summary', ''))[:100]}\n"
            "同根碰止盈止損算失敗；逾期未止盈算未成功。未扣手續費／滑價／資金費。")


def export_signal(cfg, signal, stats):
    """Write one immutable event per signal ID for a separate notifier to consume.

    This function performs local file operations only. No notification service is
    called, even if legacy notification credentials exist in the environment.
    """
    if stats.get("eligible") is not True:
        raise ValueError("未達門檻的訊號不能輸出至 outbox")
    folder = cfg["state_dir"] / "outbox"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{signal['id']}.json"
    if target.exists():
        return target
    event = dict(schema_version=1, event="qualified_pattern", signal_id=signal["id"],
                 emitted_at_ms=int(time.time()*1000), eligible=True,
                 chart_file=f"../charts/{signal['id']}.png",
                 text=make_signal_text(signal, stats), statistics=stats, signal=signal)
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)
    return target


def cycle(cfg, ledger, chart_module, exchanges, *, scan=True):
    failures = 0
    rules_id = hashlib.sha256(json.dumps([VERSION, cfg["model"], cfg["policy"]], sort_keys=True).encode()).hexdigest()[:16]
    for symbol in cfg["symbols"]:
        for tf in cfg["timeframes"]:
            tf_ms = int(ccxt.Exchange.parse_timeframe(tf) * 1000)
            frames = {}
            # Track each actual source separately, including a fallback position left
            # from an earlier run. Never evaluate a BingX plan using Binance candles.
            needed = list(exchanges)
            for eid in needed:
                market = f"{eid}|{symbol}|{tf}"
                active = ledger.active(market)
                if not active and frames:
                    continue
                now_ms = int(time.time() * 1000)
                try:
                    if active:
                        earliest = min(s["next_ms"] for s in active)
                        if earliest < now_ms - tf_ms:
                            follow = fetch_frame(exchanges[eid], symbol, tf, tf_ms, now_ms, earliest)
                            ledger.update(market, follow[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist())
                    if not frames:
                        df = fetch_frame(exchanges[eid], symbol, tf, tf_ms, int(time.time() * 1000))
                        frames[eid] = df
                except Exception as exc:
                    failures += 1
                    print(f"[行情失敗] {market}: {safe_error(exc, cfg)}", flush=True)
            if not frames or not scan:
                continue
            eid, df = next(iter(frames.items()))
            market = f"{eid}|{symbol}|{tf}"
            candle_close = int(df["timestamp"].iloc[-1]) + tf_ms
            scan_key = f"scan|{rules_id}|{market}"
            shape_key = f"shape|{rules_id}|{market}"
            if ledger.get(scan_key, 0) >= candle_close:
                continue
            # Don't produce a current alert from old or discontinuous charts.
            if (len(df) < 85 or int(time.time()*1000) - candle_close > tf_ms
                or (df["timestamp"].diff().dropna() != tf_ms).any()):
                print(f"[略過] {market}: K 線不足／過舊／不連續", flush=True)
                continue
            if ledger.active(market):
                # Keep sampling non-overlapping paper trades; analyze again after it resolves.
                continue
            try:
                chart_module._call_gemini = lambda parts: call_gemini(cfg, parts)
                df = chart_module.add_indicators(df.copy())
                image = chart_module.render_chart_png(df, symbol, tf, dpi=110)
                # One AI analysis attempt per closed candle, including refusals and
                # malformed results. A loop must not repeatedly bill the same bar.
                ledger.set(scan_key, candle_close)
                shape = chart_module.stage1_detect_pattern_from_chart(symbol, tf, df, image)
                anchor = validate_shape(shape, df)
                if anchor is None:
                    ledger.set(shape_key, None)
                    ledger.set(scan_key, candle_close)
                    print(f"[無型態] {market}", flush=True)
                    continue
                signature = fingerprint(shape["pattern_code"], shape["bias"], anchor)
                if ledger.get(shape_key) == signature:
                    ledger.set(scan_key, candle_close)
                    continue
                plan = chart_module.stage2_score_risk_and_plan(symbol, tf, df, image, shape)
                plan["bias"] = shape["bias"]
                validate_plan(plan, cfg["policy"]["min_rr"])
                last = float(df["close"].iloc[-1])
                if (plan["entry"] > last if plan["bias"] == "偏多" else plan["entry"] < last):
                    raise ValueError("本版只接受回踩限價：偏多入場不高於現價，偏空不低於現價")
                if abs(plan["entry"] / last - 1) > 0.05:
                    raise ValueError("入場距現價超過 5%")
                ledger.set(scan_key, candle_close)
                ledger.set(shape_key, signature)
                if plan["risk_score"] < cfg["min_score"]:
                    print(f"[AI未通過] {market}: {plan['risk_score']}/10", flush=True)
                    continue
                cohort = cohort_key(eid, symbol, tf, shape["pattern_code"], plan,
                                    cfg["model"], cfg["policy"])
                stats = ledger.stats(cohort, int(time.time()*1000), cfg["min_samples"], cfg["threshold"])
                # Start at the NEXT completely unseen full candle. Never credit a
                # touch earlier in the partly formed candle while Gemini was thinking.
                start_ms = (int(time.time()*1000) // tf_ms + 1) * tf_ms
                metadata = dict(exchange=eid, symbol=symbol, timeframe=tf, pattern=shape["pattern_code"],
                                summary=str(plan.get("summary", ""))[:400],
                                signal_close_ms=candle_close, shape=shape, model=cfg["model"], gate=stats)
                signal = ledger.add(market, cohort, start_ms, tf_ms, plan, cfg["policy"], metadata)
                if signal is None:
                    continue
                folder = cfg["state_dir"] / "charts"
                folder.mkdir(parents=True, exist_ok=True)
                marked = chart_module.render_chart_png(df, symbol, tf, entry=plan["entry"],
                            stop_loss=plan["stop_loss"], take_profit=plan["take_profit"],
                            trend_lines=shape["trend_lines"], dpi=110)
                (folder / f"{signal['id']}.png").write_bytes(marked)
                print(f"[已記錄] {market} {shape['pattern_code']}: {stats['wins']}/{stats['total']}, "
                      f"Wilson下限 {stats['lower95']:.1%}, 輸出門檻={stats['eligible']}", flush=True)
                if stats["eligible"]:
                    output = export_signal(cfg, signal, stats)
                    signal["exported"] = True
                    ledger.save(signal)
                    print(f"[SIGNAL] {output.resolve()}", flush=True)
            except Exception as exc:
                failures += 1
                print(f"[分析略過] {market}: {safe_error(exc, cfg)}", flush=True)
    return failures


def run_watcher(chart_module, watch=False, interval=60, track_only=False):
    cfg = settings()
    if not track_only:
        verify_model(cfg)
    if not 15 <= interval <= 3600:
        raise ValueError("輪詢間隔應為 15–3600 秒")
    cfg["state_dir"].mkdir(parents=True, exist_ok=True)
    with single_instance(cfg["state_dir"]):
        ledger = Ledger(cfg["state_dir"] / "signals.sqlite3")
        exchanges = {eid: getattr(ccxt, eid)({"enableRateLimit": True, "timeout": 15000})
                     for eid in dict.fromkeys([cfg["exchange"], cfg["fallback"]])}
        try:
            print("模擬追蹤模式：只記錄行情與訊號，不下單。Ctrl+C 停止。", flush=True)
            while True:
                failures = cycle(cfg, ledger, chart_module, exchanges, scan=not track_only)
                print("[紀錄統計] " + json.dumps(ledger.summary(), ensure_ascii=False), flush=True)
                if not watch:
                    return 1 if failures else 0
                time.sleep(interval)
        finally:
            ledger.close()


def show_status():
    cfg = settings()
    print("設定狀態（不顯示金鑰）")
    for field in ("key",):
        print(f"{field}: {'已設定' if cfg[field] else '未設定'}")
    print(f"model: {cfg['model'] or 'auto（首次執行時自動選擇）'}")
    path = cfg["state_dir"] / "signals.sqlite3"
    if path.exists():
        ledger = Ledger(path)
        try:
            print(json.dumps(ledger.summary(), ensure_ascii=False, indent=2))
        finally:
            ledger.close()


# --- watcher_chart_source.py ---

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




# --- watcher_cli_source.py ---
def main():
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
    modes.add_argument("--check-config", action="store_true", help="驗證 Key、選定模型並測試圖片／JSON（可能產生少量 API 用量）")
    modes.add_argument("--download-only", action="store_true", help="只下載行情 CSV")
    parser.add_argument("--track-only", action="store_true", help="只結算已有模擬訊號，不呼叫 Gemini")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--output", default="market_data")
    args = parser.parse_args()
    if args.status:
        show_status()
    elif args.list_models:
        print("\n".join(list_models(cfg)))
    elif args.check_config:
        verify_model(cfg)
        print("Gemini 設定檢查通過；沒有啟動盯盤或通知。")
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
        print("啟動失敗：" + safe_error(exc, {"key": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")}))
        raise SystemExit(1)
