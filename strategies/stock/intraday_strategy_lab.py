#!/usr/bin/env python
"""
Intraday Stock Strategy Lab

Nine of the most widely used intraday stock strategies, one engine, three modes:

  MODE=analyze   Backtests every strategy on every stock in SYMBOLS with costs,
                 checks each one on the older 75% of days and again on the most
                 recent 25%, and marks it KEEP or REMOVE. The result is written
                 to strategy_leaderboard.csv next to this script.
  MODE=paper     Trades STRATEGY live on SYMBOLS with OpenAlgo in Analyze
                 (sandbox) mode. Refuses to start if Analyze mode is off.
  MODE=live      Same, with real orders. Refuses to start unless Analyze mode is
                 off, CONFIRM_LIVE=YES is set, and the strategy is marked KEEP in
                 the latest leaderboard.

The strategies
--------------
  orb              Opening range breakout: break of the first 15 minutes' range.
  vwap_pullback    Trend above/below VWAP, pullback to VWAP or EMA21, resumption.
  vwap_reversion   Price stretched 2 deviations from VWAP, rejection bar, target VWAP.
  rsi2_reversion   RSI(2) extreme in the direction of the EMA50 trend, quick snap-back.
  bb_reversion     Close back inside the Bollinger band after closing outside it.
  supertrend       Supertrend(10, 3) direction flip, stop at the Supertrend line.
  ema_cross        EMA 9/21 cross on the side of VWAP.
  gap_fill         Opening gap of 0.5-2% that starts to reverse, target yesterday's close.
  pdh_pdl_breakout Break of the previous day's high or low.

Every strategy uses its standard textbook settings. Nothing is tuned to the
data, deliberately: with nine strategies there are only nine chances for luck
to look like skill, and the recent-25% check is a real test rather than a
formality.

On win rate
-----------
The reversion strategies have the highest win rates, because their targets are
closer than their stops. A 70% win rate that loses 3 on each loss and makes 1
on each win still loses money. The leaderboard shows win rate, but KEEP and
REMOVE are decided on expectancy after costs, the average rupee result per
trade, which is the number that pays.

Running
-------
Upload at /python with exchange NSE, or run standalone:

    MODE=analyze uv run python strategies/stock/intraday_strategy_lab.py
    MODE=paper STRATEGY=orb uv run python strategies/stock/intraday_strategy_lab.py

Every setting below can be overridden with an environment variable of the same
name (set them as strategy parameters in /python).
"""

import os
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from openalgo import api


def env(name, default, cast=str):
    value = os.getenv(name)
    return default if value in (None, "") else cast(value)


def env_time(name, default):
    return datetime.strptime(env(name, default), "%H:%M").time()


def minutes_of(t):
    return t.hour * 60 + t.minute


# ---------------------------------------------------------------- settings --
API_KEY = os.getenv("OPENALGO_API_KEY", "")
HOST = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")

MODE = env("MODE", "analyze").lower()  # analyze | paper | live
STRATEGY = env("STRATEGY", "orb")
STRATEGY_TAG = env("STRATEGY_NAME", "Intraday Lab")

DEFAULT_SYMBOLS = "RELIANCE,HDFCBANK,ICICIBANK,INFY,TCS,SBIN,AXISBANK,KOTAKBANK,LT,BHARTIARTL"
SYMBOLS = [s.strip().upper() for s in env("SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]
EXCHANGE = env("EXCHANGE", "NSE")
PRODUCT = "MIS"  # intraday only

TIMEFRAME = env("TIMEFRAME", 5, int)  # minutes
HISTORY_SOURCE = env("HISTORY_SOURCE", "db")  # db (Historify) or api
ANALYZE_DAYS = env("ANALYZE_DAYS", 365, int)
MIN_DAYS = env("MIN_DAYS", 60, int)

CAPITAL_PER_TRADE = env("CAPITAL_PER_TRADE", 100000.0, float)  # rupees of stock per position
COST_PCT = env("COST_PCT", 0.10, float)  # round trip: brokerage, STT, charges, slippage

ENTRY_START = env_time("ENTRY_START", "09:30")
LAST_ENTRY = env_time("LAST_ENTRY", "14:30")
SQUARE_OFF = env_time("SQUARE_OFF", "15:10")

MAX_TRADES_PER_SYMBOL_DAY = env("MAX_TRADES_PER_SYMBOL_DAY", 2, int)
MAX_OPEN_POSITIONS = env("MAX_OPEN_POSITIONS", 3, int)  # live and paper
MAX_TRADES_PER_DAY = env("MAX_TRADES_PER_DAY", 10, int)  # live and paper
MAX_DAILY_LOSS = env("MAX_DAILY_LOSS", 5000.0, float)  # live and paper, rupees

CONFIRM_LIVE = env("CONFIRM_LIVE", "NO").upper() == "YES"
FORCE_UNVERIFIED = env("FORCE_UNVERIFIED", "NO").upper() == "YES"
OUTPUT_DIR = env("OUTPUT_DIR", os.path.dirname(os.path.abspath(__file__)))
LEADERBOARD = os.path.join(OUTPUT_DIR, "strategy_leaderboard.csv")

IST = ZoneInfo("Asia/Kolkata")


def log(msg):
    print(f"[{datetime.now(IST):%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------- data --
def fetch(client, symbol, days, source):
    end = datetime.now(IST).date()
    start = end - timedelta(days=days)
    df = client.history(
        symbol=symbol,
        exchange=EXCHANGE,
        interval="1m",
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        source=source,
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def resample(df, minutes):
    if minutes <= 1:
        return df
    out = df.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["close"])


# ------------------------------------------------------------- indicators --
def rsi(close, n):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def supertrend(high, low, close, atr, mult):
    """Classic Supertrend. Returns (line, direction) with direction +1 up, -1 down."""
    h, lo, c, a = (x.to_numpy() for x in (high, low, close, atr))
    mid = (h + lo) / 2
    upper = mid + mult * a
    lower = mid - mult * a
    n = len(c)
    fu, fl = upper.copy(), lower.copy()
    line = np.full(n, np.nan)
    direction = np.ones(n)
    for i in range(1, n):
        fu[i] = upper[i] if upper[i] < fu[i - 1] or c[i - 1] > fu[i - 1] else fu[i - 1]
        fl[i] = lower[i] if lower[i] > fl[i - 1] or c[i - 1] < fl[i - 1] else fl[i - 1]
        if direction[i - 1] > 0:
            direction[i] = -1 if c[i] < fl[i] else 1
        else:
            direction[i] = 1 if c[i] > fu[i] else -1
        line[i] = fl[i] if direction[i] > 0 else fu[i]
    return pd.Series(line, index=close.index), pd.Series(direction, index=close.index)


def add_indicators(df):
    d = df.copy()
    c, h, lo = d["close"], d["high"], d["low"]
    day = d.index.normalize()
    d["day"] = day
    d["mins"] = d.index.hour * 60 + d.index.minute

    for n in (9, 21, 50):
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # Session VWAP, and the volume-less fallback of a session TWAP.
    tp = (h + lo + c) / 3
    vol = d["volume"].fillna(0)
    cum_v = vol.groupby(day).cumsum()
    cum_pv = (tp * vol).groupby(day).cumsum()
    twap = tp.groupby(day).cumsum() / tp.groupby(day).cumcount().add(1)
    d["vwap"] = np.where(cum_v > 0, cum_pv / cum_v.replace(0, np.nan), twap)
    dev2 = (tp - d["vwap"]) ** 2
    d["vwap_sd"] = np.sqrt(dev2.groupby(day).cumsum() / dev2.groupby(day).cumcount().add(1))

    d["rsi2"] = rsi(c, 2)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    d["bb_mid"], d["bb_up"], d["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    d["st_line"], d["st_dir"] = supertrend(h, lo, c, d["atr"], 3.0)

    # Day structure: today's open, yesterday's levels, the opening range.
    daily = d.groupby(day).agg(
        day_open=("open", "first"), dh=("high", "max"), dl=("low", "min"), dc=("close", "last")
    )
    prev = daily[["dh", "dl", "dc"]].shift(1)
    d["day_open"] = daily["day_open"].reindex(day).to_numpy()
    d["pdh"] = prev["dh"].reindex(day).to_numpy()
    d["pdl"] = prev["dl"].reindex(day).to_numpy()
    d["pdc"] = prev["dc"].reindex(day).to_numpy()
    opening = d[d["mins"] < minutes_of(ENTRY_START)]
    orng = opening.groupby(opening["day"]).agg(orh=("high", "max"), orl=("low", "min"))
    d["orh"] = orng["orh"].reindex(day).to_numpy()
    d["orl"] = orng["orl"].reindex(day).to_numpy()
    d["bar_of_day"] = d.groupby(day).cumcount()
    return d


# ------------------------------------------------------------- strategies --
# Each strategy reads an indicator frame and returns, for every CLOSED bar,
# side (+1 long, -1 short, 0 none), and the stop and target it wants as
# absolute prices (NaN = the engine default: 1.5 ATR stop, 2R target).
# `once` means at most one trade per symbol per day; `hold` is the most bars a
# trade may stay open before it is closed at market.


def _result(d, long_ok, short_ok, stop_l, stop_s, tgt_l=np.nan, tgt_s=np.nan):
    side = np.where(long_ok, 1, np.where(short_ok, -1, 0))
    stop = np.where(side > 0, stop_l, np.where(side < 0, stop_s, np.nan))
    target = np.where(side > 0, tgt_l, np.where(side < 0, tgt_s, np.nan))
    return pd.DataFrame({"side": side, "stop": stop, "target": target}, index=d.index)


def s_orb(d):
    c, pc = d["close"], d["close"].shift(1)
    live = d["mins"] <= 11 * 60
    long_ok = live & (c > d["orh"]) & (pc <= d["orh"])
    short_ok = live & (c < d["orl"]) & (pc >= d["orl"])
    # Stop at the other side of the range, but never wider than 2 ATR.
    stop_l = np.maximum(d["orl"], c - 2 * d["atr"])
    stop_s = np.minimum(d["orh"], c + 2 * d["atr"])
    return _result(d, long_ok, short_ok, stop_l, stop_s)


def s_vwap_pullback(d):
    c, h, lo = d["close"], d["high"], d["low"]
    touch_l = ((lo <= d["vwap"]) | (lo <= d["ema21"])).astype(float).shift(1).rolling(3).max() > 0
    touch_s = ((h >= d["vwap"]) | (h >= d["ema21"])).astype(float).shift(1).rolling(3).max() > 0
    long_ok = (
        (c > d["vwap"]) & (d["ema9"] > d["ema21"]) & touch_l & (c > h.shift(1)) & (c > d["open"])
    )
    short_ok = (
        (c < d["vwap"]) & (d["ema9"] < d["ema21"]) & touch_s & (c < lo.shift(1)) & (c < d["open"])
    )
    stop_l = lo.rolling(3).min() - 0.1 * d["atr"]
    stop_s = h.rolling(3).max() + 0.1 * d["atr"]
    return _result(d, long_ok, short_ok, stop_l, stop_s)


def s_vwap_reversion(d):
    c, o, h, lo = d["close"], d["open"], d["high"], d["low"]
    band = 2 * d["vwap_sd"]
    enough = (d["bar_of_day"] >= 6) & (d["vwap_sd"] > 0)
    long_ok = enough & (lo < d["vwap"] - band) & (c > o)
    short_ok = enough & (h > d["vwap"] + band) & (c < o)
    return _result(
        d, long_ok, short_ok, lo - 0.5 * d["atr"], h + 0.5 * d["atr"], d["vwap"], d["vwap"]
    )


def s_rsi2_reversion(d):
    c = d["close"]
    long_ok = (c > d["ema50"]) & (d["rsi2"] < 10)
    short_ok = (c < d["ema50"]) & (d["rsi2"] > 90)
    a = d["atr"]
    return _result(d, long_ok, short_ok, c - 1.5 * a, c + 1.5 * a, c + 1.0 * a, c - 1.0 * a)


def s_bb_reversion(d):
    c, pc = d["close"], d["close"].shift(1)
    long_ok = (pc < d["bb_lo"].shift(1)) & (c > d["bb_lo"])
    short_ok = (pc > d["bb_up"].shift(1)) & (c < d["bb_up"])
    a = d["atr"]
    stop_l = d["low"].rolling(2).min() - 0.5 * a
    stop_s = d["high"].rolling(2).max() + 0.5 * a
    return _result(d, long_ok, short_ok, stop_l, stop_s, d["bb_mid"], d["bb_mid"])


def s_supertrend(d):
    flip = d["st_dir"].diff()
    long_ok = flip > 0
    short_ok = flip < 0
    c = d["close"]
    risk = (c - d["st_line"]).abs()
    return _result(d, long_ok, short_ok, d["st_line"], d["st_line"], c + 3 * risk, c - 3 * risk)


def s_ema_cross(d):
    f, s = d["ema9"], d["ema21"]
    up = (f > s) & (f.shift(1) <= s.shift(1))
    down = (f < s) & (f.shift(1) >= s.shift(1))
    c = d["close"]
    return _result(d, up & (c > d["vwap"]), down & (c < d["vwap"]), np.nan, np.nan)


def s_gap_fill(d):
    gap = (d["day_open"] - d["pdc"]) / d["pdc"] * 100
    first = d["mins"] == minutes_of(ENTRY_START) - TIMEFRAME  # the bar that closes at ENTRY_START
    c, o = d["close"], d["open"]
    short_ok = first & gap.between(0.5, 2.0) & (c < o) & (c > d["pdc"])
    long_ok = first & gap.between(-2.0, -0.5) & (c > o) & (c < d["pdc"])
    day_hi = d.groupby("day")["high"].cummax()
    day_lo = d.groupby("day")["low"].cummin()
    a = d["atr"]
    return _result(d, long_ok, short_ok, day_lo - 0.2 * a, day_hi + 0.2 * a, d["pdc"], d["pdc"])


def s_pdh_pdl_breakout(d):
    c, pc = d["close"], d["close"].shift(1)
    live = d["mins"].between(9 * 60 + 45, 13 * 60 + 30)
    long_ok = live & (c > d["pdh"]) & (pc <= d["pdh"])
    short_ok = live & (c < d["pdl"]) & (pc >= d["pdl"])
    return _result(d, long_ok, short_ok, np.nan, np.nan)


STRATEGIES = {
    "orb": {"fn": s_orb, "once": True, "hold": 60, "style": "breakout"},
    "vwap_pullback": {"fn": s_vwap_pullback, "once": False, "hold": 24, "style": "trend"},
    "vwap_reversion": {"fn": s_vwap_reversion, "once": False, "hold": 12, "style": "reversion"},
    "rsi2_reversion": {"fn": s_rsi2_reversion, "once": False, "hold": 6, "style": "reversion"},
    "bb_reversion": {"fn": s_bb_reversion, "once": False, "hold": 12, "style": "reversion"},
    "supertrend": {"fn": s_supertrend, "once": False, "hold": 60, "style": "trend"},
    "ema_cross": {"fn": s_ema_cross, "once": False, "hold": 36, "style": "trend"},
    "gap_fill": {"fn": s_gap_fill, "once": True, "hold": 60, "style": "reversion"},
    "pdh_pdl_breakout": {"fn": s_pdh_pdl_breakout, "once": True, "hold": 60, "style": "breakout"},
}


def plan_levels(side, entry, stop, target, atr):
    """Sanitise a strategy's stop/target around the actual entry price.

    Returns (stop, target) or None when the trade no longer makes sense, for
    example the target was already passed by the time of the fill.
    """
    if not np.isfinite(atr) or atr <= 0:
        return None
    if not np.isfinite(stop) or side * (entry - stop) < 0.1 * atr:
        stop = entry - side * 1.5 * atr
    risk = side * (entry - stop)
    if not np.isfinite(target):
        target = entry + side * 2 * risk
    if side * (target - entry) < 0.1 * atr:
        return None
    return stop, target


# ---------------------------------------------------------------- backtest --
def simulate(d, sig, spec, symbol):
    o, h, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = d["atr"].to_numpy()
    day = d["day"].to_numpy()
    mins = d["mins"].to_numpy()
    side_a, stop_a, tgt_a = (sig[k].to_numpy() for k in ("side", "stop", "target"))
    start_m, last_m, sq_m = minutes_of(ENTRY_START), minutes_of(LAST_ENTRY), minutes_of(SQUARE_OFF)
    n = len(d)

    rows = []
    next_free = 0
    counts = {}
    for i in np.flatnonzero(side_a):
        j = i + 1
        if i < next_free or j >= n or day[j] != day[i] or i < 60:
            continue
        # The fill happens at the open of bar j; it must be inside entry hours.
        if not (start_m <= mins[j] <= last_m):
            continue
        key = day[i]
        if counts.get(key, 0) >= (1 if spec["once"] else MAX_TRADES_PER_SYMBOL_DAY):
            continue
        side = int(side_a[i])
        entry = o[j]
        levels = plan_levels(side, entry, stop_a[i], tgt_a[i], atr[i])
        if levels is None:
            continue
        stop, target = levels
        counts[key] = counts.get(key, 0) + 1

        reason, exit_price, k = None, None, j
        while k < n and day[k] == day[i]:
            # Worst case inside a bar: the stop is checked before the target.
            if side * (lo[k] if side > 0 else h[k]) <= side * stop:
                reason, exit_price = "STOP", stop
            elif side * (h[k] if side > 0 else lo[k]) >= side * target:
                reason, exit_price = "TARGET", target
            elif k - j + 1 >= spec["hold"]:
                reason, exit_price = "TIME", c[k]
            elif mins[k] >= sq_m:
                reason, exit_price = "SQUARE_OFF", c[k]
            if reason:
                break
            k += 1
        if reason is None:
            k = min(k, n - 1)
            reason, exit_price = "EOD", c[k]

        qty = max(1, int(CAPITAL_PER_TRADE // entry))
        gross_pct = side * (exit_price / entry - 1) * 100
        net_pct = gross_pct - COST_PCT
        rows.append(
            {
                "symbol": symbol,
                "day": pd.Timestamp(day[i]).date(),
                "side": side,
                "reason": reason,
                "gross_pct": gross_pct,
                "net_pct": net_pct,
                "rupees": net_pct / 100 * entry * qty,
            }
        )
        next_free = k + 1
    return rows


def stats(t):
    if t.empty:
        return {
            "trades": 0,
            "win": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "pf": 0.0,
            "exp_pct": 0.0,
            "exp_rs": 0.0,
            "gross_pct": 0.0,
        }
    r = t["net_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "trades": len(t),
        "win": float((r > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "pf": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "exp_pct": float(r.mean()),
        "exp_rs": float(t["rupees"].mean()),
        "gross_pct": float(t["gross_pct"].mean()),
    }


def analyze(client):
    frames = {}
    for sym in SYMBOLS:
        raw = fetch(client, sym, ANALYZE_DAYS, HISTORY_SOURCE)
        if raw is None:
            log(
                f"{sym}: no history ({HISTORY_SOURCE}). Download it in Historify or check the symbol."
            )
            continue
        d = add_indicators(resample(raw, TIMEFRAME))
        n_days = d["day"].nunique()
        if n_days < MIN_DAYS:
            log(f"{sym}: only {n_days} days, need {MIN_DAYS}. Skipped.")
            continue
        frames[sym] = d
        log(f"{sym}: {n_days} days loaded")
    if not frames:
        log("No usable history for any symbol. Nothing to analyze.")
        return

    all_days = sorted({dd for d in frames.values() for dd in d["day"].dt.date.unique()})
    cut = all_days[int(len(all_days) * 0.75)]
    log(f"Earlier period to {cut}, recent period from {cut} ({len(all_days)} days in total)")
    log(
        f"Costs {COST_PCT}% per round trip, {CAPITAL_PER_TRADE:.0f} rupees per position, {TIMEFRAME}m bars"
    )

    board, per_symbol = [], []
    for name, spec in STRATEGIES.items():
        rows = []
        for sym, d in frames.items():
            rows += simulate(d, spec["fn"](d), spec, sym)
        t = pd.DataFrame(rows)
        if t.empty:
            board.append(
                {
                    "strategy": name,
                    "style": spec["style"],
                    "verdict": "REMOVE",
                    "reason": "no trades",
                }
            )
            continue
        s_all = stats(t)
        s_old = stats(t[t["day"] < cut])
        s_new = stats(t[t["day"] >= cut])
        by_sym = t.groupby("symbol")["net_pct"].mean()
        share = float((by_sym > 0).mean())
        for sym, v in by_sym.items():
            per_symbol.append(
                {
                    "strategy": name,
                    "symbol": sym,
                    "trades": int((t["symbol"] == sym).sum()),
                    "exp_pct": round(v, 3),
                }
            )

        reasons = []
        if s_all["trades"] < 60:
            reasons.append(f"too few trades ({s_all['trades']})")
        if s_old["exp_pct"] <= 0:
            reasons.append("loses in earlier period")
        if s_new["exp_pct"] <= 0:
            reasons.append("loses in recent period")
        if s_all["pf"] < 1.1:
            reasons.append(f"profit factor {s_all['pf']:.2f} below 1.1")
        if share < 0.5:
            reasons.append(f"profitable on only {share:.0%} of stocks")
        if s_all["gross_pct"] > 0 >= s_all["exp_pct"]:
            reasons.append("edge smaller than costs")
        board.append(
            {
                "strategy": name,
                "style": spec["style"],
                "verdict": "KEEP" if not reasons else "REMOVE",
                "trades": s_all["trades"],
                "win_rate": round(s_all["win"] * 100, 1),
                "avg_win_pct": round(s_all["avg_win"], 3),
                "avg_loss_pct": round(s_all["avg_loss"], 3),
                "profit_factor": round(s_all["pf"], 2),
                "before_costs_pct": round(s_all["gross_pct"], 3),
                "per_trade_pct": round(s_all["exp_pct"], 3),
                "per_trade_rs": round(s_all["exp_rs"], 0),
                "earlier_pct": round(s_old["exp_pct"], 3),
                "recent_pct": round(s_new["exp_pct"], 3),
                "stocks_profitable": f"{share:.0%}",
                "reason": "; ".join(reasons) or "passes every check",
            }
        )

    lb = pd.DataFrame(board).sort_values(
        ["verdict", "per_trade_pct"], ascending=[True, False], na_position="last"
    )
    try:
        lb.to_csv(LEADERBOARD, index=False)
        pd.DataFrame(per_symbol).to_csv(
            os.path.join(OUTPUT_DIR, "strategy_by_symbol.csv"), index=False
        )
    except OSError as e:
        log(f"Could not write results: {e}")

    log("-" * 150)
    log(
        f"{'strategy':<17}{'verdict':<8}{'trades':>7}{'win%':>7}{'avg win%':>10}{'avg loss%':>11}{'PF':>6}"
        f"{'pre-cost%':>11}{'net%':>8}{'net Rs':>8}{'earlier%':>10}{'recent%':>9}{'stocks+':>9}"
    )
    for _, r in lb.iterrows():
        if pd.isna(r.get("trades")):
            log(f"{r['strategy']:<17}{r['verdict']:<8} {r['reason']}")
            continue
        log(
            f"{r['strategy']:<17}{r['verdict']:<8}{int(r['trades']):>7}{r['win_rate']:>7.1f}{r['avg_win_pct']:>10.3f}"
            f"{r['avg_loss_pct']:>11.3f}{r['profit_factor']:>6.2f}{r['before_costs_pct']:>11.3f}{r['per_trade_pct']:>8.3f}"
            f"{r['per_trade_rs']:>8.0f}{r['earlier_pct']:>10.3f}{r['recent_pct']:>9.3f}{r['stocks_profitable']:>9}"
        )
    log("-" * 150)
    for _, r in lb[lb["verdict"] == "REMOVE"].iterrows():
        log(f"REMOVE {r['strategy']}: {r['reason']}")
    kept = lb[lb["verdict"] == "KEEP"]["strategy"].tolist()
    if kept:
        log(
            f"KEEP: {', '.join(kept)}. Next step: MODE=paper STRATEGY=<name> with Analyze mode ON for a few weeks."
        )
    else:
        log("No strategy survived after costs. Do not trade any of them live on these stocks.")
    log(f"Leaderboard written to {LEADERBOARD}")


# ------------------------------------------------------------ paper / live --
class Position:
    def __init__(self, symbol, side, qty, stop, target, hold_bars, entry_ref):
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.stop = stop
        self.target = target
        self.deadline = datetime.now(IST) + timedelta(minutes=hold_bars * TIMEFRAME)
        self.entry_ref = entry_ref
        self.entry_price = None
        self.exiting = False  # claim flag: one exit decision per position


class Trader:
    def __init__(self, client, name):
        self.client = client
        self.name = name
        self.spec = STRATEGIES[name]
        self.lock = threading.Lock()
        self.ltp = {}
        self.positions = {}
        self.day = None
        self.trades_today = 0
        self.pnl_today = 0.0
        self.symbol_trades = {}

    # --- safety checks before anything is sent
    def preflight(self):
        res = self.client.analyzerstatus()
        if res.get("status") != "success":
            log(f"Could not read OpenAlgo's Analyze mode, not starting: {res.get('message', res)}")
            return False
        sandbox = bool((res.get("data") or {}).get("analyze_mode"))
        if MODE == "paper" and not sandbox:
            log(
                "MODE=paper but OpenAlgo is in LIVE mode. Turn Analyze mode on first. Not starting."
            )
            return False
        if MODE == "live":
            if sandbox:
                log(
                    "MODE=live but OpenAlgo is in Analyze mode, so orders would be simulated. Not starting."
                )
                return False
            if not CONFIRM_LIVE:
                log("MODE=live places real orders. Set CONFIRM_LIVE=YES to confirm. Not starting.")
                return False
            verdict = self.leaderboard_verdict()
            if verdict != "KEEP" and not FORCE_UNVERIFIED:
                log(
                    f"{self.name} is marked {verdict or 'untested'} in the leaderboard. Run MODE=analyze and "
                    "paper trade it first. Not starting."
                )
                return False
        log(
            f"{'PAPER (Analyze mode)' if sandbox else 'LIVE'} trading {self.name} on {', '.join(SYMBOLS)}"
        )
        return True

    def leaderboard_verdict(self):
        try:
            lb = pd.read_csv(LEADERBOARD)
            row = lb[lb["strategy"] == self.name]
            return None if row.empty else str(row["verdict"].iloc[0])
        except (OSError, pd.errors.ParserError, KeyError):
            return None

    # --- market data
    def on_tick(self, data):
        try:
            symbol = data.get("symbol") or data["data"].get("symbol")
            ltp = float(data["data"]["ltp"])
        except (AttributeError, KeyError, TypeError, ValueError):
            return
        hit = None
        with self.lock:
            self.ltp[symbol] = ltp
            p = self.positions.get(symbol)
            if p is None or p.exiting:
                return
            if p.side * (ltp - p.stop) <= 0:
                hit = "STOP"
            elif p.side * (ltp - p.target) >= 0:
                hit = "TARGET"
            if hit is None:
                return
            p.exiting = True
        self.exit(symbol, hit)

    # --- orders
    def fill_price(self, order_id):
        for _ in range(5):
            time.sleep(0.6)
            res = self.client.orderstatus(order_id=order_id, strategy=STRATEGY_TAG)
            price = float((res.get("data") or {}).get("average_price") or 0)
            if price > 0:
                return price
        return None

    def enter(self, symbol, side, stop, target, atr, ref):
        levels = plan_levels(side, ref, stop, target, atr)
        if levels is None:
            return
        stop, target = levels
        qty = max(1, int(CAPITAL_PER_TRADE // ref))
        res = self.client.placeorder(
            strategy=STRATEGY_TAG,
            symbol=symbol,
            action="BUY" if side > 0 else "SELL",
            exchange=EXCHANGE,
            price_type="MARKET",
            product=PRODUCT,
            quantity=qty,
        )
        if res.get("status") != "success":
            log(f"{symbol}: entry refused, staying flat: {res.get('message', res)}")
            return
        p = Position(symbol, side, qty, stop, target, self.spec["hold"], ref)
        with self.lock:
            self.positions[symbol] = p
            self.trades_today += 1
            self.symbol_trades[symbol] = self.symbol_trades.get(symbol, 0) + 1
        p.entry_price = self.fill_price(res["orderid"])
        log(
            f"ENTER {'LONG' if side > 0 else 'SHORT'} {symbol} x{qty} at {p.entry_price or ref} | "
            f"stop {stop:.2f} target {target:.2f}"
        )

    def exit(self, symbol, reason):
        with self.lock:
            p = self.positions.get(symbol)
        if p is None:
            return
        res = self.client.placeorder(
            strategy=STRATEGY_TAG,
            symbol=symbol,
            action="SELL" if p.side > 0 else "BUY",
            exchange=EXCHANGE,
            price_type="MARKET",
            product=PRODUCT,
            quantity=p.qty,
        )
        if res.get("status") != "success":
            # Still held: release the claim so the next tick or loop retries.
            log(f"{symbol}: exit refused ({reason}), will retry: {res.get('message', res)}")
            with self.lock:
                p.exiting = False
            return
        exit_price = self.fill_price(res["orderid"])
        with self.lock:
            last = self.ltp.get(symbol, p.entry_ref)
            self.positions.pop(symbol, None)
        entry = p.entry_price or p.entry_ref
        pnl = p.side * ((exit_price or last) - entry) * p.qty
        with self.lock:
            self.pnl_today += pnl
        log(
            f"EXIT {reason} {symbol} {entry} -> {exit_price or last} | P&L {pnl:.0f} | today {self.pnl_today:.0f}"
        )

    def claim_exit(self, symbol, reason):
        with self.lock:
            p = self.positions.get(symbol)
            if p is None or p.exiting:
                return
            p.exiting = True
        self.exit(symbol, reason)

    # --- signals
    def evaluate(self, now):
        with self.lock:
            open_count = len(self.positions)
            can_trade = self.trades_today < MAX_TRADES_PER_DAY and self.pnl_today > -MAX_DAILY_LOSS
            busy = set(self.positions)
        if not can_trade or not (ENTRY_START <= now.time() <= LAST_ENTRY):
            return
        limit = 1 if self.spec["once"] else MAX_TRADES_PER_SYMBOL_DAY
        for symbol in SYMBOLS:
            if open_count >= MAX_OPEN_POSITIONS:
                return
            if symbol in busy or self.symbol_trades.get(symbol, 0) >= limit:
                continue
            raw = fetch(self.client, symbol, 5, "api")
            if raw is None:
                continue
            bars = resample(raw, TIMEFRAME)
            now_minute = pd.Timestamp(now).floor("min")
            bars = bars[
                bars.index + pd.Timedelta(minutes=TIMEFRAME) <= now_minute
            ]  # closed bars only
            if len(bars) < 60 or bars.index[-1].date() != now.date():
                continue
            d = add_indicators(bars)
            sig = self.spec["fn"](d).iloc[-1]
            if sig["side"] == 0:
                continue
            ref = self.ltp.get(symbol) or float(d["close"].iloc[-1])
            self.enter(
                symbol, int(sig["side"]), sig["stop"], sig["target"], float(d["atr"].iloc[-1]), ref
            )
            open_count += 1

    def run(self):
        if not self.preflight():
            return
        self.client.connect()
        self.client.subscribe_ltp(
            [{"exchange": EXCHANGE, "symbol": s} for s in SYMBOLS], on_data_received=self.on_tick
        )
        last_bar = None
        try:
            while True:
                now = datetime.now(IST)
                if now.date() != self.day:
                    self.day, self.trades_today, self.pnl_today, self.symbol_trades = (
                        now.date(),
                        0,
                        0.0,
                        {},
                    )

                with self.lock:
                    held = list(self.positions.values())
                for p in held:
                    if now.time() >= SQUARE_OFF:
                        self.claim_exit(p.symbol, "SQUARE_OFF")
                    elif now >= p.deadline:
                        self.claim_exit(p.symbol, "TIME")

                if now.time() >= SQUARE_OFF and not self.positions:
                    log("Square-off time reached and flat. Done for the day.")
                    break

                bar_close = now.replace(second=0, microsecond=0)
                if bar_close.minute % TIMEFRAME == 0 and now.second >= 3 and bar_close != last_bar:
                    last_bar = bar_close
                    self.evaluate(now)
                time.sleep(1)
        finally:
            for symbol in list(self.positions):
                self.claim_exit(symbol, "SHUTDOWN")
            self.client.disconnect()


def main():
    if not API_KEY:
        log("OPENALGO_API_KEY is not set. Generate one at /apikey.")
        return
    client = api(api_key=API_KEY, host=HOST, ws_url=WS_URL)
    if MODE == "analyze":
        analyze(client)
    elif MODE in ("paper", "live"):
        if STRATEGY not in STRATEGIES:
            log(f"Unknown STRATEGY '{STRATEGY}'. Choose one of: {', '.join(STRATEGIES)}")
            return
        Trader(client, STRATEGY).run()
    else:
        log("MODE must be analyze, paper or live.")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
