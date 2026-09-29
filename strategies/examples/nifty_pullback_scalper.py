#!/usr/bin/env python
"""
NIFTY Trend-Pullback Scalper (ATM option buying)

Idea
----
Scalps that work are taken WITH the intraday trend, after a pause, at a moment
when the trend resumes. Scalps that fail are usually taken in chop, against the
trend, or chasing an extended move. The rules below try to keep only the first
kind:

  1. Trend      fast EMA above slow EMA, slow EMA rising, close above the
                session average price (mirror image for shorts).
  2. Pullback   one of the last PULLBACK_BARS bars dipped to the fast EMA.
  3. Trigger    the just-closed bar closes beyond the previous bar's high (low),
                i.e. the trend is resuming now.
  4. Regime     ATR above MIN_ATR (skip dead, choppy tape) and price not more
                than MAX_EXTENSION_ATR away from the slow EMA (skip chasing).
  5. Time       only in the liquid windows (default 09:30-11:30, 13:30-14:45).

Bars are TIMEFRAME minutes (1, 3, 5, ...), built from 1-minute history. A long
signal buys the ATM CE, a short signal buys the ATM PE. The signal and all
exits are measured on the NIFTY index itself, so the backtest, the optimizer
and the live run make exactly the same decisions:

  stop     SL_ATR x ATR from the entry
  target   RR x stop distance
  trail    after +1R the stop moves to breakeven, then trails TRAIL_ATR x ATR
           behind the best price seen (TRAIL_ATR=0 turns both off)
  time     a scalp that has not worked in MAX_HOLD_BARS bars is closed
  day      MAX_TRADES_PER_DAY, MAX_CONSECUTIVE_LOSSES and MAX_DAILY_LOSS stop
           new entries for the day; everything is squared off at SQUARE_OFF

Modes
-----
  MODE=backtest   one run of the current settings, with and without costs.
  MODE=optimize   searches a fixed grid of settings for an edge, and refuses to
                  call anything an edge unless it survives days the search
                  never saw. See optimize() for the exact gates. Schedule it
                  daily after the close: each run adds the newest days, and a
                  winner that keeps passing on fresh days is the real evidence.
  MODE=live       trades the current settings. Start in Analyze (sandbox) mode.

Edge is not a promise
---------------------
Searching many settings on the same data always finds one that looks good by
luck. That is why the optimizer holds back the most recent days, looks at them
once for the single winner, and prints NO EDGE when that winner fails there.
The backtest also converts index points to option premium with a fixed DELTA,
which is an approximation: real ATM premiums lose theta and move with IV.

Running
-------
Upload it at /python (exchange NFO), or run it standalone:

    MODE=optimize uv run python strategies/examples/nifty_pullback_scalper.py
    MODE=backtest uv run python strategies/examples/nifty_pullback_scalper.py
    MODE=live     uv run python strategies/examples/nifty_pullback_scalper.py

Every setting below can be overridden with an environment variable of the same
name (set them as strategy parameters in /python).
"""

import itertools
import json
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


# ---------------------------------------------------------------- settings --
API_KEY = os.getenv("OPENALGO_API_KEY", "")
HOST = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")
WS_URL = os.getenv("WEBSOCKET_URL", "ws://127.0.0.1:8765")

MODE = env("MODE", "backtest").lower()  # backtest | optimize | live
STRATEGY = env("STRATEGY_NAME", "NIFTY Pullback Scalper")

UNDERLYING = env("UNDERLYING", "NIFTY")
INDEX_EXCHANGE = env("INDEX_EXCHANGE", "NSE_INDEX")
OPTION_EXCHANGE = env("OPTION_EXCHANGE", "NFO")
PRODUCT = env("PRODUCT", "MIS")  # MIS or NRML, never CNC for options
LOTS = env("LOTS", 1, int)
OFFSET = env("OFFSET", "ATM")

TIMEFRAME = env("TIMEFRAME", 1, int)  # bar size in minutes
EMA_FAST = env("EMA_FAST", 9, int)
EMA_SLOW = env("EMA_SLOW", 21, int)
ATR_LEN = env("ATR_LEN", 14, int)
SLOPE_BARS = env("SLOPE_BARS", 5, int)
PULLBACK_BARS = env("PULLBACK_BARS", 3, int)
MIN_ATR = env("MIN_ATR", 6.0, float)  # index points on TIMEFRAME bars
MAX_EXTENSION_ATR = env("MAX_EXTENSION_ATR", 1.5, float)

SL_ATR = env("SL_ATR", 1.2, float)
RR = env("RR", 1.5, float)
TRAIL_ATR = env("TRAIL_ATR", 1.0, float)
MAX_HOLD_BARS = env("MAX_HOLD_BARS", 10, int)

WINDOWS = [
    (env_time("WINDOW1_START", "09:30"), env_time("WINDOW1_END", "11:30")),
    (env_time("WINDOW2_START", "13:30"), env_time("WINDOW2_END", "14:45")),
]
SQUARE_OFF = env_time("SQUARE_OFF", "15:10")

MAX_TRADES_PER_DAY = env("MAX_TRADES_PER_DAY", 6, int)
MAX_CONSECUTIVE_LOSSES = env("MAX_CONSECUTIVE_LOSSES", 3, int)
MAX_DAILY_LOSS = env("MAX_DAILY_LOSS", 5000.0, float)  # rupees, all lots

# Backtest-only conversion from index points to premium.
DELTA = env("DELTA", 0.5, float)
COST_PREMIUM_POINTS = env("COST_PREMIUM_POINTS", 1.5, float)  # per round trip
BACKTEST_DAYS = env("BACKTEST_DAYS", 60, int)
BACKTEST_LOT_SIZE = env("BACKTEST_LOT_SIZE", 65, int)
HISTORY_SOURCE = env("HISTORY_SOURCE", "api")  # api or db (Historify)

# Optimizer.
OPT_DAYS = env("OPT_DAYS", 365, int)
OPT_MIN_DAYS = env("OPT_MIN_DAYS", 40, int)
OPT_HOLDOUT_FRACTION = env("OPT_HOLDOUT_FRACTION", 0.25, float)
OPT_FOLDS = env("OPT_FOLDS", 4, int)
OPT_OUTPUT_DIR = env("OPT_OUTPUT_DIR", os.path.dirname(os.path.abspath(__file__)))

IST = ZoneInfo("Asia/Kolkata")


def log(msg):
    print(f"[{datetime.now(IST):%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------- data --
def fetch_history(client, days, source=HISTORY_SOURCE):
    end = datetime.now(IST).date()
    start = end - timedelta(days=days)
    df = client.history(
        symbol=UNDERLYING,
        exchange=INDEX_EXCHANGE,
        interval="1m",
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        source=source,
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        log(f"No history returned: {df}")
        return None
    return df[["open", "high", "low", "close"]]


def resample(df, minutes):
    """1-minute bars to `minutes`-minute bars, labelled by their start time.

    Bars are aligned to midnight; 09:15 is a multiple of 1, 3, 5 and 15, so
    every bar starts on the session grid.
    """
    if minutes <= 1:
        return df
    out = df.resample(f"{minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return out.dropna()


# ------------------------------------------------------------- indicators --
def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / ATR_LEN, adjust=False).mean()

    # Index bars carry no volume, so the session anchor is the running mean
    # of the typical price since the open (a session TWAP).
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.date
    df["session_avg"] = typical.groupby(day).cumsum() / typical.groupby(day).cumcount().add(1)
    return df


def in_window(t):
    return any(start <= t <= end for start, end in WINDOWS)


def compute_signals(df):
    """+1 long, -1 short, 0 nothing, for each CLOSED bar of an indicator frame."""
    c, o, h, lo = df["close"], df["open"], df["high"], df["low"]
    ef, es, atr, sa = df["ema_fast"], df["ema_slow"], df["atr"], df["session_avg"]

    minutes = df.index.hour * 60 + df.index.minute
    window = np.zeros(len(df), dtype=bool)
    for start, end in WINDOWS:
        window |= (minutes >= start.hour * 60 + start.minute) & (
            minutes <= end.hour * 60 + end.minute
        )

    # Every lookback must stay inside today's session: yesterday's close is
    # not a pullback and not a slope.
    day = pd.Series(df.index.normalize().asi8, index=df.index)
    same_day = day == day.shift(max(PULLBACK_BARS, SLOPE_BARS))

    regime = (atr >= MIN_ATR) & ((c - es).abs() <= MAX_EXTENSION_ATR * atr)
    slope = es - es.shift(SLOPE_BARS)

    long_pullback = (lo <= ef).astype(float).shift(1).rolling(PULLBACK_BARS).max() > 0
    short_pullback = (h >= ef).astype(float).shift(1).rolling(PULLBACK_BARS).max() > 0
    long_ok = (ef > es) & (slope > 0) & (c > sa) & long_pullback & (c > h.shift(1)) & (c > o)
    short_ok = (ef < es) & (slope < 0) & (c < sa) & short_pullback & (c < lo.shift(1)) & (c < o)

    base = window & same_day.to_numpy() & regime.to_numpy()
    sig = np.where(base & long_ok.to_numpy(), 1, np.where(base & short_ok.to_numpy(), -1, 0))
    sig[: max(EMA_SLOW, ATR_LEN, SLOPE_BARS, PULLBACK_BARS) + 1] = 0
    return sig


# ------------------------------------------------------------ exit engine --
class Trade:
    """Exit rules on index prices. Shared by the backtest and the live run."""

    def __init__(self, side, entry, atr, entry_time):
        self.side = side
        self.sign = 1 if side == "LONG" else -1
        self.entry = entry
        self.atr = atr
        self.entry_time = entry_time
        self.risk = SL_ATR * atr
        self.stop = entry - self.sign * self.risk
        self.target = entry + self.sign * RR * self.risk
        self.best = entry

    def update(self, price):
        """Feed a price; return an exit reason or None."""
        if self.sign * (price - self.best) > 0:
            self.best = price
        gain = self.sign * (self.best - self.entry)
        if TRAIL_ATR > 0 and gain >= self.risk:
            trail = self.best - self.sign * TRAIL_ATR * self.atr
            be_or_trail = max(self.entry, trail) if self.sign > 0 else min(self.entry, trail)
            if self.sign * (be_or_trail - self.stop) > 0:
                self.stop = be_or_trail
        if self.sign * (price - self.stop) <= 0:
            if self.stop == self.entry:
                return "BREAKEVEN"
            return "TRAIL" if self.sign * (self.stop - self.entry) > 0 else "STOP"
        if self.sign * (price - self.target) >= 0:
            return "TARGET"
        return None

    def points(self, exit_price):
        return self.sign * (exit_price - self.entry)


class DayGuard:
    def __init__(self):
        self.day = None
        self.reset(None)

    def reset(self, day):
        self.day = day
        self.trades = 0
        self.losses_in_row = 0
        self.pnl = 0.0

    def roll(self, day):
        if day != self.day:
            self.reset(day)

    def can_trade(self):
        return (
            self.trades < MAX_TRADES_PER_DAY
            and self.losses_in_row < MAX_CONSECUTIVE_LOSSES
            and self.pnl > -MAX_DAILY_LOSS
        )

    def record(self, pnl):
        self.trades += 1
        self.pnl += pnl
        self.losses_in_row = self.losses_in_row + 1 if pnl < 0 else 0


# --------------------------------------------------------------- backtest --
def simulate(df, sig):
    """Replay signals on an indicator frame; return one row per trade."""
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    day = df.index.normalize().asi8
    minutes = df.index.hour * 60 + df.index.minute
    square_off = SQUARE_OFF.hour * 60 + SQUARE_OFF.minute
    lot_value = BACKTEST_LOT_SIZE * LOTS
    n = len(df)

    guard = DayGuard()
    rows = []
    next_free = 0
    for i in np.flatnonzero(sig):
        if i < next_free or i + 1 >= n:
            continue
        guard.roll(day[i])
        if not guard.can_trade():
            continue
        # Signal on bar i's close, filled at bar i+1's open.
        j = i + 1
        if day[j] != day[i]:
            continue
        side = "LONG" if sig[i] > 0 else "SHORT"
        trade = Trade(side, o[j], atr[i], df.index[j])
        reason, exit_price = None, None
        k = j
        while k < n and day[k] == day[i]:
            # Worst case inside a bar: the adverse extreme comes first.
            adverse, favourable = (lo[k], h[k]) if side == "LONG" else (h[k], lo[k])
            for price in (adverse, favourable, c[k]):
                reason = trade.update(price)
                if reason:
                    exit_price = trade.target if reason == "TARGET" else trade.stop
                    break
            if reason:
                break
            if k - j + 1 >= MAX_HOLD_BARS:
                reason, exit_price = "TIME", c[k]
                break
            if minutes[k] >= square_off:
                reason, exit_price = "SQUARE_OFF", c[k]
                break
            k += 1
        if reason is None:
            k = min(k, n - 1)
            reason, exit_price = "EOD", c[k]

        index_pts = trade.points(exit_price)
        gross = index_pts * DELTA * lot_value
        rupees = (index_pts * DELTA - COST_PREMIUM_POINTS) * lot_value
        guard.record(rupees)
        rows.append(
            {
                "day": df.index[i].date(),
                "entry_time": trade.entry_time,
                "side": side,
                "reason": reason,
                "index_pts": index_pts,
                "gross": gross,
                "rupees": rupees,
            }
        )
        next_free = k + 1
    return pd.DataFrame(rows)


def build_frame(raw):
    return add_indicators(resample(raw, TIMEFRAME))


def backtest(client):
    raw = fetch_history(client, BACKTEST_DAYS)
    if raw is None:
        return
    df = build_frame(raw)
    log(f"Backtesting {len(df)} bars of {TIMEFRAME}m, {df.index[0]} to {df.index[-1]}")
    trades = simulate(df, compute_signals(df))
    if trades.empty:
        log("No trades. Loosen MIN_ATR or widen the windows, then re-test.")
        return
    report(trades)


def stats(t):
    """Summary numbers for a trade list, in rupees after costs."""
    if t.empty:
        return {
            "trades": 0,
            "win": 0.0,
            "pf": 0.0,
            "exp": 0.0,
            "gross_exp": 0.0,
            "net": 0.0,
            "dd": 0.0,
        }
    r = t["rupees"]
    gross_win = r[r > 0].sum()
    gross_loss = -r[r <= 0].sum()
    equity = r.cumsum()
    return {
        "trades": len(t),
        "win": float((r > 0).mean()),
        "pf": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "exp": float(r.mean()),
        "gross_exp": float(t["gross"].mean()),
        "net": float(r.sum()),
        "dd": float((equity - equity.cummax()).min()),
    }


def summarize(t, label):
    s = stats(t)
    wins = t[t["rupees"] > 0]["rupees"]
    losses = t[t["rupees"] <= 0]["rupees"]
    log(
        f"{label:<14} trades {s['trades']:>4} | win {s['win']:6.1%} | "
        f"avg win {wins.mean() if len(wins) else 0:8.0f} | "
        f"avg loss {losses.mean() if len(losses) else 0:8.0f} | "
        f"PF {s['pf']:5.2f} | before costs {s['gross_exp']:6.0f}/trade | "
        f"after costs {s['exp']:6.0f}/trade | net {s['net']:9.0f} | max DD {s['dd']:8.0f}"
    )
    return s["exp"]


def report(t):
    days = sorted(t["day"].unique())
    cut = days[int(len(days) * 0.7)] if len(days) >= 5 else None
    log("-" * 130)
    log(
        f"{TIMEFRAME}m bars | costs {COST_PREMIUM_POINTS} premium pts/round trip, "
        f"delta {DELTA}, lot {BACKTEST_LOT_SIZE} x {LOTS}"
    )
    summarize(t, "ALL")
    for side in ("LONG", "SHORT"):
        if (t["side"] == side).any():
            summarize(t[t["side"] == side], side)
    verdict = None
    if cut is not None:
        first = summarize(t[t["day"] < cut], "FIRST 70%")
        last = summarize(t[t["day"] >= cut], "LAST 30%")
        verdict = first > 0 and last > 0
    log("Exit reasons: " + ", ".join(f"{k} {v}" for k, v in t["reason"].value_counts().items()))
    log("-" * 130)
    if verdict is None:
        log("Too few days to judge. Raise BACKTEST_DAYS.")
    elif verdict:
        log(
            "Positive after costs in both halves. Next step: run MODE=live in Analyze mode for a few weeks."
        )
    else:
        log("NO EDGE on this data after costs. Do not trade this live as configured.")


# -------------------------------------------------------------- optimizer --
# A fixed, deliberately small grid. Every extra option is another chance for
# luck to look like skill, so the grid only holds choices that change the
# character of the trade: bar size, trend speed, chop filter, stop width,
# reward, trailing and patience. MIN_ATR_MULT scales the chop filter with the
# bar size (ATR grows roughly with the square root of the bar length).
GRID = {
    "TIMEFRAME": [1, 3, 5],
    "EMAS": [(5, 13), (9, 21), (20, 50)],
    "MIN_ATR_MULT": [0.0, 1.0, 1.5],
    "SL_ATR": [0.8, 1.2, 1.6],
    "RR": [1.0, 1.5, 2.0, 3.0],
    "TRAIL_ATR": [0.0, 1.0],
    "MAX_HOLD_BARS": [5, 10, 20],
}
MIN_ATR_BASE = 5.0  # index points on 1-minute bars


def apply_params(p):
    """Point the module-level settings at one grid combination."""
    global TIMEFRAME, EMA_FAST, EMA_SLOW, MIN_ATR, SL_ATR, RR, TRAIL_ATR, MAX_HOLD_BARS
    TIMEFRAME = p["TIMEFRAME"]
    EMA_FAST, EMA_SLOW = p["EMAS"]
    MIN_ATR = round(p["MIN_ATR_MULT"] * MIN_ATR_BASE * np.sqrt(TIMEFRAME), 1)
    SL_ATR = p["SL_ATR"]
    RR = p["RR"]
    TRAIL_ATR = p["TRAIL_ATR"]
    MAX_HOLD_BARS = p["MAX_HOLD_BARS"]


def live_settings():
    return {
        "TIMEFRAME": TIMEFRAME,
        "EMA_FAST": EMA_FAST,
        "EMA_SLOW": EMA_SLOW,
        "MIN_ATR": MIN_ATR,
        "SL_ATR": SL_ATR,
        "RR": RR,
        "TRAIL_ATR": TRAIL_ATR,
        "MAX_HOLD_BARS": MAX_HOLD_BARS,
    }


def fold_scores(trades, folds):
    """Expectancy after costs in each search fold, plus trade counts."""
    exps, counts = [], []
    for fold_days in folds:
        t = trades[trades["day"].isin(fold_days)] if not trades.empty else trades
        s = stats(t)
        exps.append(s["exp"])
        counts.append(s["trades"])
    return exps, counts


def optimize(client):
    """Search GRID for an edge that survives data the search never saw.

    1. The most recent OPT_HOLDOUT_FRACTION of days is the holdout. Nothing in
       the search looks at it.
    2. The remaining days are cut into OPT_FOLDS consecutive folds. A
       combination qualifies only if it is profitable after costs overall,
       in at least all-but-one fold, with a profit factor of 1.15 or better
       and enough trades in every fold to mean something.
    3. Qualifiers are ranked by their WORST fold, not their best, and the
       leader must sit on a plateau: its neighbours in the grid (one setting
       moved one step) must also be profitable on average. A lone spike
       among losing neighbours is luck.
    4. The single winner is run on the holdout once. It passes only if it is
       profitable there after costs with a profit factor of 1.1 or better.

    Every run is appended to optimizer_history.csv. Scheduled daily, the
    holdout rolls forward over new days; the same winner passing run after
    run on days it has never seen is the evidence worth trading on.
    """
    raw = fetch_history(client, OPT_DAYS)
    if raw is None:
        return
    days = sorted(pd.Series(raw.index.date).unique())
    log(
        f"Loaded {len(raw)} one-minute bars over {len(days)} trading days ({days[0]} to {days[-1]})"
    )
    if len(days) < OPT_MIN_DAYS:
        log(
            f"Only {len(days)} days of history, need at least {OPT_MIN_DAYS} to tell skill from luck. "
            "Download NIFTY 1-minute history into Historify and set HISTORY_SOURCE=db."
        )
        return

    n_hold = max(10, int(len(days) * OPT_HOLDOUT_FRACTION))
    search_days, holdout_days = days[:-n_hold], days[-n_hold:]
    folds = [list(f) for f in np.array_split(np.array(search_days, dtype=object), OPT_FOLDS)]
    min_fold_trades = 5
    log(
        f"Search on {len(search_days)} days in {OPT_FOLDS} folds, holdout {len(holdout_days)} days "
        f"({holdout_days[0]} to {holdout_days[-1]}), untouched until the end"
    )

    keys = list(GRID)
    combos = list(itertools.product(*(range(len(GRID[k])) for k in keys)))
    log(f"Testing {len(combos)} combinations, costs {COST_PREMIUM_POINTS} premium pts/round trip")

    frames = {}
    results = {}
    trades_by_combo = {}
    started = time.time()
    for n, idx in enumerate(combos, 1):
        p = {k: GRID[k][i] for k, i in zip(keys, idx, strict=True)}
        apply_params(p)
        frame_key = (TIMEFRAME, EMA_FAST, EMA_SLOW)
        if frame_key not in frames:
            frames[frame_key] = build_frame(raw)
        df = frames[frame_key]
        trades = simulate(df, compute_signals(df))
        search = trades[trades["day"].isin(search_days)] if not trades.empty else trades
        s = stats(search)
        exps, counts = fold_scores(search, folds)
        qualifies = (
            s["exp"] > 0
            and s["pf"] >= 1.15
            and min(counts) >= min_fold_trades
            and sum(e > 0 for e in exps) >= OPT_FOLDS - 1
        )
        results[idx] = {
            "params": p,
            "search": s,
            "fold_exps": exps,
            "fold_counts": counts,
            "worst_fold": min(exps),
            "qualifies": qualifies,
        }
        trades_by_combo[idx] = trades
        if n % 200 == 0:
            log(f"  {n}/{len(combos)} tested, {time.time() - started:.0f}s")

    qualified = sorted(
        (i for i, r in results.items() if r["qualifies"]),
        key=lambda i: results[i]["worst_fold"],
        reverse=True,
    )
    write_results(results, keys)
    log(f"{len(qualified)} of {len(combos)} combinations qualified on the search days")

    winner = None
    for idx in qualified:
        neighbours = []
        for pos in range(len(keys)):
            for step in (-1, 1):
                j = list(idx)
                j[pos] += step
                if 0 <= j[pos] < len(GRID[keys[pos]]):
                    neighbours.append(results[tuple(j)]["search"]["exp"])
        plateau = float(np.mean(neighbours)) if neighbours else float("-inf")
        if plateau > 0:
            winner = idx
            results[idx]["plateau"] = plateau
            break
        log(
            f"  rejected a spike: {describe(results[idx]['params'])}, neighbours average {plateau:.0f}/trade"
        )

    if winner is None:
        record_run(None, None, None)
        log("-" * 130)
        log("NO EDGE: nothing qualified on the search days AND sat on a profitable plateau.")
        log("This family of setups does not beat costs on this data. Do not trade it live.")
        return

    r = results[winner]
    apply_params(r["params"])
    trades = trades_by_combo[winner]
    hold = trades[trades["day"].isin(holdout_days)] if not trades.empty else trades
    h = stats(hold)
    passed = h["trades"] >= 15 and h["exp"] > 0 and h["pf"] >= 1.1

    log("-" * 130)
    log(f"Winner on search days: {describe(r['params'])}")
    log(
        f"  search   trades {r['search']['trades']}, after costs {r['search']['exp']:.0f}/trade, "
        f"PF {r['search']['pf']:.2f}, worst fold {r['worst_fold']:.0f}/trade, "
        f"neighbours {r['plateau']:.0f}/trade"
    )
    log(
        f"  holdout  trades {h['trades']}, before costs {h['gross_exp']:.0f}/trade, "
        f"after costs {h['exp']:.0f}/trade, PF {h['pf']:.2f}, win {h['win']:.1%}, max DD {h['dd']:.0f}"
    )
    streak = record_run(r["params"], h, passed)
    log("-" * 130)
    if not passed:
        log("NO EDGE: the best setting on the search days failed on the unseen holdout days.")
        log("It was luck on the search days. Do not trade it live.")
        return
    log(f"PASSED on unseen days. Same winner has passed {streak} run(s) in a row.")
    log("To paper trade it, set these parameters in /python with MODE=live and Analyze mode ON:")
    for k, v in live_settings().items():
        log(f"  {k}={v}")
    if streak < 5:
        log(
            "One pass is not proof. Keep this optimizer scheduled daily and wait for a streak of 5+."
        )


def describe(p):
    ema_fast, ema_slow = p["EMAS"]
    return (
        f"{p['TIMEFRAME']}m, EMA {ema_fast}/{ema_slow}, chop filter x{p['MIN_ATR_MULT']}, "
        f"SL {p['SL_ATR']} ATR, RR {p['RR']}, trail {p['TRAIL_ATR'] or 'off'}, hold {p['MAX_HOLD_BARS']} bars"
    )


def write_results(results, keys):
    rows = []
    for r in results.values():
        row = {k: str(r["params"][k]) for k in keys}
        row.update({f"search_{k}": v for k, v in r["search"].items()})
        row["worst_fold"] = r["worst_fold"]
        row["qualifies"] = r["qualifies"]
        rows.append(row)
    path = os.path.join(OPT_OUTPUT_DIR, "optimizer_results.csv")
    try:
        pd.DataFrame(rows).sort_values("worst_fold", ascending=False).to_csv(path, index=False)
        log(f"All combinations written to {path}")
    except OSError as e:
        log(f"Could not write {path}: {e}")


def record_run(params, holdout, passed):
    """Append this run to the history file; return the current pass streak."""
    path = os.path.join(OPT_OUTPUT_DIR, "optimizer_history.csv")
    row = {
        "run_at": datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
        "params": json.dumps(params, default=str) if params else "",
        "passed": bool(passed),
        "holdout_trades": holdout["trades"] if holdout else 0,
        "holdout_exp": round(holdout["exp"], 1) if holdout else 0.0,
        "holdout_pf": round(holdout["pf"], 2) if holdout else 0.0,
    }
    try:
        history = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        history.to_csv(path, index=False)
    except (OSError, pd.errors.ParserError) as e:
        log(f"Could not update {path}: {e}")
        return 1 if passed else 0
    streak = 0
    for _, h in history[::-1].iterrows():
        if bool(h["passed"]) and h["params"] == row["params"]:
            streak += 1
        else:
            break
    return streak


# ------------------------------------------------------------------- live --
class LiveTrader:
    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self.ltp = None
        self.trade = None  # Trade, while a position is open
        self.symbol = None
        self.qty = 0
        self.entry_premium = None
        self.exiting = False  # claim flag: one exit decision per position
        self.guard = DayGuard()
        self.expiry = None
        self.lot_size = None

    # --- setup
    def resolve_contract(self):
        res = self.client.expiry(
            symbol=UNDERLYING, exchange=OPTION_EXCHANGE, instrumenttype="options"
        )
        dates = res.get("data") or []
        if res.get("status") != "success" or not dates:
            raise RuntimeError(f"Could not load expiries: {res}")
        self.expiry = dates[0].replace("-", "").upper()  # 10-JUL-25 -> 10JUL25
        info = self.client.optionsymbol(
            underlying=UNDERLYING,
            exchange=INDEX_EXCHANGE,
            expiry_date=self.expiry,
            offset=OFFSET,
            option_type="CE",
        )
        if info.get("status") != "success":
            raise RuntimeError(f"Could not resolve option symbol: {info}")
        self.lot_size = int(info["lotsize"])
        log(f"Expiry {self.expiry}, lot size {self.lot_size}, trading {LOTS} lot(s)")

    def on_tick(self, data):
        try:
            ltp = float(data["data"]["ltp"])
        except (KeyError, TypeError, ValueError):
            return
        with self.lock:
            self.ltp = ltp
            trade = self.trade
            if trade is None or self.exiting:
                return
            reason = trade.update(ltp)
            if reason is None:
                return
            self.exiting = True
        # Order work happens outside the lock.
        self.exit(reason)

    # --- orders
    def fill_price(self, order_id):
        for _ in range(5):
            time.sleep(0.6)
            res = self.client.orderstatus(order_id=order_id, strategy=STRATEGY)
            data = res.get("data") or {}
            price = float(data.get("average_price") or 0)
            if price > 0:
                return price
        return None

    def enter(self, side, atr):
        option_type = "CE" if side == "LONG" else "PE"
        qty = LOTS * self.lot_size
        res = self.client.optionsorder(
            strategy=STRATEGY,
            underlying=UNDERLYING,
            exchange=INDEX_EXCHANGE,
            expiry_date=self.expiry,
            offset=OFFSET,
            option_type=option_type,
            action="BUY",
            quantity=qty,
            price_type="MARKET",
            product=PRODUCT,
        )
        if res.get("status") != "success":
            log(f"Entry refused, staying flat: {res.get('message', res)}")
            return
        with self.lock:
            ref = self.ltp or float(res.get("underlying_ltp") or 0)
            self.trade = Trade(side, ref, atr, datetime.now(IST))
            self.symbol = res["symbol"]
            self.qty = qty
            self.exiting = False
        self.entry_premium = self.fill_price(res["orderid"])
        t = self.trade
        log(
            f"ENTER {side} {self.symbol} x{qty} premium {self.entry_premium} | "
            f"index {t.entry:.2f} stop {t.stop:.2f} target {t.target:.2f}"
        )

    def exit(self, reason):
        with self.lock:
            trade, symbol, qty = self.trade, self.symbol, self.qty
        if trade is None:
            return
        res = self.client.placeorder(
            strategy=STRATEGY,
            symbol=symbol,
            action="SELL",
            exchange=OPTION_EXCHANGE,
            price_type="MARKET",
            product=PRODUCT,
            quantity=qty,
        )
        if res.get("status") != "success":
            # The position is still held: release the claim so the next tick
            # or the main loop retries, never forget the position.
            log(f"Exit order refused ({reason}), will retry: {res.get('message', res)}")
            with self.lock:
                self.exiting = False
            return
        exit_premium = self.fill_price(res["orderid"])
        with self.lock:
            index_now = self.ltp or trade.entry
            self.trade = None
            self.symbol = None
            self.exiting = False
        if self.entry_premium and exit_premium:
            pnl = (exit_premium - self.entry_premium) * qty
        else:
            pnl = trade.points(index_now) * DELTA * qty  # estimate when fills are unknown
        self.guard.record(pnl)
        log(
            f"EXIT {reason} {symbol} premium {self.entry_premium} -> {exit_premium} | "
            f"P&L {pnl:.0f} | today {self.guard.pnl:.0f} over {self.guard.trades} trade(s)"
        )

    def claim_exit(self, reason):
        with self.lock:
            if self.trade is None or self.exiting:
                return
            self.exiting = True
        self.exit(reason)

    # --- loop
    def closed_bars(self):
        raw = fetch_history(self.client, 5, source="api")
        if raw is None:
            return None
        now_minute = pd.Timestamp(datetime.now(IST)).floor("min")
        df = resample(raw, TIMEFRAME)
        # Keep only bars that have fully closed.
        df = df[df.index + pd.Timedelta(minutes=TIMEFRAME) <= now_minute]
        return add_indicators(df)

    def run(self):
        self.resolve_contract()
        self.client.connect()
        self.client.subscribe_ltp(
            [{"exchange": INDEX_EXCHANGE, "symbol": UNDERLYING}], on_data_received=self.on_tick
        )
        log(f"Live on {TIMEFRAME}m bars. Windows {WINDOWS}, square-off {SQUARE_OFF}")
        log(f"Settings: {live_settings()}")
        last_bar = None
        try:
            while True:
                now = datetime.now(IST)
                self.guard.roll(now.date())

                with self.lock:
                    trade = self.trade
                if trade is not None:
                    held = (now - trade.entry_time).total_seconds() / 60
                    if now.time() >= SQUARE_OFF:
                        self.claim_exit("SQUARE_OFF")
                    elif held >= MAX_HOLD_BARS * TIMEFRAME:
                        self.claim_exit("TIME")

                if now.time() >= SQUARE_OFF and self.trade is None:
                    log("Square-off time reached and flat. Done for the day.")
                    break

                # Evaluate once per closed bar, a couple of seconds after it
                # closes. The bar that just closed started TIMEFRAME minutes ago.
                bar_close = now.replace(second=0, microsecond=0)
                bar_start = bar_close - timedelta(minutes=TIMEFRAME)
                on_boundary = (bar_close.hour * 60 + bar_close.minute) % TIMEFRAME == 0
                if (
                    self.trade is None
                    and on_boundary
                    and now.second >= 2
                    and bar_close != last_bar
                    and in_window(bar_start.time())
                ):
                    last_bar = bar_close
                    self.evaluate()
                time.sleep(1)
        finally:
            if self.trade is not None:
                self.claim_exit("SHUTDOWN")
            self.client.disconnect()

    def evaluate(self):
        if not self.guard.can_trade():
            return
        df = self.closed_bars()
        if df is None or len(df) < EMA_SLOW + 5:
            return
        if df.index[-1].date() != datetime.now(IST).date():
            return
        sig = compute_signals(df)[-1]
        if sig:
            self.enter("LONG" if sig > 0 else "SHORT", float(df["atr"].iloc[-1]))


def main():
    if not API_KEY:
        log("OPENALGO_API_KEY is not set. Generate one at /apikey.")
        return
    client = api(api_key=API_KEY, host=HOST, ws_url=WS_URL)
    if MODE == "live":
        LiveTrader(client).run()
    elif MODE == "optimize":
        optimize(client)
    else:
        backtest(client)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
