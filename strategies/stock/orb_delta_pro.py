#!/usr/bin/env python
"""
ORB Delta Pro v2 - rolling-channel breakout for one NSE stock, intraday (MIS), with real risk control.

What was driving the drawdown in v1, and what replaced it
---------------------------------------------------------
1. The stop lived only in this script and was checked every 30 seconds. A fast move went
   straight through it and the market exit filled far beyond the stop.
   -> A real SL-M stop order sits at the broker from the moment of the fill and is MODIFIED as
      the trail tightens. The in-script stop is only a backup, and the loop polls every 5s.
2. Every 30-minute breakout was taken, all day, in both directions, with no trend filter. On a
   single stock most of those are chop, and each one cost a full 1% of equity.
   -> A breakout is taken only WITH the trend (EMA 20/50 and the session VWAP agree), only when
      the tape is actually trending (Kaufman efficiency ratio), only when the channel is neither
      noise-thin nor exhausted-wide, only when the breakout bar has not already run away, and
      only in the liquid windows (default 09:30-11:30 and 13:15-14:45).
3. Risk per trade was 1% with up to 5 losses in a row allowed, and nothing watched the
   strategy's own equity curve, so bad days stacked into a deep drawdown.
   -> 0.5% per trade, at most 3 trades and 2 losses in a row per day, a daily loss limit of 2R
      (and 1.5% of equity), a weekly loss limit of 4%, and a peak-to-trough brake: size halves
      when the strategy is 5% below its equity peak and trading halts at 10% until you reset it.
4. The trail sat 3 ATR behind the best CLOSE with no breakeven step, so winners were handed back.
   -> Staged management: breakeven (plus costs) at +1R, half the position booked at +1.5R, a
      2.5 ATR chandelier from the best HIGH/LOW that tightens to 1.5 ATR after +2.5R, and a time
      stop that closes a trade that has not reached +0.5R within 6 bars (30 minutes).
5. Losers were never cut early and the bot re-entered immediately after a stop-out.
   -> The time stop above, plus a cooldown of 3 bars after every exit.

The backtest replays the SAME entry filter, trade manager and risk governor bar by bar, so its
drawdown number describes what the live loop would have done. It still cannot promise profit.

Trading mode
------------
  TRADING_MODE=sandbox (default)  orders go to OpenAlgo's Analyze mode. Refuses to start if
                                  Analyze mode is off, so it can never send a real order.
  TRADING_MODE=live               real orders. Refuses unless Analyze mode is OFF and
                                  CONFIRM_LIVE=YES. The first RAMP_IN_TRADES live trades use
                                  RAMP_IN_SIZE_PCT of normal size.

Commands (MODE=<cmd> for the /python host, or the first command-line argument)
------------------------------------------------------------------------------
  monitor (default)  live loop          backtest   BACKTEST_DAYS of history (HISTORY_SOURCE=db for Historify)
  scan               one signal check   report     P&L from the trade log
  status             state and brakes   reset-breaker   clear the drawdown halt after you have reviewed it
  selftest           offline tests of the mechanics

Every setting is an environment variable of the same name.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def env(name, default, cast=str):
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    if cast is bool:
        return v.strip().lower() in ("1", "true", "yes", "y")
    return cast(v)


def clock(hhmm: str) -> dtime:
    h, m = (int(x) for x in hhmm.strip().split(":")[:2])
    return dtime(h, m)


def parse_windows(spec: str) -> list[tuple[dtime, dtime]]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.append((clock(a), clock(b)))
    return out


IST = ZoneInfo("Asia/Kolkata")

# ---------------------------------------------------------------- connection / mode --
SYMBOL = env("SYMBOL", "PINELABS").upper()
EXCHANGE = env("OPENALGO_STRATEGY_EXCHANGE", env("EXCHANGE", "NSE"))
API_KEY = os.getenv("OPENALGO_API_KEY", "")
HOST = env("HOST_SERVER", "") or env("OPENALGO_HOST", "http://127.0.0.1:5000")
PRODUCT = env("PRODUCT", "MIS")
STRATEGY_TAG = env("STRATEGY_NAME", "Orb Delta Pro")
TRADING_MODE = env("TRADING_MODE", "sandbox").lower()  # sandbox | live
CONFIRM_LIVE = env("CONFIRM_LIVE", False, bool)
TICK_SIZE = env("TICK_SIZE", 0.05, float)

# ---------------------------------------------------------------- money / risk --
CAPITAL_RUPEES = env("CAPITAL_RUPEES", 20000.0, float)  # capital allocated to THIS strategy
RISK_PCT = env("RISK_PER_TRADE_PCT", 0.005, float)  # 0.5% of strategy equity per trade
MAX_LEVERAGE = env("MAX_LEVERAGE", 3.0, float)
MAX_NOTIONAL_RUPEES = env("MAX_NOTIONAL_RUPEES", CAPITAL_RUPEES * MAX_LEVERAGE, float)
MIN_STOP_DISTANCE_PCT = env(
    "MIN_STOP_DISTANCE_PCT", 0.15, float
)  # stop never closer than this % of price
MAX_TRADES_PER_DAY = env("MAX_TRADES_PER_DAY", 3, int)
CONSECUTIVE_LOSS_LIMIT = env("CONSECUTIVE_LOSS_LIMIT", 2, int)  # per day
DAILY_LOSS_R = env("DAILY_LOSS_R", 2.0, float)
DAILY_LOSS_PCT = env("DAILY_LOSS_PCT", 1.5, float)
WEEKLY_LOSS_PCT = env("WEEKLY_LOSS_PCT", 4.0, float)
DD_THROTTLE_PCT = env("DD_THROTTLE_PCT", 5.0, float)  # below peak by this much -> half size
DD_HALT_PCT = env("DD_HALT_PCT", 10.0, float)  # below peak by this much -> halt until reset
RAMP_IN_TRADES = env("RAMP_IN_TRADES", 5, int)
RAMP_IN_SIZE_PCT = env("RAMP_IN_SIZE_PCT", 0.5, float)
PRICE_SANITY_MAX_PCT = env("PRICE_SANITY_MAX_PCT", 1.0, float)

# ---------------------------------------------------------------- signal --
BAR_MINUTES = 5
CHANNEL_BARS = env("CHANNEL_BARS", 6, int)  # 6 x 5m = 30-minute channel
ATR_PERIOD = env("ATR_PERIOD", 14, int)
EMA_FAST = env("EMA_FAST", 20, int)
EMA_SLOW = env("EMA_SLOW", 50, int)
ER_BARS = env("ER_BARS", 10, int)
ER_MIN = env("ER_MIN", 0.30, float)  # 0 disables the trend-quality filter
MIN_CHANNEL_ATR = env("MIN_CHANNEL_ATR", 0.8, float)
MAX_CHANNEL_ATR = env("MAX_CHANNEL_ATR", 4.0, float)
MAX_EXTENSION_ATR = env("MAX_EXTENSION_ATR", 0.8, float)
USE_TREND_FILTER = env("USE_TREND_FILTER", True, bool)
ENTRY_WINDOWS = parse_windows(env("ENTRY_WINDOWS", "09:30-11:30,13:15-14:45"))
EOD_EXIT = clock(env("EOD_EXIT_CLOCK", "15:15"))
MARKET_OPEN = clock(env("MARKET_OPEN_CLOCK", "09:15"))
MARKET_CLOSE = clock(env("MARKET_CLOSE_CLOCK", "15:30"))

# ---------------------------------------------------------------- trade management --
STOP_MIN_ATR = env("STOP_MIN_ATR", 0.5, float)
STOP_MAX_ATR = env("STOP_MAX_ATR", 1.5, float)
BE_AT_R = env("BE_AT_R", 1.0, float)
BE_BUFFER_PCT = env(
    "BE_BUFFER_PCT", 0.08, float
)  # breakeven stop sits this % past entry to cover costs
PARTIAL_AT_R = env("PARTIAL_AT_R", 1.5, float)
PARTIAL_PCT = env("PARTIAL_PCT", 0.5, float)  # 0 disables the partial
TRAIL_ATR = env("TRAIL_ATR", 2.5, float)
TIGHTEN_AT_R = env("TIGHTEN_AT_R", 2.5, float)
TRAIL_TIGHT_ATR = env("TRAIL_TIGHT_ATR", 1.5, float)
TIME_STOP_BARS = env("TIME_STOP_BARS", 6, int)
TIME_STOP_MIN_R = env("TIME_STOP_MIN_R", 0.5, float)
COOLDOWN_BARS = env("COOLDOWN_BARS", 3, int)

# ---------------------------------------------------------------- runtime --
POLL_SECONDS = env("POLL_SECONDS", 5.0, float)
STATUS_EVERY_SEC = env("STATUS_EVERY_SEC", 5.0, float)
SOFT_STOP_SLACK_PCT = env(
    "SOFT_STOP_SLACK_PCT", 0.2, float
)  # backup exit if the broker stop lags this far
DATA_SOURCE = env("ORB_DATA_SOURCE", "auto").lower()  # auto | history | quotes
HISTORY_SOURCE = env("HISTORY_SOURCE", "api")  # backtest: api | db
BACKTEST_DAYS = env("BACKTEST_DAYS", 180, int)
SLIPPAGE_PCT = env("SLIPPAGE_PCT", 0.03, float)  # backtest, per fill
LOG_DIR = Path(env("ORB_LOG_DIR", str(Path(__file__).resolve().parent / "log")))
STATE_PATH = LOG_DIR / f"orb_v2_state_{SYMBOL}.json"
TRADES_PATH = LOG_DIR / f"orb_v2_trades_{SYMBOL}.csv"
QUOTES_PATH = LOG_DIR / f"orb_v2_quotes_{SYMBOL}.json"


def now_ist() -> datetime:
    """Single clock for the bot (the self-test patches it)."""
    return datetime.now(IST)


def log(msg: str) -> None:
    print(f"[{now_ist():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def round_tick(px: float, mode: str = "nearest") -> float:
    q = px / TICK_SIZE
    q = (
        math.floor(q + 1e-9)
        if mode == "down"
        else math.ceil(q - 1e-9)
        if mode == "up"
        else round(q)
    )
    return round(q * TICK_SIZE, 2)


# ---------------------------------------------------------------- costs (Groww MIS schedule; verify yours) --
_STT_SELL = env("STT_SELL_PCT", 0.025, float) / 100
_EXCH_TXN = env("EXCHANGE_TXN_PCT", 0.00297, float) / 100
_SEBI = env("SEBI_FEE_PCT", 0.0001, float) / 100
_IPFT = env("IPFT_PCT", 0.0001, float) / 100
_STAMP_BUY = env("STAMP_DUTY_BUY_PCT", 0.003, float) / 100
_BROK_FLAT = env("BROKERAGE_FLAT_RUPEES", 20.0, float)
_BROK_PCT = env("BROKERAGE_PCT", 0.1, float) / 100


def _leg_cost(value: float, is_buy: bool) -> float:
    brok = min(_BROK_FLAT, value * _BROK_PCT)
    fees = value * (_EXCH_TXN + _SEBI + _IPFT)
    gst = 0.18 * (brok + fees)
    return brok + fees + gst + value * (_STAMP_BUY if is_buy else _STT_SELL)


def round_trip_cost(entry: float, exit_: float, qty: int, side: int) -> float:
    ev, xv = entry * qty, exit_ * qty
    return _leg_cost(ev, side > 0) + _leg_cost(xv, side < 0)


def net_pnl(entry: float, exit_: float, qty: int, side: int) -> float:
    return side * (exit_ - entry) * qty - round_trip_cost(entry, exit_, qty, side)


cc = SimpleNamespace(round_trip_cost=round_trip_cost, net_pnl=net_pnl)


# ---------------------------------------------------------------- indicators and entry filter --
def wilder_atr(h: pd.Series, lo: pd.Series, c: pd.Series, n: int) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """5-minute OHLCV (IST index) -> frame with every column the entry filter and manager need."""
    d = df.copy()
    if "volume" not in d.columns:
        d["volume"] = 0.0
    d = d[["open", "high", "low", "close", "volume"]].astype(float)
    h, lo, c = d["high"], d["low"], d["close"]
    day = d.index.normalize()
    d["day"] = day
    d["mins"] = d.index.hour * 60 + d.index.minute
    d["bar_of_day"] = d.groupby(day).cumcount()
    d["atr"] = wilder_atr(h, lo, c, ATR_PERIOD)
    d["ema_f"] = c.ewm(span=EMA_FAST, adjust=False).mean()
    d["ema_s"] = c.ewm(span=EMA_SLOW, adjust=False).mean()
    tp = (h + lo + c) / 3
    vol = d["volume"].fillna(0)
    cum_v = vol.groupby(day).cumsum()
    cum_pv = (tp * vol).groupby(day).cumsum()
    twap = tp.groupby(day).cumsum() / tp.groupby(day).cumcount().add(1)
    d["vwap"] = np.where(cum_v > 0, cum_pv / cum_v.replace(0, np.nan), twap)
    change = (c - c.shift(ER_BARS)).abs()
    noise = c.diff().abs().rolling(ER_BARS).sum()
    d["er"] = (change / noise.replace(0, np.nan)).fillna(0)
    d["ch_hi"] = h.shift(1).rolling(CHANNEL_BARS).max()
    d["ch_lo"] = lo.shift(1).rolling(CHANNEL_BARS).min()
    return d


def candidates(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per CLOSED bar: side (+1/-1/0) and the initial stop the entry filter wants."""
    c, h, lo = d["close"], d["high"], d["low"]
    atr, ch_hi, ch_lo = d["atr"], d["ch_hi"], d["ch_lo"]
    close_mins = d["mins"] + BAR_MINUTES
    in_win = np.zeros(len(d), dtype=bool)
    for a, b in ENTRY_WINDOWS:
        in_win |= (
            (close_mins >= a.hour * 60 + a.minute) & (close_mins <= b.hour * 60 + b.minute)
        ).to_numpy()
    width = ch_hi - ch_lo
    base = (
        in_win
        & (d["bar_of_day"] >= CHANNEL_BARS).to_numpy()  # the channel is built from today only
        & (atr > 0).to_numpy()
        & (width >= MIN_CHANNEL_ATR * atr).to_numpy()
        & (width <= MAX_CHANNEL_ATR * atr).to_numpy()
        & (d["er"] >= ER_MIN).to_numpy()
    )
    long_ok = (h > ch_hi) & (c > ch_hi) & (c - ch_hi <= MAX_EXTENSION_ATR * atr)
    short_ok = (lo < ch_lo) & (c < ch_lo) & (ch_lo - c <= MAX_EXTENSION_ATR * atr)
    if USE_TREND_FILTER:
        long_ok &= (d["ema_f"] > d["ema_s"]) & (c > d["vwap"])
        short_ok &= (d["ema_f"] < d["ema_s"]) & (c < d["vwap"])
    side = np.where(base & long_ok.to_numpy(), 1, np.where(base & short_ok.to_numpy(), -1, 0))
    raw_stop = np.where(side > 0, ch_lo.to_numpy(), ch_hi.to_numpy())
    cv, av = c.to_numpy(), atr.to_numpy()
    dist = np.abs(cv - raw_stop)
    dist = np.clip(dist, STOP_MIN_ATR * av, STOP_MAX_ATR * av)
    dist = np.maximum(dist, cv * MIN_STOP_DISTANCE_PCT / 100)
    stop = cv - side * dist
    side[: max(EMA_SLOW, ATR_PERIOD, CHANNEL_BARS) + 1] = 0
    return side, stop


# ---------------------------------------------------------------- trade manager --
class Trade:
    """Stop, breakeven, partial, trail and time stop for one position. Pure: no I/O.

    check_price() is fed every price the loop sees (backtest: bar extremes, adverse first).
    on_bar_close() is fed each closed bar and moves the stop.
    """

    def __init__(self, side: int, entry: float, qty: int, stop: float, opened: str):
        self.side, self.entry, self.qty, self.open_qty = side, entry, qty, qty
        self.stop = stop
        self.risk = abs(entry - stop)
        self.best = entry
        self.bars = 0
        self.max_r = 0.0
        self.partial_done = False
        self.opened = opened
        self.fills: list[list] = []  # [qty, price, reason]

    def r_at(self, price: float) -> float:
        return self.side * (price - self.entry) / self.risk if self.risk > 0 else 0.0

    @property
    def partial_level(self) -> float:
        return self.entry + self.side * PARTIAL_AT_R * self.risk

    @property
    def partial_qty(self) -> int:
        return int(self.qty * PARTIAL_PCT) if PARTIAL_PCT > 0 and not self.partial_done else 0

    def stop_reason(self) -> str:
        gap = self.side * (self.stop - self.entry)
        if gap < 0:
            return "stop"
        return "breakeven" if gap <= self.entry * BE_BUFFER_PCT / 100 + 1e-9 else "trail"

    def check_price(self, price: float):
        if self.side * (price - self.stop) <= 0:
            return "EXIT", self.stop_reason()
        if (
            self.partial_qty >= 1
            and self.open_qty > self.partial_qty
            and self.side * (price - self.partial_level) >= 0
        ):
            return "PARTIAL", "partial"
        return None

    def on_bar_close(self, high: float, low: float, close: float, atr: float):
        self.bars += 1
        self.best = max(self.best, high) if self.side > 0 else min(self.best, low)
        r = self.r_at(self.best)
        self.max_r = max(self.max_r, r)
        new = self.stop
        if r >= BE_AT_R:
            be = self.entry * (1 + self.side * BE_BUFFER_PCT / 100)
            new = max(new, be) if self.side > 0 else min(new, be)
        if atr > 0:
            mult = TRAIL_TIGHT_ATR if r >= TIGHTEN_AT_R else TRAIL_ATR
            trail = self.best - self.side * mult * atr
            new = max(new, trail) if self.side > 0 else min(new, trail)
        self.stop = new
        if self.bars >= TIME_STOP_BARS and self.max_r < TIME_STOP_MIN_R:
            return "EXIT", "time"
        return None

    def fill(self, qty: int, price: float, reason: str) -> None:
        self.fills.append([qty, price, reason])
        self.open_qty -= qty
        if reason == "partial":
            self.partial_done = True

    def pnl(self) -> float:
        return sum(net_pnl(self.entry, px, q, self.side) for q, px, _ in self.fills)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, s: dict) -> Trade:
        t = cls(s["side"], s["entry"], s["qty"], s["stop"], s["opened"])
        t.__dict__.update(s)
        return t


# ---------------------------------------------------------------- risk governor --
class Governor:
    """Daily, weekly and peak-to-trough brakes on THIS strategy's own realized P&L. Pure: no I/O."""

    def __init__(self, capital: float):
        self.capital = capital
        self.realized = 0.0
        self.peak = capital
        self.halted = False  # drawdown halt, cleared only by reset-breaker
        self.day = ""
        self.week = ""
        self.day_start_equity = capital
        self.week_start_equity = capital
        self.day_pnl = self.week_pnl = 0.0
        self.day_trades = self.losses_in_row = 0
        self.live_fills = 0

    @property
    def equity(self) -> float:
        return self.capital + self.realized

    def dd_pct(self) -> float:
        return max(0.0, (self.peak - self.equity) / self.peak * 100) if self.peak > 0 else 0.0

    def roll(self, d) -> None:
        day, week = d.isoformat(), "{}-W{:02d}".format(*d.isocalendar()[:2])
        if day != self.day:
            self.day, self.day_pnl, self.day_trades, self.losses_in_row = day, 0.0, 0, 0
            self.day_start_equity = self.equity
        if week != self.week:
            self.week, self.week_pnl, self.week_start_equity = week, 0.0, self.equity

    def size_mult(self) -> float:
        dd = self.dd_pct()
        if self.halted or dd >= DD_HALT_PCT:
            self.halted = True
            return 0.0
        return 0.5 if dd >= DD_THROTTLE_PCT else 1.0

    def r_unit(self) -> float:
        return self.equity * RISK_PCT

    def can_enter(self) -> tuple[bool, str]:
        if self.size_mult() == 0:
            return (
                False,
                f"drawdown halt: {self.dd_pct():.1f}% below peak (limit {DD_HALT_PCT}%), run reset-breaker",
            )
        if self.day_trades >= MAX_TRADES_PER_DAY:
            return False, f"{self.day_trades} trades today (limit {MAX_TRADES_PER_DAY})"
        if self.losses_in_row >= CONSECUTIVE_LOSS_LIMIT:
            return (
                False,
                f"{self.losses_in_row} losses in a row today (limit {CONSECUTIVE_LOSS_LIMIT})",
            )
        if (
            self.day_pnl <= -DAILY_LOSS_R * self.r_unit()
            or self.day_pnl <= -self.day_start_equity * DAILY_LOSS_PCT / 100
        ):
            return False, f"daily loss limit hit (Rs {self.day_pnl:.0f})"
        if self.week_pnl <= -self.week_start_equity * WEEKLY_LOSS_PCT / 100:
            return False, f"weekly loss limit hit (Rs {self.week_pnl:.0f})"
        return True, ""

    def record_entry(self) -> None:
        self.day_trades += 1

    def record_close(self, pnl: float, live: bool) -> None:
        self.realized += pnl
        self.peak = max(self.peak, self.equity)
        self.day_pnl += pnl
        self.week_pnl += pnl
        self.losses_in_row = self.losses_in_row + 1 if pnl <= 0 else 0
        if live:
            self.live_fills += 1
        self.size_mult()  # latch the halt the moment the limit is crossed

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, s: dict) -> Governor:
        g = cls(s.get("capital", CAPITAL_RUPEES))
        g.__dict__.update(s)
        return g


def size_qty(equity: float, risk_pct: float, entry: float, stop: float) -> int:
    dist = abs(entry - stop)
    if dist <= 0 or equity <= 0 or entry <= 0:
        return 0
    per_share_cost = round_trip_cost(entry, entry, 1000, 1) / 1000 + entry * 2 * SLIPPAGE_PCT / 100
    qty = int(equity * risk_pct // (dist + per_share_cost))
    cap = int(min(MAX_NOTIONAL_RUPEES, equity * MAX_LEVERAGE) // entry)
    return max(0, min(qty, cap))


# ---------------------------------------------------------------- backtest --
def backtest_frame(
    df: pd.DataFrame, capital: float = CAPITAL_RUPEES
) -> tuple[pd.DataFrame, Governor]:
    d = prepare(df)
    side_a, stop_a = candidates(d)
    o, h, lo, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = d["atr"].to_numpy()
    day = d["day"].to_numpy()
    close_mins = (d["mins"] + BAR_MINUTES).to_numpy()
    eod = EOD_EXIT.hour * 60 + EOD_EXIT.minute
    slip = SLIPPAGE_PCT / 100
    gov = Governor(capital)
    rows = []
    n = len(d)
    last_exit = -(10**9)
    i = 0
    while i < n - 1:
        gov.roll(pd.Timestamp(day[i]).date())
        s = int(side_a[i])
        j = i + 1
        if (
            s == 0
            or day[j] != day[i]
            or (day[last_exit] == day[i] if last_exit >= 0 else False)
            and i - last_exit < COOLDOWN_BARS
        ):
            i += 1
            continue
        ok, _ = gov.can_enter()
        mult = gov.size_mult()
        if not ok or mult == 0:
            i += 1
            continue
        entry = o[j] * (1 + s * slip)
        stop = stop_a[i]
        if s * (entry - stop) <= 0:  # gapped through the stop before the fill
            i += 1
            continue
        qty = size_qty(gov.equity, RISK_PCT * mult, entry, stop)
        if qty < 1:
            i += 1
            continue
        t = Trade(s, entry, qty, stop, str(d.index[j]))
        gov.record_entry()
        k = j
        while k < n and day[k] == day[i]:
            if k > j and s * (o[k] - t.stop) <= 0:  # opened through the stop: fill at the open
                t.fill(t.open_qty, o[k] * (1 - s * slip), t.stop_reason())
                break
            adverse, favourable = (lo[k], h[k]) if s > 0 else (h[k], lo[k])
            hit = t.check_price(adverse)
            if hit and hit[0] == "EXIT":
                t.fill(t.open_qty, t.stop * (1 - s * slip), hit[1])
                break
            hit = t.check_price(favourable)
            if hit and hit[0] == "PARTIAL":
                t.fill(t.partial_qty, t.partial_level, "partial")
            if close_mins[k] >= eod:
                t.fill(t.open_qty, c[k] * (1 - s * slip), "eod")
                break
            hit = t.on_bar_close(h[k], lo[k], c[k], atr[k])
            if hit:
                t.fill(t.open_qty, c[k] * (1 - s * slip), hit[1])
                break
            k += 1
        k = min(k, n - 1)
        if t.open_qty > 0:
            t.fill(t.open_qty, c[k] * (1 - s * slip), "eod")
        pnl = t.pnl()
        gov.record_close(pnl, live=False)
        rows.append(
            {
                "day": pd.Timestamp(day[i]).date(),
                "time": str(d.index[j].time()),
                "side": "LONG" if s > 0 else "SHORT",
                "qty": qty,
                "entry": round(entry, 2),
                "stop0": round(stop, 2),
                "exits": ";".join(f"{q}@{px:.2f}:{r}" for q, px, r in t.fills),
                "reason": t.fills[-1][2],
                "pnl": round(pnl, 2),
                "r": round(pnl / (t.risk * qty), 3),
                "equity": round(gov.equity, 2),
            }
        )
        last_exit = k
        i = k + 1
    return pd.DataFrame(rows), gov


def summarize(t: pd.DataFrame, capital: float, label: str) -> None:
    if t.empty:
        print(f"[{label}] no trades")
        return
    eq = capital + t["pnl"].cumsum()
    peak = eq.cummax().clip(lower=capital)
    dd = (eq - peak).min()
    dd_pct = ((eq - peak) / peak).min() * 100
    wins, losses = t[t.pnl > 0], t[t.pnl <= 0]
    pf = wins.pnl.sum() / -losses.pnl.sum() if losses.pnl.sum() < 0 else float("inf")
    streak = run = 0
    for p in t.pnl:
        run = run + 1 if p <= 0 else 0
        streak = max(streak, run)
    print(
        f"[{label}] trades {len(t)} | win {len(wins) / len(t):.1%} | avg R {t.r.mean():+.2f} | PF {pf:.2f} | "
        f"net Rs {t.pnl.sum():+,.0f} ({t.pnl.sum() / capital:+.1%}) | max DD Rs {dd:,.0f} ({dd_pct:.1f}%) | "
        f"worst losing streak {streak}"
    )
    print(f"        exits: {t.reason.value_counts().to_dict()}")


def run_backtest(client) -> None:
    end = now_ist().date()
    start = end - timedelta(days=BACKTEST_DAYS)
    df = client.history(
        symbol=SYMBOL,
        exchange=EXCHANGE,
        interval="5m",
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        source=HISTORY_SOURCE,
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        log(f"No history for {SYMBOL}: {df}. For Historify data set HISTORY_SOURCE=db.")
        return
    df = to_ist(df)
    t, gov = backtest_frame(df)
    print(
        f"\nORB Delta Pro v2 backtest {SYMBOL} {df.index[0]:%Y-%m-%d}..{df.index[-1]:%Y-%m-%d}, capital Rs {CAPITAL_RUPEES:,.0f}"
    )
    summarize(t, CAPITAL_RUPEES, "ALL")
    if not t.empty:
        days = sorted(t.day.unique())
        cut = days[int(len(days) * 0.6)] if len(days) > 4 else None
        if cut:
            summarize(t[t.day < cut], CAPITAL_RUPEES, "FIRST 60%")
            summarize(t[t.day >= cut], CAPITAL_RUPEES, "LAST 40%")
        out = LOG_DIR / f"orb_v2_backtest_{SYMBOL}.csv"
        t.to_csv(out, index=False)
        print(f"Trades written to {out}")
        if gov.halted:
            print(
                "NOTE: the drawdown halt tripped during the test; trading stopped there, as it would live."
            )
    print(
        "Only trade it if BOTH halves are profitable after costs and the max drawdown is one you can sit through."
    )


def to_ist(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    df = df.copy()
    df.index = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    return df.sort_index()


# ---------------------------------------------------------------- live broker helpers --
_STATUS = {
    "complete": "complete",
    "completed": "complete",
    "filled": "complete",
    "traded": "complete",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "rejected": "rejected",
}


class Broker:
    """Thin OpenAlgo wrapper. Every call returns a plain value and never raises."""

    def __init__(self, client):
        self.c = client

    def ltp(self) -> float | None:
        try:
            q = self.c.quotes(symbol=SYMBOL, exchange=EXCHANGE)
            data = q.get("data", {}) if isinstance(q, dict) else {}
            v = data.get("ltp") or data.get("last_price")
            return float(v) if v else None
        except Exception as e:
            log(f"quote failed: {e}")
            return None

    def order(
        self, action: str, qty: int, price_type: str = "MARKET", trigger: float = 0.0
    ) -> str | None:
        try:
            r = self.c.placeorder(
                strategy=STRATEGY_TAG,
                symbol=SYMBOL,
                action=action,
                exchange=EXCHANGE,
                price_type=price_type,
                product=PRODUCT,
                quantity=int(qty),
                price=0,
                trigger_price=trigger,
            )
        except Exception as e:
            log(f"{action} {price_type} order failed: {e}")
            return None
        if not isinstance(r, dict) or r.get("status") != "success":
            log(
                f"{action} {price_type} order refused: {r.get('message', r) if isinstance(r, dict) else r}"
            )
            return None
        return str(r.get("orderid"))

    def modify_stop(self, oid: str, action: str, qty: int, trigger: float) -> bool:
        try:
            r = self.c.modifyorder(
                order_id=oid,
                strategy=STRATEGY_TAG,
                symbol=SYMBOL,
                action=action,
                exchange=EXCHANGE,
                price_type="SL-M",
                product=PRODUCT,
                quantity=int(qty),
                price=0,
                trigger_price=trigger,
            )
            return isinstance(r, dict) and r.get("status") == "success"
        except Exception as e:
            log(f"stop modify failed: {e}")
            return False

    def cancel(self, oid: str) -> None:
        try:
            self.c.cancelorder(order_id=oid, strategy=STRATEGY_TAG)
        except Exception as e:
            log(f"cancel {oid} failed: {e}")

    def status(self, oid: str) -> tuple[str, float]:
        try:
            r = self.c.orderstatus(order_id=oid, strategy=STRATEGY_TAG)
            data = r.get("data", {}) if isinstance(r, dict) else {}
            st = _STATUS.get(str(data.get("order_status", "")).strip().lower(), "open")
            return st, float(data.get("average_price") or 0)
        except Exception as e:
            log(f"orderstatus {oid} failed: {e}")
            return "unknown", 0.0

    def wait(self, oid: str, seconds: float = 5.0) -> tuple[str, float]:
        end = time.time() + seconds
        st, px = self.status(oid)
        while st not in ("complete", "cancelled", "rejected") and time.time() < end:
            time.sleep(0.5)
            st, px = self.status(oid)
        return st, px

    def position(self) -> int | None:
        """Signed shares held for SYMBOL, or None when unknown (None is never 'flat')."""
        try:
            r = self.c.openposition(
                strategy=STRATEGY_TAG, symbol=SYMBOL, exchange=EXCHANGE, product=PRODUCT
            )
        except Exception as e:
            log(f"position read failed: {e}")
            return None
        if not isinstance(r, dict) or r.get("status") != "success":
            return None
        try:
            return int(float(r.get("quantity", 0) or 0))
        except (TypeError, ValueError):
            return None

    def analyze_mode(self) -> bool | None:
        try:
            r = self.c.analyzerstatus()
            if isinstance(r, dict) and r.get("status") == "success":
                return bool((r.get("data") or {}).get("analyze_mode"))
        except Exception as e:
            log(f"analyzer status failed: {e}")
        return None


# ---------------------------------------------------------------- live engine --
class Live:
    def __init__(self, client):
        self.b = Broker(client)
        self.client = client
        s = load_json(STATE_PATH)
        self.gov = Governor.from_dict(s["gov"]) if s.get("gov") else Governor(CAPITAL_RUPEES)
        self.trade = Trade.from_dict(s["trade"]) if s.get("trade") else None
        self.stop_oid = s.get("stop_oid")
        self.stop_sent = s.get("stop_sent")  # trigger currently at the broker
        self.last_bar = s.get("last_bar")
        self.last_exit_bar = s.get("last_exit_bar")
        self.last_status = 0.0
        self.use_quotes = DATA_SOURCE == "quotes"

    def save(self) -> None:
        save_json(
            STATE_PATH,
            {
                "gov": self.gov.to_dict(),
                "trade": self.trade.to_dict() if self.trade else None,
                "stop_oid": self.stop_oid,
                "stop_sent": self.stop_sent,
                "last_bar": self.last_bar,
                "last_exit_bar": self.last_exit_bar,
            },
        )

    # --- safety gates
    def preflight(self) -> bool:
        mode = self.b.analyze_mode()
        if mode is None:
            log("Could not read OpenAlgo's Analyze mode. Not starting.")
            return False
        if TRADING_MODE == "sandbox" and not mode:
            log(
                "TRADING_MODE=sandbox but OpenAlgo is LIVE. Turn Analyze mode on first. Not starting."
            )
            return False
        if TRADING_MODE == "live":
            if mode:
                log(
                    "TRADING_MODE=live but OpenAlgo is in Analyze mode, orders would be simulated. Not starting."
                )
                return False
            if not CONFIRM_LIVE:
                log(
                    "TRADING_MODE=live sends real orders. Set CONFIRM_LIVE=YES to confirm. Not starting."
                )
                return False
        if TRADING_MODE not in ("sandbox", "live"):
            log("TRADING_MODE must be sandbox or live. Not starting.")
            return False
        held = self.b.position()
        if held is None:
            log("Could not read the position book. Not starting.")
            return False
        if held and not self.trade:
            log(
                f"{SYMBOL} already has an open {PRODUCT} position ({held}) this bot did not open. Close it first."
            )
            return False
        if self.trade and held == 0:
            log("Saved position is flat at the broker; clearing it (closed outside the bot).")
            self.trade, self.stop_oid, self.stop_sent = None, None, None
        log(
            f"{'SANDBOX (Analyze mode)' if mode else 'LIVE'} trading {SYMBOL}, strategy equity Rs {self.gov.equity:,.0f}"
        )
        return True

    # --- data
    def bars(self, now: datetime) -> pd.DataFrame | None:
        if not self.use_quotes:
            try:
                df = self.client.history(
                    symbol=SYMBOL,
                    exchange=EXCHANGE,
                    interval="5m",
                    start_date=(now - timedelta(days=7)).date().isoformat(),
                    end_date=now.date().isoformat(),
                )
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return closed_only(to_ist(df), now)
                msg = df.get("message", df) if isinstance(df, dict) else "no bars"
            except Exception as e:
                msg = str(e)
            if DATA_SOURCE == "history":
                log(f"history unavailable ({msg})")
                return None
            log(f"history unavailable ({msg}); building bars from live quotes from now on")
            self.use_quotes = True
        samples = load_json(QUOTES_PATH)
        if samples.get("date") != now.date().isoformat():
            return None
        return closed_only(quote_bars(samples["samples"]), now)

    def record_quote(self, now: datetime, ltp: float) -> None:
        if not self.use_quotes:
            return
        s = load_json(QUOTES_PATH)
        if s.get("date") != now.date().isoformat():
            s = {"date": now.date().isoformat(), "samples": []}
        s["samples"].append([now.isoformat(), ltp])
        save_json(QUOTES_PATH, s)

    # --- orders
    def exit_action(self) -> str:
        return "SELL" if self.trade.side > 0 else "BUY"

    def place_stop(self) -> None:
        t = self.trade
        trig = round_tick(t.stop, "down" if t.side > 0 else "up")
        self.stop_oid = self.b.order(self.exit_action(), t.open_qty, "SL-M", trig)
        self.stop_sent = trig if self.stop_oid else None
        if not self.stop_oid:
            log(
                "Broker stop order could not be placed: the in-script stop is the only protection now."
            )

    def sync_stop(self) -> None:
        t = self.trade
        if not t:
            return
        if not self.stop_oid:
            self.place_stop()
            return
        trig = round_tick(t.stop, "down" if t.side > 0 else "up")
        if self.stop_sent is None or abs(trig - self.stop_sent) >= TICK_SIZE - 1e-9:
            if self.b.modify_stop(self.stop_oid, self.exit_action(), t.open_qty, trig):
                log(f"stop moved {self.stop_sent} -> {trig}")
                self.stop_sent = trig

    def enter(self, side: int, stop: float, ref: float, now: datetime) -> None:
        ok, why = self.gov.can_enter()
        if not ok:
            log(f"signal skipped: {why}")
            return
        ltp = self.b.ltp()
        if ltp is None or abs(ltp - ref) / ref * 100 > PRICE_SANITY_MAX_PCT:
            log(f"signal skipped: live price {ltp} too far from signal close {ref:.2f}")
            return
        if side * (ltp - stop) <= 0:
            log("signal skipped: price is already through the stop")
            return
        held = self.b.position()
        if held is None or held != 0:
            log(f"signal skipped: position is {'unknown' if held is None else held}, not flat")
            return
        mult = self.gov.size_mult()
        if TRADING_MODE == "live" and self.gov.live_fills < RAMP_IN_TRADES:
            mult *= RAMP_IN_SIZE_PCT
        qty = size_qty(self.gov.equity, RISK_PCT * mult, ltp, stop)
        if qty < 1:
            log("signal skipped: size rounds to zero shares")
            return
        oid = self.b.order("BUY" if side > 0 else "SELL", qty)
        if not oid:
            return
        st, px = self.b.wait(oid)
        if st != "complete":
            held = self.b.position()
            if not held:
                log(f"entry {st}, no position: staying flat")
                return
            qty = abs(held)
        fill = px or ltp
        self.trade = Trade(side, fill, qty, stop, now.isoformat())
        self.gov.record_entry()
        log(
            f"ENTER {'LONG' if side > 0 else 'SHORT'} {qty} @ {fill:.2f} | stop {stop:.2f} "
            f"| risk Rs {abs(fill - stop) * qty:.0f} | size x{mult:.2f}"
        )
        write_trade(now, "ENTRY", "LONG" if side > 0 else "SHORT", qty, fill, "signal", None)
        if side * (fill - stop) <= 0:
            self.exit_all("stop", fill, now)
            return
        self.place_stop()
        self.save()

    def exit_part(self, qty: int, reason: str, ltp: float, now: datetime) -> None:
        oid = self.b.order(self.exit_action(), qty)
        if not oid:
            return
        st, px = self.b.wait(oid)
        self.trade.fill(qty, px or ltp, reason)
        log(f"PARTIAL {qty} @ {px or ltp:.2f}, {self.trade.open_qty} left")
        write_trade(now, "PARTIAL", self.side_label(), qty, px or ltp, reason, None)
        if self.stop_oid:
            self.b.modify_stop(
                self.stop_oid,
                self.exit_action(),
                self.trade.open_qty,
                self.stop_sent or self.trade.stop,
            )
        self.save()

    def exit_all(self, reason: str, ltp: float, now: datetime) -> None:
        t = self.trade
        if self.stop_oid:
            self.b.cancel(self.stop_oid)
            st, px = self.b.wait(self.stop_oid)
            if st == "complete":
                return self.book(px or t.stop, t.stop_reason(), now)
            if st not in ("cancelled", "rejected"):
                log(f"stop order cancel not confirmed ({st}); retrying next loop")
                return
            self.stop_oid = self.stop_sent = None
        held = self.b.position()
        if held is None:
            log("position unknown; exit retried next loop")
            return
        if held == 0:
            return self.book(ltp, reason, now)
        oid = self.b.order(self.exit_action(), abs(held))
        if not oid:
            log(f"{reason} exit refused; retrying next loop")
            return
        st, px = self.b.wait(oid)
        if st == "rejected":
            log(f"{reason} exit rejected; retrying next loop")
            return
        self.book(px or ltp, reason, now)

    def book(self, px: float, reason: str, now: datetime) -> None:
        t = self.trade
        if t.open_qty > 0:
            t.fill(t.open_qty, px, reason)
        pnl = t.pnl()
        self.gov.record_close(pnl, live=TRADING_MODE == "live")
        log(
            f"EXIT {reason} @ {px:.2f} | trade P&L Rs {pnl:+.0f} ({pnl / (t.risk * t.qty):+.2f}R) | "
            f"day Rs {self.gov.day_pnl:+.0f} | equity Rs {self.gov.equity:,.0f} ({self.gov.dd_pct():.1f}% below peak)"
        )
        write_trade(now, "EXIT", self.side_label(), t.qty, px, reason, round(pnl, 2))
        self.trade, self.stop_oid, self.stop_sent = None, None, None
        self.last_exit_bar = self.last_bar
        self.save()

    def side_label(self) -> str:
        return "LONG" if self.trade and self.trade.side > 0 else "SHORT"

    # --- loop pieces
    def manage_tick(self, ltp: float, now: datetime) -> None:
        t = self.trade
        if now.time() >= EOD_EXIT:
            return self.exit_all("eod", ltp, now)
        if self.stop_oid and time.time() - self.last_status >= STATUS_EVERY_SEC:
            self.last_status = time.time()
            st, px = self.b.status(self.stop_oid)
            if st == "complete":
                return self.book(px or t.stop, t.stop_reason(), now)
            if st in ("cancelled", "rejected"):
                log(f"broker stop order {st}; placing it again")
                self.stop_oid = self.stop_sent = None
                self.place_stop()
        hit = t.check_price(ltp)
        if hit and hit[0] == "PARTIAL":
            self.exit_part(t.partial_qty, "partial", ltp, now)
        elif hit and hit[0] == "EXIT":
            beyond = t.side * (t.stop - ltp) >= t.stop * SOFT_STOP_SLACK_PCT / 100
            if not self.stop_oid or beyond:
                self.exit_all(hit[1], ltp, now)

    def on_bar(self, d: pd.DataFrame, now: datetime) -> None:
        last = d.iloc[-1]
        if self.trade:
            hit = self.trade.on_bar_close(last["high"], last["low"], last["close"], last["atr"])
            if hit:
                return self.exit_all(hit[1], float(last["close"]), now)
            self.sync_stop()
            self.save()
            return
        if (
            self.last_exit_bar
            and len(d)
            and (d.index[-1] - pd.Timestamp(self.last_exit_bar))
            < pd.Timedelta(minutes=COOLDOWN_BARS * BAR_MINUTES)
        ):
            return
        side, stop = candidates(d)
        if side[-1]:
            log(
                f"breakout {'LONG' if side[-1] > 0 else 'SHORT'} on the {d.index[-1]:%H:%M} bar, stop {stop[-1]:.2f}"
            )
            self.enter(int(side[-1]), float(stop[-1]), float(last["close"]), now)

    def step(self) -> None:
        now = now_ist()
        self.gov.roll(now.date())
        if now.weekday() >= 5 or not (MARKET_OPEN <= now.time() < MARKET_CLOSE):
            return
        ltp = self.b.ltp()
        if ltp is None:
            return
        self.record_quote(now, ltp)
        if self.trade:
            self.manage_tick(ltp, now)
        bar_key = pd.Timestamp(now).floor(f"{BAR_MINUTES}min").isoformat()
        if bar_key != self.last_bar and now.second >= 3:
            d = self.bars(now)
            self.last_bar = bar_key
            if (
                d is not None
                and len(d) > max(EMA_SLOW, ATR_PERIOD, CHANNEL_BARS) + 1
                and d.index[-1].date() == now.date()
            ):
                self.on_bar(prepare(d), now)
            self.save()

    def run(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if not self.preflight():
            return
        self.save()
        log(
            f"risk {RISK_PCT:.2%}/trade | max {MAX_TRADES_PER_DAY} trades/day | daily limit {DAILY_LOSS_R}R or "
            f"{DAILY_LOSS_PCT}% | weekly {WEEKLY_LOSS_PCT}% | half size at -{DD_THROTTLE_PCT}%, halt at -{DD_HALT_PCT}%"
        )
        while True:
            try:
                self.step()
            except KeyboardInterrupt:
                log(
                    "stopped by user"
                    + (
                        " - a position is still open, the broker stop stays in place"
                        if self.trade
                        else ""
                    )
                )
                self.save()
                return
            except Exception as e:
                log(f"loop error: {e}")
                time.sleep(5)
            time.sleep(POLL_SECONDS)


def closed_only(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    return df[df.index + pd.Timedelta(minutes=BAR_MINUTES) <= pd.Timestamp(now)]


def quote_bars(samples: list) -> pd.DataFrame:
    if not samples:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    s = pd.DataFrame(samples, columns=["ts", "ltp"])
    s["ts"] = pd.to_datetime(s["ts"], format="ISO8601")
    s = s.set_index("ts").sort_index()
    s.index = s.index.tz_convert(IST) if s.index.tz is not None else s.index.tz_localize(IST)
    bars = (
        s["ltp"]
        .astype(float)
        .resample(f"{BAR_MINUTES}min")
        .agg(["first", "max", "min", "last"])
        .dropna()
    )
    bars.columns = ["open", "high", "low", "close"]
    bars["volume"] = 0.0
    return bars


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError) as e:
        log(f"could not read {path.name}: {e}")
        return {}


def save_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        log(f"could not save {path.name}: {e}")


def write_trade(
    now: datetime, event: str, side: str, qty: int, price: float, reason: str, pnl
) -> None:
    try:
        new = not TRADES_PATH.exists()
        TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADES_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "mode", "event", "side", "qty", "price", "reason", "pnl"])
            w.writerow(
                [
                    now.isoformat(timespec="seconds"),
                    TRADING_MODE,
                    event,
                    side,
                    qty,
                    round(price, 2),
                    reason,
                    pnl,
                ]
            )
    except OSError as e:
        log(f"could not write trade log: {e}")


# ---------------------------------------------------------------- read-only commands --
def report() -> None:
    if not TRADES_PATH.exists():
        log("No trades yet.")
        return
    t = pd.read_csv(TRADES_PATH)
    closed = t[t.event == "EXIT"]
    if closed.empty:
        log("No closed trades yet.")
        return
    for mode, g in closed.groupby("mode"):
        wins = g[g.pnl > 0]
        log(
            f"{mode}: {len(g)} trades, win {len(wins) / len(g):.0%}, net Rs {g.pnl.sum():+,.0f}, exits {g.reason.value_counts().to_dict()}"
        )


def status() -> None:
    s = load_json(STATE_PATH)
    g = Governor.from_dict(s["gov"]) if s.get("gov") else Governor(CAPITAL_RUPEES)
    ok, why = g.can_enter()
    log(
        f"equity Rs {g.equity:,.0f}, peak Rs {g.peak:,.0f}, {g.dd_pct():.1f}% below peak, size x{g.size_mult():.2f}"
    )
    log(f"today: {g.day_trades} trades, P&L Rs {g.day_pnl:+.0f}; week P&L Rs {g.week_pnl:+.0f}")
    log("entries allowed" if ok else f"entries blocked: {why}")
    if s.get("trade"):
        t = s["trade"]
        log(
            f"OPEN {'LONG' if t['side'] > 0 else 'SHORT'} {t['open_qty']} @ {t['entry']:.2f}, stop {t['stop']:.2f}"
        )


def reset_breaker() -> None:
    s = load_json(STATE_PATH)
    g = Governor.from_dict(s["gov"]) if s.get("gov") else Governor(CAPITAL_RUPEES)
    g.halted = False
    g.peak = g.equity  # the reset accepts the current equity as the new reference
    s["gov"] = g.to_dict()
    save_json(STATE_PATH, s)
    log(f"drawdown halt cleared; new peak reference Rs {g.equity:,.0f}")


# ---------------------------------------------------------------- self-test --
def selftest() -> bool:
    global LOG_DIR, STATE_PATH, TRADES_PATH, QUOTES_PATH, now_ist
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    LOG_DIR, STATE_PATH, TRADES_PATH, QUOTES_PATH = (
        tmp,
        tmp / "s.json",
        tmp / "t.csv",
        tmp / "q.json",
    )
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("1) trade manager")
    t = Trade(1, 100.0, 10, 99.0, "x")
    t.on_bar_close(101.0, 100.2, 100.8, 0.5)
    check("breakeven at +1R moves the stop to entry plus costs", abs(t.stop - 100.08) < 1e-6)
    check("partial fires at +1.5R", t.check_price(101.5) == ("PARTIAL", "partial"))
    t.fill(t.partial_qty, 101.5, "partial")
    check("partial only once", t.check_price(101.6) is None and t.open_qty == 5)
    t.on_bar_close(103.0, 101.5, 102.9, 0.5)
    check("trail tightens to 1.5 ATR after +2.5R", abs(t.stop - 102.25) < 1e-6)
    check("trail exit is labelled trail", t.check_price(102.2) == ("EXIT", "trail"))
    t2 = Trade(-1, 100.0, 10, 101.0, "x")
    for _ in range(TIME_STOP_BARS - 1):
        check_none = t2.on_bar_close(100.3, 99.9, 100.1, 0.5)
    check("no time stop before TIME_STOP_BARS", check_none is None)
    check(
        "time stop after TIME_STOP_BARS without +0.5R",
        t2.on_bar_close(100.3, 99.9, 100.1, 0.5) == ("EXIT", "time"),
    )
    t3 = Trade(-1, 100.0, 10, 101.0, "x")
    check("short stop hit", t3.check_price(101.0) == ("EXIT", "stop"))

    print("2) risk governor")
    g = Governor(100000)
    g.roll(datetime(2026, 3, 2).date())
    g.record_close(-400, False)
    g.record_close(-400, False)
    check("2 losses in a row blocks the day", not g.can_enter()[0])
    g.roll(datetime(2026, 3, 3).date())
    check("new day unblocks", g.can_enter()[0])
    g.record_close(-5500, False)
    check("5% below peak halves size", g.size_mult() == 0.5)
    g.roll(datetime(2026, 3, 9).date())
    g.record_close(-5000, False)
    check("10% below peak halts", g.size_mult() == 0.0 and not g.can_enter()[0])
    g.record_close(20000, False)
    check("halt stays until reset even after a win", g.halted and g.size_mult() == 0.0)
    g2 = Governor(100000)
    g2.roll(datetime(2026, 3, 2).date())
    g2.record_close(-4100, False)
    g2.roll(datetime(2026, 3, 3).date())
    check("weekly loss limit blocks the rest of the week", not g2.can_enter()[0])

    print("3) sizing")
    check(
        "risk sizing",
        size_qty(100000, 0.005, 100, 99)
        == int(500 // (1 + round_trip_cost(100, 100, 1000, 1) / 1000 + 0.06)),
    )
    check(
        "notional cap binds on a tight stop",
        size_qty(20000, 0.005, 100, 99.99) == int(min(MAX_NOTIONAL_RUPEES, 60000) // 100),
    )

    print("4) entry filter and backtest")
    rng = np.random.default_rng(3)
    frames = []
    for k, dday in enumerate(pd.bdate_range("2026-01-01", periods=120)):
        idx = pd.date_range(dday + pd.Timedelta("9h15m"), periods=75, freq="5min", tz=IST)
        trend = rng.choice([-1, 0, 0, 1]) * 0.0012
        r = rng.normal(trend, 0.002, 75)
        c = 500 * np.exp(np.cumsum(r)) * (1 + 0.001 * k)
        o = np.r_[c[0], c[:-1]]
        h = np.maximum(o, c) * (1 + rng.uniform(0, 0.001, 75))
        lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.001, 75))
        frames.append(
            pd.DataFrame(
                {"open": o, "high": h, "low": lo, "close": c, "volume": rng.integers(1e4, 5e4, 75)},
                index=idx,
            )
        )
    df = pd.concat(frames)
    side, _ = candidates(prepare(df))
    global ER_MIN, USE_TREND_FILTER
    saved = (ER_MIN, USE_TREND_FILTER)
    ER_MIN, USE_TREND_FILTER = 0.0, False
    side_raw, _ = candidates(prepare(df))
    ER_MIN, USE_TREND_FILTER = saved
    check(
        f"filters cut chop signals ({np.count_nonzero(side)} vs {np.count_nonzero(side_raw)} unfiltered)",
        0 < np.count_nonzero(side) < np.count_nonzero(side_raw),
    )
    t, gov = backtest_frame(df, 100000)
    check("backtest produced trades", len(t) > 0)
    if len(t):
        check(
            "never more than MAX_TRADES_PER_DAY per day",
            t.groupby("day").size().max() <= MAX_TRADES_PER_DAY,
        )
        check("stops cost about 1R, never much more", t.r.min() > -1.6)
        known = {"stop", "breakeven", "trail", "partial", "time", "eod"}
        check("every exit has a known reason", set(t.reason) <= known)
        summarize(t, 100000, "synthetic")

    print("5) live order flow (fake broker)")

    class Fake:
        def __init__(self, analyze=True):
            self.analyze, self.orders, self.mods, self.cancels, self.pos = analyze, [], [], [], 0
            self.px = 100.0

        def analyzerstatus(self):
            return {"status": "success", "data": {"analyze_mode": self.analyze}}

        def quotes(self, **k):
            return {"status": "success", "data": {"ltp": self.px}}

        def placeorder(self, **k):
            self.orders.append(k)
            if k["price_type"] == "MARKET":
                self.pos += k["quantity"] if k["action"] == "BUY" else -k["quantity"]
            return {"status": "success", "orderid": str(len(self.orders))}

        def modifyorder(self, **k):
            self.mods.append(k)
            return {"status": "success"}

        def cancelorder(self, **k):
            self.cancels.append(k["order_id"])
            return {"status": "success"}

        def orderstatus(self, order_id, **k):
            o = self.orders[int(order_id) - 1]
            if o["price_type"] == "SL-M":
                return {
                    "status": "success",
                    "data": {
                        "order_status": "cancelled"
                        if order_id in self.cancels
                        else "trigger pending"
                    },
                }
            return {
                "status": "success",
                "data": {"order_status": "complete", "average_price": self.px},
            }

        def openposition(self, **k):
            return {"status": "success", "quantity": self.pos}

    real_now = now_ist
    clock_t = {"t": datetime(2026, 3, 2, 10, 0, 5, tzinfo=IST)}
    now_ist = lambda: clock_t["t"]  # noqa: E731
    try:
        f = Fake()
        lv = Live(f)
        check("sandbox starts with Analyze mode on", lv.preflight())
        lv.enter(1, 99.0, 100.0, clock_t["t"])
        check(
            "entry is a MARKET buy followed by an SL-M stop",
            [o["price_type"] for o in f.orders] == ["MARKET", "SL-M"],
        )
        check(
            "stop order sits at the stop",
            f.orders[1]["trigger_price"] == 99.0 and f.orders[1]["action"] == "SELL",
        )
        f.px = 101.2
        bar = pd.DataFrame(
            {"open": [100.5], "high": [101.2], "low": [100.4], "close": [101.1], "atr": [0.4]},
            index=[pd.Timestamp("2026-03-02 10:00", tz=IST)],
        )
        lv.on_bar(bar, clock_t["t"])
        check(
            "breakeven is pushed to the broker stop", f.mods and f.mods[-1]["trigger_price"] > 100.0
        )
        f.px = 101.6
        lv.manage_tick(101.6, clock_t["t"])
        check(
            "partial exit sells half at market",
            f.orders[-1]["price_type"] == "MARKET"
            and f.orders[-1]["quantity"] == lv.trade.qty // 2,
        )
        check(
            "stop order resized to the remaining shares",
            f.mods[-1]["quantity"] == lv.trade.open_qty,
        )
        clock_t["t"] = datetime(2026, 3, 2, 15, 16, tzinfo=IST)
        lv.manage_tick(101.0, clock_t["t"])
        check(
            "EOD cancels the stop order and flattens", f.cancels and lv.trade is None and f.pos == 0
        )
        check("closed trade reaches the governor", lv.gov.day_trades == 1 and lv.gov.realized != 0)
        check("sandbox refuses while OpenAlgo is live", not Live(Fake(analyze=False)).preflight())
        global TRADING_MODE
        TRADING_MODE = "live"
        check("live refuses without CONFIRM_LIVE", not Live(Fake(analyze=False)).preflight())
        TRADING_MODE = "sandbox"
    finally:
        now_ist = real_now
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser(description="ORB Delta Pro v2")
    ap.add_argument("cmd", nargs="?", default=None)
    args = ap.parse_args()
    cmd = (args.cmd or env("MODE", "monitor")).lower()
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if cmd in ("report", "status", "reset-breaker"):
        {"report": report, "status": status, "reset-breaker": reset_breaker}[cmd]()
        return
    if not API_KEY:
        sys.exit("Set OPENALGO_API_KEY first (the /python host injects it automatically).")
    from openalgo import api

    client = api(api_key=API_KEY, host=HOST)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if cmd == "backtest":
        run_backtest(client)
    elif cmd == "scan":
        lv = Live(client)
        d = lv.bars(now_ist())
        if d is None or d.empty:
            log("no bars available")
            return
        side, stop = candidates(prepare(d))
        log(
            f"last closed bar {d.index[-1]}: {'LONG' if side[-1] > 0 else 'SHORT' if side[-1] < 0 else 'no signal'}"
        )
    elif cmd == "monitor":
        Live(client).run()
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
