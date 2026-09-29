#!/usr/bin/env python
"""
NIFTY Trend-Pullback Scalper (ATM option buying)

Idea
----
Scalps that work are taken WITH the intraday trend, after a pause, at a moment
when the trend resumes. Scalps that fail are usually taken in chop, against the
trend, or chasing an extended move. The rules below try to keep only the first
kind:

  1. Trend      EMA9 above EMA21, EMA21 rising, close above the session average
                price (mirror image for shorts).
  2. Pullback   one of the last PULLBACK_BARS bars dipped to EMA9.
  3. Trigger    the just-closed bar closes beyond the previous bar's high (low),
                i.e. the trend is resuming now.
  4. Regime     ATR(14) above MIN_ATR (skip dead, choppy tape) and price not
                more than MAX_EXTENSION_ATR away from EMA21 (skip chasing).
  5. Time       only in the liquid windows (default 09:30-11:30, 13:30-14:45).

A long signal buys the ATM CE, a short signal buys the ATM PE. The signal and
all exits are measured on the NIFTY index itself, so the backtest and the live
run make exactly the same decisions:

  stop     SL_ATR x ATR from the entry
  target   RR x stop distance
  trail    after +1R the stop moves to breakeven, then trails TRAIL_ATR x ATR
           behind the best price seen
  time     a scalp that has not worked in MAX_HOLD_BARS minutes is closed
  day      MAX_TRADES_PER_DAY, MAX_CONSECUTIVE_LOSSES and MAX_DAILY_LOSS stop
           new entries for the day; everything is squared off at SQUARE_OFF

Edge is not a promise
---------------------
No script has an edge by construction. Run MODE=backtest first: it replays the
exact same signal and exit code over NIFTY 1-minute history, charges costs, and
prints expectancy separately for the first 70% and last 30% of days. Only trade
it live if BOTH halves are positive after costs, and start in Analyze (sandbox)
mode. The backtest converts index points to option premium with a fixed DELTA,
which is an approximation: real ATM premiums also lose theta and move with IV.

Running
-------
Upload it at /python (exchange NFO), or run it standalone:

    MODE=backtest uv run python strategies/examples/nifty_pullback_scalper.py
    MODE=live     uv run python strategies/examples/nifty_pullback_scalper.py

Every setting below can be overridden with an environment variable of the same
name (set them as strategy parameters in /python).
"""

import os
import threading
import time
from datetime import datetime, timedelta
from datetime import time as dtime
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

MODE = env("MODE", "backtest").lower()  # backtest | live
STRATEGY = env("STRATEGY_NAME", "NIFTY Pullback Scalper")

UNDERLYING = env("UNDERLYING", "NIFTY")
INDEX_EXCHANGE = env("INDEX_EXCHANGE", "NSE_INDEX")
OPTION_EXCHANGE = env("OPTION_EXCHANGE", "NFO")
PRODUCT = env("PRODUCT", "MIS")  # MIS or NRML, never CNC for options
LOTS = env("LOTS", 1, int)
OFFSET = env("OFFSET", "ATM")

EMA_FAST = env("EMA_FAST", 9, int)
EMA_SLOW = env("EMA_SLOW", 21, int)
ATR_LEN = env("ATR_LEN", 14, int)
SLOPE_BARS = env("SLOPE_BARS", 5, int)
PULLBACK_BARS = env("PULLBACK_BARS", 3, int)
MIN_ATR = env("MIN_ATR", 6.0, float)  # index points on 1m bars
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

IST = ZoneInfo("Asia/Kolkata")


def log(msg):
    print(f"[{datetime.now(IST):%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


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


def signal_at(df, i):
    """Return 'LONG', 'SHORT' or None for the CLOSED bar at position i."""
    if i < max(EMA_SLOW, ATR_LEN, SLOPE_BARS, PULLBACK_BARS) + 1:
        return None
    bar = df.iloc[i]
    prev = df.iloc[i - 1]
    if not in_window(df.index[i].time()):
        return None
    # Every lookback must stay inside today's session: yesterday's close is
    # not a pullback.
    if df.index[i - PULLBACK_BARS].date() != df.index[i].date():
        return None

    atr = bar["atr"]
    if atr < MIN_ATR:
        return None
    if abs(bar["close"] - bar["ema_slow"]) > MAX_EXTENSION_ATR * atr:
        return None

    recent = df.iloc[i - PULLBACK_BARS : i]
    slope = bar["ema_slow"] - df["ema_slow"].iloc[i - SLOPE_BARS]

    long_trend = (
        bar["ema_fast"] > bar["ema_slow"] and slope > 0 and bar["close"] > bar["session_avg"]
    )
    long_pullback = (recent["low"] <= recent["ema_fast"]).any()
    long_trigger = bar["close"] > prev["high"] and bar["close"] > bar["open"]
    if long_trend and long_pullback and long_trigger:
        return "LONG"

    short_trend = (
        bar["ema_fast"] < bar["ema_slow"] and slope < 0 and bar["close"] < bar["session_avg"]
    )
    short_pullback = (recent["high"] >= recent["ema_fast"]).any()
    short_trigger = bar["close"] < prev["low"] and bar["close"] < bar["open"]
    if short_trend and short_pullback and short_trigger:
        return "SHORT"
    return None


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
        if gain >= self.risk:
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
def backtest(client):
    end = datetime.now(IST).date()
    start = end - timedelta(days=BACKTEST_DAYS)
    df = client.history(
        symbol=UNDERLYING,
        exchange=INDEX_EXCHANGE,
        interval="1m",
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        source=HISTORY_SOURCE,
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        log(f"No history returned: {df}")
        return
    df = add_indicators(df)
    log(f"Backtesting {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    guard = DayGuard()
    trades = []
    i = 0
    n = len(df)
    while i < n - 1:
        ts = df.index[i]
        guard.roll(ts.date())
        side = signal_at(df, i) if guard.can_trade() else None
        if side is None:
            i += 1
            continue

        # Signal on bar i's close, filled at bar i+1's open.
        j = i + 1
        if df.index[j].date() != ts.date():
            i += 1
            continue
        trade = Trade(side, df["open"].iloc[j], df["atr"].iloc[i], df.index[j])
        exit_price, reason = None, None
        k = j
        while k < n and df.index[k].date() == ts.date():
            row = df.iloc[k]
            # Worst case inside a bar: the adverse extreme comes first.
            adverse, favourable = (
                (row["low"], row["high"]) if side == "LONG" else (row["high"], row["low"])
            )
            for price in (adverse, favourable, row["close"]):
                reason = trade.update(price)
                if reason:
                    exit_price = trade.target if reason == "TARGET" else trade.stop
                    break
            if reason:
                break
            if k - j + 1 >= MAX_HOLD_BARS:
                reason, exit_price = "TIME", row["close"]
                break
            if df.index[k].time() >= SQUARE_OFF:
                reason, exit_price = "SQUARE_OFF", row["close"]
                break
            k += 1
        if reason is None:
            k = min(k, n - 1)
            reason, exit_price = "EOD", df["close"].iloc[k]

        index_pts = trade.points(exit_price)
        premium_pts = index_pts * DELTA - COST_PREMIUM_POINTS
        rupees = premium_pts * BACKTEST_LOT_SIZE * LOTS
        guard.record(rupees)
        trades.append(
            {
                "day": ts.date(),
                "entry_time": trade.entry_time,
                "side": side,
                "reason": reason,
                "index_pts": index_pts,
                "premium_pts": premium_pts,
                "rupees": rupees,
            }
        )
        i = k + 1

    if not trades:
        log("No trades. Loosen MIN_ATR or widen the windows, then re-test.")
        return
    report(pd.DataFrame(trades))


def summarize(t, label):
    wins = t[t["rupees"] > 0]
    losses = t[t["rupees"] <= 0]
    gross_win = wins["rupees"].sum()
    gross_loss = -losses["rupees"].sum()
    equity = t["rupees"].cumsum()
    drawdown = (equity - equity.cummax()).min()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    log(
        f"{label:<14} trades {len(t):>4} | win {len(wins) / len(t):6.1%} | "
        f"avg win {wins['rupees'].mean() if len(wins) else 0:8.0f} | "
        f"avg loss {losses['rupees'].mean() if len(losses) else 0:8.0f} | "
        f"PF {pf:5.2f} | expectancy {t['rupees'].mean():7.0f}/trade | "
        f"net {t['rupees'].sum():9.0f} | max DD {drawdown:8.0f}"
    )
    return t["rupees"].mean()


def report(t):
    days = sorted(t["day"].unique())
    cut = days[int(len(days) * 0.7)] if len(days) >= 5 else None
    log("-" * 110)
    log(
        f"Costs: {COST_PREMIUM_POINTS} premium pts/round trip, delta {DELTA}, lot {BACKTEST_LOT_SIZE} x {LOTS}"
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
    log("-" * 110)
    if verdict is None:
        log("Too few days to judge. Raise BACKTEST_DAYS.")
    elif verdict:
        log(
            "Positive after costs in both halves. Next step: run MODE=live in Analyze mode for a few weeks."
        )
    else:
        log("NO EDGE on this data after costs. Do not trade this live as configured.")


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
        today = datetime.now(IST).date()
        df = self.client.history(
            symbol=UNDERLYING,
            exchange=INDEX_EXCHANGE,
            interval="1m",
            start_date=(today - timedelta(days=5)).strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        now_minute = pd.Timestamp(datetime.now(IST)).floor("min")
        df = df[df.index < now_minute]  # drop the bar still forming
        return add_indicators(df)

    def run(self):
        self.resolve_contract()
        self.client.connect()
        self.client.subscribe_ltp(
            [{"exchange": INDEX_EXCHANGE, "symbol": UNDERLYING}], on_data_received=self.on_tick
        )
        log(f"Live. Windows {WINDOWS}, square-off {SQUARE_OFF}")
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
                    elif held >= MAX_HOLD_BARS:
                        self.claim_exit("TIME")

                if now.time() >= SQUARE_OFF and self.trade is None:
                    log("Square-off time reached and flat. Done for the day.")
                    break

                # Evaluate once per closed bar, a couple of seconds after it closes.
                if self.trade is None and now.second >= 2 and in_window(now.time()):
                    bar_key = now.replace(second=0, microsecond=0)
                    if bar_key != last_bar:
                        last_bar = bar_key
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
        i = len(df) - 1
        if df.index[i].date() != datetime.now(IST).date():
            return
        side = signal_at(df, i)
        if side:
            self.enter(side, float(df["atr"].iloc[i]))


def main():
    if not API_KEY:
        log("OPENALGO_API_KEY is not set. Generate one at /apikey.")
        return
    client = api(api_key=API_KEY, host=HOST, ws_url=WS_URL)
    if MODE == "live":
        LiveTrader(client).run()
    else:
        backtest(client)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
