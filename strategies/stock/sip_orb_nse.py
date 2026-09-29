"""
sip_orb_nse.py - "Stocks in Play" Opening Range Breakout for NSE intraday (MIS) via OpenAlgo
============================================================================================

WHY THIS AND NOT ANOTHER INDICATOR COMBO
----------------------------------------
Your earlier bots (1-min VWAP/Bollinger fades, GainzAlgo-style signals, EMA crossovers) all try to
predict the next few bars of an ordinary stock on an ordinary day from price patterns. On a normal
day a liquid large-cap is close to a random walk at 1-5 min horizons, and Indian round-trip costs
(0.1%-per-leg brokerage below Rs 20k, STT, stamp, GST) are bigger than the pattern's edge. That is
why they keep "almost" working and then bleed.

This strategy trades a different thing: an *information imbalance*. On a few days each stock has
news (results, orders, block deals, rating changes, sector news) and trades several times its normal
opening volume. On those days big players are repositioning and cannot finish in 5 minutes, so the
direction of the first 5 minutes tends to continue for the rest of the session. That is the finding
of Zarattini, Barbon & Aziz (2024, "A Profitable Day Trading Strategy For The U.S. Equity Market",
SSRN 4729284): a plain 5-min ORB was weak on all stocks, but restricted to the day's top stocks by
OPENING RELATIVE VOLUME it was strongly profitable after costs. Relative volume did almost all the
work: RVOL < 1 lost money, RVOL > 1 made money, very high RVOL made the most.

  Edge source   : who is trading (unusual participation), not what a chart pattern looks like.
  Payoff shape  : low win rate (~30-40%), small fixed losses, occasional large all-day winners.
                  You MUST let winners run to the close - cutting them early destroys the edge.
  Trades        : a handful per day, only in stocks that are actually "in play".

HONEST STATUS: the research is on US stocks. It has NOT been verified on NSE. Indian differences
that matter: pre-open auction (09:00-09:08) already absorbs part of the overnight news, MIS short
availability, higher STT/stamp costs, circuit limits. So run MODE=backtest on your own history first
(it prints the RVOL-bucket table that tells you whether the edge exists here), then paper/sandbox
trade, then go live small.

RULES (all evaluated from completed 5-minute bars)
--------------------------------------------------
Every morning, for every stock in UNIVERSE:
  1. Opening range (OR) = the 09:15-09:20 5-min candle.
  2. RVOL = OR volume / average OR volume of the previous RVOL_DAYS (14) sessions.
  3. Filters: price >= MIN_PRICE, 14-day average turnover >= MIN_TURNOVER_CR, RVOL >= MIN_RVOL,
     OR candle not a doji, OR range between MIN_OR_ATR and MAX_OR_ATR of the 14-day daily ATR.
  4. Keep the TOP_N stocks by RVOL. These are today's "stocks in play".
  5. Direction = colour of the OR candle: green -> only a LONG above OR high,
                                          red   -> only a SHORT below OR low (if ALLOW_SHORT).
  6. Entry  : price trades beyond the OR (+1 tick). Max chase = MAX_CHASE_PCT past the trigger.
  7. Stop   : the other side of the OR (STOP_MODE=or, default), capped at MAX_STOP_ATR x ATR,
              or STOP_ATR_FRAC x ATR (STOP_MODE=atr, the paper's 10% ATR stop).
              Placed at the broker as an SL-M/SL order right after the fill - it survives a crash.
  8. Exit   : at SQUAREOFF_TIME (15:10) at market, unless stopped. Optional TARGET_R (0 = off).
  9. Size   : risk RISK_PCT of CAPITAL_INR per trade (stop distance + estimated costs), and the
              total MIS notional of all positions <= CAPITAL_INR x MAX_LEVERAGE.
  10. One trade per stock per day. No new entries after ENTRY_CUTOFF. DAILY_LOSS_LIMIT_R stops the day.

HOW ORDERS FIRE (TRADING_MODE)
------------------------------
  paper    Fills are simulated inside this script. Nothing reaches OpenAlgo, so the OpenAlgo order
           book stays empty. Useful offline; not a test of the real order path.
  sandbox  Real orders through OpenAlgo with Analyze mode ON. They appear in OpenAlgo's sandbox
           order book, positions and P&L, and fill against live prices. Refuses to start if
           Analyze mode is off, so it can never send a real order.
  live     Real orders with real money. Refuses to start unless Analyze mode is OFF and
           CONFIRM_LIVE=YES is set.
  (If TRADING_MODE is not set, the old PAPER variable still decides: PAPER=0 means live.)

Entries and the software stop react to the live WebSocket price feed (FEED=ws). If the feed is
unavailable the bot falls back to REST quotes on its own.

MODES
-----
Every mode can be chosen with the MODE parameter (for the /python host, which cannot pass
command-line flags) or with the flag shown:

  MODE=run       (default)      live loop: waits for 09:20, scans, trades, squares off, repeats daily
  MODE=backtest  --backtest     history for UNIVERSE via OpenAlgo (HISTORY_SOURCE=db for Historify)
                 --csvdir data/ or CSV_DIR=data/ : data/<SYMBOL>.csv (timestamp,o,h,l,c,v)
  MODE=scan      --scan         after 09:20: print today's stocks in play, no orders
  MODE=selftest  --selftest     offline tests of the mechanics (synthetic data)

Everything is configured by environment variables (see Config). Verify your broker's charges and
that each symbol is MIS (and MIS-short) enabled. Nothing here guarantees profit.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date as ddate
from datetime import datetime, timedelta, timezone
from datetime import time as dtime

import numpy as np
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
BOT_DIR = os.getenv("BOT_DIR") or os.path.dirname(os.path.abspath(__file__))


def now_ist():
    """Single clock for the bot (the self-test patches this)."""
    return datetime.now(IST)


def bot_path(name):
    """Output files live next to the script (or BOT_DIR), not in whatever directory it was started from."""
    return name if os.path.isabs(name) else os.path.join(BOT_DIR, name)


# --------------------------------------------------------------------------- config
def _env(name, default, cast=str):
    v = os.getenv(name)
    if v is None or v == "":
        return default
    if cast is bool:
        return v.strip().lower() in ("1", "true", "yes", "y")
    return cast(v)


def _env_time(name, default_hhmm):
    h, m = _env(name, default_hhmm).split(":")
    return dtime(int(h), int(m))


def _trading_mode():
    m = _env("TRADING_MODE", "").strip().lower()
    if m in ("paper", "sandbox", "live"):
        return m
    return "paper" if _env("PAPER", True, bool) else "live"


# Liquid NSE F&O stocks (MIS-short friendly). VERIFY symbols against your OpenAlgo master contract -
# index constituents and ticker names change (e.g. corporate actions / renames).
DEFAULT_UNIVERSE = (
    "RELIANCE,HDFCBANK,ICICIBANK,INFY,TCS,SBIN,AXISBANK,KOTAKBANK,LT,ITC,BHARTIARTL,HINDUNILVR,"
    "BAJFINANCE,BAJAJFINSV,MARUTI,M&M,TATASTEEL,JSWSTEEL,HINDALCO,VEDL,SUNPHARMA,DRREDDY,CIPLA,"
    "DIVISLAB,APOLLOHOSP,TITAN,ASIANPAINT,ULTRACEMCO,GRASIM,NTPC,POWERGRID,TATAPOWER,ONGC,COALINDIA,"
    "BPCL,IOC,ADANIENT,ADANIPORTS,WIPRO,HCLTECH,TECHM,LTIM,INDUSINDBK,BANKBARODA,PNB,CANBK,"
    "SHRIRAMFIN,EICHERMOT,HEROMOTOCO,BAJAJ-AUTO,NESTLEIND,BEL,HAL,TRENT,ETERNAL,DLF,JIOFIN,"
    "CHOLAFIN,TVSMOTOR,POLYCAB"
)


@dataclass
class Config:
    # --- connection
    api_key: str = field(default_factory=lambda: _env("OPENALGO_API_KEY", ""))
    host: str = field(
        default_factory=lambda: (
            _env("HOST_SERVER", "") or _env("OPENALGO_HOST", "http://127.0.0.1:5000")
        )
    )
    ws_url: str = field(default_factory=lambda: _env("WEBSOCKET_URL", "ws://127.0.0.1:8765"))
    feed: str = field(default_factory=lambda: _env("FEED", "ws").lower())  # ws | rest
    stale_sec: float = field(
        default_factory=lambda: _env("STALE_SEC", 5.0, float)
    )  # tick older -> REST
    strategy: str = field(default_factory=lambda: _env("STRATEGY_NAME", "sip_orb_nse"))
    exchange: str = field(default_factory=lambda: _env("EXCHANGE", "NSE"))
    product: str = field(default_factory=lambda: _env("PRODUCT", "MIS"))
    trading_mode: str = field(default_factory=_trading_mode)  # paper | sandbox | live
    confirm_live: bool = field(default_factory=lambda: _env("CONFIRM_LIVE", False, bool))
    universe: str = field(default_factory=lambda: _env("UNIVERSE", DEFAULT_UNIVERSE))
    universe_file: str = field(
        default_factory=lambda: _env("UNIVERSE_FILE", "")
    )  # one symbol per line
    api_sleep: float = field(
        default_factory=lambda: _env("API_SLEEP", 0.12, float)
    )  # between REST calls
    history_source: str = field(
        default_factory=lambda: _env("HISTORY_SOURCE", "api")
    )  # backtest: api | db

    # --- money / risk
    capital_inr: float = field(default_factory=lambda: _env("CAPITAL_INR", 100000.0, float))
    risk_pct: float = field(
        default_factory=lambda: _env("RISK_PCT", 0.005, float)
    )  # 0.5% per trade
    max_leverage: float = field(
        default_factory=lambda: _env("MAX_LEVERAGE", 4.0, float)
    )  # total MIS notional
    live_max_qty: int = field(
        default_factory=lambda: _env("LIVE_MAX_QTY", 0, int)
    )  # 0 = no extra cap
    daily_loss_limit_r: float = field(
        default_factory=lambda: _env("DAILY_LOSS_LIMIT_R", 3.0, float)
    )
    tick_size: float = field(default_factory=lambda: _env("TICK_SIZE", 0.05, float))

    # --- selection ("stocks in play")
    top_n: int = field(default_factory=lambda: _env("TOP_N", 5, int))
    min_rvol: float = field(default_factory=lambda: _env("MIN_RVOL", 2.0, float))
    rvol_days: int = field(default_factory=lambda: _env("RVOL_DAYS", 14, int))
    atr_days: int = field(default_factory=lambda: _env("ATR_DAYS", 14, int))
    min_price: float = field(default_factory=lambda: _env("MIN_PRICE", 50.0, float))
    min_turnover_cr: float = field(
        default_factory=lambda: _env("MIN_TURNOVER_CR", 50.0, float)
    )  # avg daily Rs cr
    min_or_atr: float = field(
        default_factory=lambda: _env("MIN_OR_ATR", 0.08, float)
    )  # OR too tiny -> noise
    max_or_atr: float = field(
        default_factory=lambda: _env("MAX_OR_ATR", 0.80, float)
    )  # OR too big -> exhausted
    doji_frac: float = field(
        default_factory=lambda: _env("DOJI_FRAC", 0.10, float)
    )  # |c-o| < 10% of range
    allow_short: bool = field(default_factory=lambda: _env("ALLOW_SHORT", True, bool))

    # --- entry / exit
    stop_mode: str = field(default_factory=lambda: _env("STOP_MODE", "or"))  # or | atr
    stop_atr_frac: float = field(default_factory=lambda: _env("STOP_ATR_FRAC", 0.10, float))
    max_stop_atr: float = field(default_factory=lambda: _env("MAX_STOP_ATR", 0.50, float))
    target_r: float = field(
        default_factory=lambda: _env("TARGET_R", 0.0, float)
    )  # 0 = hold to close
    max_chase_pct: float = field(default_factory=lambda: _env("MAX_CHASE_PCT", 0.0015, float))
    stop_order_type: str = field(
        default_factory=lambda: _env("STOP_ORDER_TYPE", "SL-M")
    )  # SL-M | SL
    stop_limit_buf_pct: float = field(
        default_factory=lambda: _env("STOP_LIMIT_BUF_PCT", 0.005, float)
    )
    slippage_pct: float = field(
        default_factory=lambda: _env("SLIPPAGE_PCT", 0.0005, float)
    )  # backtest, per fill

    # --- session (IST)
    market_open: dtime = field(default_factory=lambda: _env_time("MARKET_OPEN", "09:15"))
    or_minutes: int = field(default_factory=lambda: _env("OR_MINUTES", 5, int))
    entry_cutoff: dtime = field(default_factory=lambda: _env_time("ENTRY_CUTOFF", "14:30"))
    squareoff_time: dtime = field(default_factory=lambda: _env_time("SQUAREOFF_TIME", "15:10"))
    market_close: dtime = field(default_factory=lambda: _env_time("MARKET_CLOSE", "15:30"))

    # --- charges (Indian intraday equity; verify with your broker)
    brokerage_flat: float = field(default_factory=lambda: _env("BROKERAGE_FLAT", 20.0, float))
    brokerage_pct: float = field(default_factory=lambda: _env("BROKERAGE_PCT", 0.001, float))
    stt_sell_pct: float = field(default_factory=lambda: _env("STT_SELL_PCT", 0.00025, float))
    exch_txn_pct: float = field(default_factory=lambda: _env("EXCH_TXN_PCT", 0.0000297, float))
    sebi_pct: float = field(default_factory=lambda: _env("SEBI_PCT", 0.000001, float))
    stamp_pct: float = field(default_factory=lambda: _env("STAMP_PCT", 0.00003, float))
    gst: float = field(default_factory=lambda: _env("GST", 0.18, float))

    # --- live loop
    poll_sec: float = field(default_factory=lambda: _env("POLL_SEC", 1.0, float))
    status_every_sec: float = field(default_factory=lambda: _env("STATUS_EVERY_SEC", 3.0, float))
    entry_timeout_sec: int = field(default_factory=lambda: _env("ENTRY_TIMEOUT_SEC", 20, int))
    fill_wait_sec: float = field(default_factory=lambda: _env("FILL_WAIT_SEC", 5.0, float))
    scan_delay_sec: int = field(
        default_factory=lambda: _env("SCAN_DELAY_SEC", 8, int)
    )  # after OR bar closes
    scan_give_up_min: int = field(default_factory=lambda: _env("SCAN_GIVE_UP_MIN", 5, int))
    state_file: str = field(
        default_factory=lambda: bot_path(_env("STATE_FILE", "sip_orb_state.json"))
    )
    trade_log: str = field(
        default_factory=lambda: bot_path(_env("TRADE_LOG", "sip_orb_trades.csv"))
    )

    @property
    def paper(self):
        """True only for the in-script simulation; sandbox and live both send orders to OpenAlgo."""
        return self.trading_mode == "paper"

    @paper.setter
    def paper(self, value):
        self.trading_mode = "paper" if value else "live"

    def symbols(self):
        if self.universe_file and os.path.exists(self.universe_file):
            with open(self.universe_file) as fh:
                syms = [ln.strip().upper() for ln in fh if ln.strip() and not ln.startswith("#")]
        else:
            syms = [s.strip().upper() for s in self.universe.split(",") if s.strip()]
        return list(dict.fromkeys(syms))

    @property
    def or_end(self):
        t = datetime.combine(ddate(2000, 1, 1), self.market_open) + timedelta(
            minutes=self.or_minutes
        )
        return t.time()


# --------------------------------------------------------------------------- logging
log = logging.getLogger("sip_orb")
if not log.handlers:
    log.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(_fmt)
    log.addHandler(_h)
    if "--selftest" not in sys.argv and _env("MODE", "").lower() != "selftest":
        _fh = logging.FileHandler(bot_path("sip_orb_nse.log"), encoding="utf-8")
        _fh.setFormatter(_fmt)
        log.addHandler(_fh)
    log.propagate = False
for _noisy in ("httpx", "httpcore", "urllib3", "requests", "websocket"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- helpers
def round_tick(px, tick, mode="nearest"):
    q = px / tick
    q = (
        math.floor(q + 1e-9)
        if mode == "down"
        else math.ceil(q - 1e-9)
        if mode == "up"
        else round(q)
    )
    return round(q * tick, 2)


def charges_inr(qty, buy_price, sell_price, cfg: Config) -> float:
    """Round trip: brokerage both legs + STT (sell) + exch txn + SEBI + stamp (buy) + GST."""
    buy_val, sell_val = qty * buy_price, qty * sell_price
    brok = min(cfg.brokerage_flat, cfg.brokerage_pct * buy_val) + min(
        cfg.brokerage_flat, cfg.brokerage_pct * sell_val
    )
    stt = cfg.stt_sell_pct * sell_val
    exch = cfg.exch_txn_pct * (buy_val + sell_val)
    sebi = cfg.sebi_pct * (buy_val + sell_val)
    stamp = cfg.stamp_pct * buy_val
    return brok + stt + exch + sebi + stamp + cfg.gst * (brok + exch + sebi)


def pnl_inr(side, entry, exit_, qty, cfg: Config):
    gross = (exit_ - entry) * qty * (1 if side == "LONG" else -1)
    buy, sell = (entry, exit_) if side == "LONG" else (exit_, entry)
    fees = charges_inr(qty, buy, sell, cfg)
    return gross - fees, gross, fees


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Any OpenAlgo/CSV OHLCV frame -> float OHLCV indexed by IST timestamps."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        tcol = next(
            (c for c in ("timestamp", "datetime", "date", "time") if c in df.columns), df.columns[0]
        )
        s = df[tcol]
        if pd.api.types.is_numeric_dtype(s):
            idx = pd.to_datetime(s, unit="ms" if s.max() > 1e12 else "s", utc=True)
        else:
            idx = pd.to_datetime(s)
        df.index = pd.DatetimeIndex(idx)
        df = df.drop(columns=[tcol])
    df.index = (
        df.index.tz_localize("Asia/Kolkata")
        if df.index.tz is None
        else df.index.tz_convert("Asia/Kolkata")
    )
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


def to_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample (e.g. 1m) bars to 5m session-aligned bars; 5m input passes through unchanged."""
    if len(df) > 2:
        step = pd.Series(df.index[1:] - df.index[:-1]).median()
        if step >= pd.Timedelta(minutes=5):
            return df
    out = df.resample("5min", label="left", closed="left", origin="start_day", offset="15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open"])


# --------------------------------------------------------------------------- daily statistics
def daily_table(df5: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per session: OR candle, day OHLC, and baselines built ONLY from prior sessions
    (atr = ATR of previous ATR_DAYS days, or_vol_avg = mean OR volume of previous RVOL_DAYS days)."""
    t = df5.index.time
    s = df5[(t >= cfg.market_open) & (t < cfg.market_close)]
    if s.empty:
        return pd.DataFrame()
    day = pd.Index(s.index.date, name="day")
    d = s.groupby(day).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    orb = s[s.index.time < cfg.or_end]
    oday = pd.Index(orb.index.date, name="day")
    o = orb.groupby(oday).agg(
        or_open=("open", "first"),
        or_high=("high", "max"),
        or_low=("low", "min"),
        or_close=("close", "last"),
        or_vol=("volume", "sum"),
        or_bars=("open", "size"),
    )
    d = d.join(o, how="left")
    pc = d["close"].shift()
    tr = pd.concat(
        [d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1
    ).max(axis=1)
    d["atr"] = tr.rolling(cfg.atr_days, min_periods=cfg.atr_days).mean().shift(1)
    d["or_vol_avg"] = d["or_vol"].rolling(cfg.rvol_days, min_periods=cfg.rvol_days).mean().shift(1)
    d["turnover_cr"] = (d["close"] * d["volume"]).rolling(
        cfg.rvol_days, min_periods=cfg.rvol_days
    ).mean().shift(1) / 1e7
    d["prev_close"] = pc
    return d


def rank_candidates(rows: dict, cfg: Config, apply_rvol=True):
    """rows: {symbol: dict(or_open, or_high, or_low, or_close, or_vol, atr, or_vol_avg, turnover_cr)}.
    Returns (ranked list of candidate dicts, {symbol: reason skipped})."""
    cands, skipped = [], {}
    for sym, r in rows.items():
        need = (
            "or_open",
            "or_high",
            "or_low",
            "or_close",
            "or_vol",
            "atr",
            "or_vol_avg",
            "turnover_cr",
        )
        if any(
            r.get(k) is None or (isinstance(r.get(k), float) and np.isnan(r.get(k))) for k in need
        ):
            skipped[sym] = "not enough history"
            continue
        if r["or_vol_avg"] <= 0 or r["atr"] <= 0:
            skipped[sym] = "zero baseline"
            continue
        rvol = r["or_vol"] / r["or_vol_avg"]
        rng = r["or_high"] - r["or_low"]
        body = r["or_close"] - r["or_open"]
        why = None
        if r["or_close"] < cfg.min_price:
            why = f"price {r['or_close']:.0f} < {cfg.min_price:.0f}"
        elif r["turnover_cr"] < cfg.min_turnover_cr:
            why = f"turnover {r['turnover_cr']:.0f}cr < {cfg.min_turnover_cr:.0f}cr"
        elif apply_rvol and rvol < cfg.min_rvol:
            why = f"RVOL {rvol:.2f} < {cfg.min_rvol:g}"
        elif rng <= 0 or abs(body) < cfg.doji_frac * rng:
            why = "doji opening candle"
        elif rng / r["atr"] < cfg.min_or_atr:
            why = f"OR {rng / r['atr']:.2f} ATR < {cfg.min_or_atr:g}"
        elif rng / r["atr"] > cfg.max_or_atr:
            why = f"OR {rng / r['atr']:.2f} ATR > {cfg.max_or_atr:g} (move already done)"
        elif body < 0 and not cfg.allow_short:
            why = "red OR, shorts disabled"
        if why:
            skipped[sym] = why
            continue
        cands.append(
            {
                "symbol": sym,
                "side": "LONG" if body > 0 else "SHORT",
                "rvol": rvol,
                "or_high": r["or_high"],
                "or_low": r["or_low"],
                "or_close": r["or_close"],
                "atr": r["atr"],
                "or_atr": rng / r["atr"],
            }
        )
    cands.sort(key=lambda c: -c["rvol"])
    for c in cands[cfg.top_n :]:
        skipped[c["symbol"]] = f"RVOL {c['rvol']:.2f} not in top {cfg.top_n}"
    return cands[: cfg.top_n], skipped


def make_plan(c: dict, cfg: Config, n_slots: int):
    """Trigger, stop, optional target and share qty for one candidate. None if it cannot be sized."""
    tick = cfg.tick_size
    long = c["side"] == "LONG"
    trigger = (
        round_tick(c["or_high"] + tick, tick, "up")
        if long
        else round_tick(c["or_low"] - tick, tick, "down")
    )
    cap = cfg.max_stop_atr * c["atr"]
    if cfg.stop_mode == "atr":
        dist = cfg.stop_atr_frac * c["atr"]
    else:
        dist = (trigger - (c["or_low"] - tick)) if long else ((c["or_high"] + tick) - trigger)
    dist = min(dist, cap)
    stop = (
        round_tick(trigger - dist, tick, "down") if long else round_tick(trigger + dist, tick, "up")
    )
    stop_dist = abs(trigger - stop)
    if stop_dist < tick:
        return None, "stop distance below one tick"
    risk_inr = cfg.capital_inr * cfg.risk_pct
    per_share_cost = (
        charges_inr(1000, trigger, trigger, cfg) / 1000
    )  # % costs, flat brokerage amortised
    qty = int(risk_inr // (stop_dist + per_share_cost + trigger * cfg.slippage_pct))
    max_notional = cfg.capital_inr * cfg.max_leverage / max(n_slots, 1)
    qty = min(qty, int(max_notional // trigger))
    if not cfg.paper and cfg.live_max_qty > 0:
        qty = min(qty, cfg.live_max_qty)
    if qty < 1:
        return (
            None,
            f"qty 0 (risk Rs {risk_inr:.0f}, stop Rs {stop_dist:.2f}/sh, notional cap Rs {max_notional:.0f})",
        )
    target = None
    if cfg.target_r > 0:
        target = (
            round_tick(trigger + cfg.target_r * stop_dist, tick, "down")
            if long
            else round_tick(trigger - cfg.target_r * stop_dist, tick, "up")
        )
    worst, _, _ = pnl_inr(c["side"], trigger, stop, qty, cfg)
    return {
        **c,
        "trigger": trigger,
        "stop": stop,
        "target": target,
        "qty": qty,
        "risk_inr": stop_dist * qty,
        "worst_inr": worst,
    }, "ok"


# --------------------------------------------------------------------------- backtest
def simulate_trade(plan, bars: pd.DataFrame, cfg: Config):
    """Walk 5m bars after the OR. Conservative: stop wins ties, gaps fill at the open, slippage on stop fills."""
    long = plan["side"] == "LONG"
    trig, stop, tgt = plan["trigger"], plan["stop"], plan["target"]
    sl = cfg.slippage_pct
    entry = entry_ts = None
    for ts, b in bars.iterrows():
        tt = ts.time()
        if entry is None:
            if tt >= cfg.squareoff_time or tt > cfg.entry_cutoff:
                return None
            hit = b["high"] >= trig if long else b["low"] <= trig
            if not hit:
                continue
            base = max(trig, b["open"]) if long else min(trig, b["open"])
            if abs(base - trig) / trig > cfg.max_chase_pct:  # gapped past the chase limit -> skip
                return None
            entry = base * (1 + sl) if long else base * (1 - sl)
            entry_ts = ts
            if (
                (b["low"] <= stop) if long else (b["high"] >= stop)
            ):  # same-bar stop: assume the worst
                return entry_ts, ts, entry, stop * (1 - sl) if long else stop * (1 + sl), "STOP"
            continue
        if tt >= cfg.squareoff_time:
            px = b["open"] * (1 - sl) if long else b["open"] * (1 + sl)
            return entry_ts, ts, entry, px, "EOD"
        if (b["open"] <= stop) if long else (b["open"] >= stop):
            px = b["open"] * (1 - sl) if long else b["open"] * (1 + sl)
            return entry_ts, ts, entry, px, "STOP"
        if (b["low"] <= stop) if long else (b["high"] >= stop):
            return entry_ts, ts, entry, stop * (1 - sl) if long else stop * (1 + sl), "STOP"
        if tgt is not None and ((b["high"] >= tgt) if long else (b["low"] <= tgt)):
            return entry_ts, ts, entry, tgt, "TARGET"
    if entry is None:
        return None
    last_ts, last = bars.index[-1], bars.iloc[-1]
    return entry_ts, last_ts, entry, last["close"], "EOD"


def backtest(data: dict, cfg: Config, apply_rvol=True, out_csv="sip_orb_backtest.csv", quiet=False):
    """data: {symbol: 5m OHLCV DataFrame (IST index)}. Returns the trades DataFrame."""
    tables, bars_by_day = {}, {}
    for sym, df in data.items():
        df5 = to_5m(normalize(df))
        tables[sym] = daily_table(df5, cfg)
        t = df5.index.time
        s = df5[(t >= cfg.or_end) & (t < cfg.market_close)]
        bars_by_day[sym] = dict(list(s.groupby(s.index.date)))
    all_days = (
        sorted(set().union(*[set(t.index) for t in tables.values() if len(t)])) if tables else []
    )
    trades = []
    for d in all_days:
        rows = {}
        for sym, tab in tables.items():
            if d in tab.index and tab.loc[d, "or_bars"] == cfg.or_minutes // 5:
                rows[sym] = tab.loc[d].to_dict()
        if not rows:
            continue
        cands, _ = rank_candidates(rows, cfg, apply_rvol=apply_rvol)
        realised_r = 0.0
        for c in cands:
            plan, _ = make_plan(c, cfg, len(cands))
            if not plan or d not in bars_by_day[c["symbol"]]:
                continue
            res = simulate_trade(plan, bars_by_day[c["symbol"]][d], cfg)
            if not res:
                continue
            ets, xts, entry, exit_, why = res
            net, gross, fees = pnl_inr(plan["side"], entry, exit_, plan["qty"], cfg)
            r_mult = net / (cfg.capital_inr * cfg.risk_pct)
            trades.append(
                {
                    "day": d,
                    "symbol": c["symbol"],
                    "side": plan["side"],
                    "rvol": round(c["rvol"], 2),
                    "or_atr": round(c["or_atr"], 2),
                    "qty": plan["qty"],
                    "entry_time": ets,
                    "exit_time": xts,
                    "entry": round(entry, 2),
                    "stop": plan["stop"],
                    "exit": round(exit_, 2),
                    "reason": why,
                    "gross_inr": round(gross, 2),
                    "fees_inr": round(fees, 2),
                    "net_inr": round(net, 2),
                    "r": round(r_mult, 3),
                }
            )
            realised_r += r_mult
    t = pd.DataFrame(trades)
    if not quiet:
        report(
            t,
            cfg,
            len(all_days),
            "RVOL-filtered (stocks in play)" if apply_rvol else "NO RVOL filter",
        )
        if len(t):
            out_csv = bot_path(out_csv)
            t.to_csv(out_csv, index=False)
            print(f"Trade list saved  : {out_csv}")
    return t


def report(t: pd.DataFrame, cfg: Config, n_days: int, title: str):
    print("\n" + "=" * 72)
    print(
        f"BACKTEST [{title}]  {n_days} sessions  | capital Rs {cfg.capital_inr:,.0f}  risk/trade "
        f"Rs {cfg.capital_inr * cfg.risk_pct:,.0f} (=1R)  top {cfg.top_n}  stop={cfg.stop_mode}"
    )
    if t.empty:
        print("No trades.")
        print("=" * 72)
        return
    wins = t[t.net_inr > 0]
    eq = t.groupby("day").net_inr.sum().cumsum()
    gp, gl = wins.net_inr.sum(), -t[t.net_inr <= 0].net_inr.sum()
    print(
        f"Trades            : {len(t)}  ({len(t) / max(n_days, 1):.2f}/day, {t.day.nunique()} days traded)"
    )
    print(
        f"Win rate          : {len(wins) / len(t):.1%}   (low is normal - edge is in the size of winners)"
    )
    print(
        f"Avg R / trade     : {t.r.mean():+.3f} R   (median {t.r.median():+.2f} R, best {t.r.max():+.1f} R)"
    )
    print(f"Profit factor     : {gp / gl if gl > 0 else float('inf'):.2f}")
    print(f"Exit reasons      : {t.reason.value_counts().to_dict()}")
    print(
        f"Charges paid      : Rs {t.fees_inr.sum():,.0f}  ({t.fees_inr.sum() / max(t.gross_inr.abs().sum(), 1):.0%} of gross moves)"
    )
    print(
        f"NET P&L           : Rs {t.net_inr.sum():,.0f}  ({t.net_inr.sum() / cfg.capital_inr:+.1%} of capital)"
    )
    print(f"Max drawdown      : Rs {(eq - eq.cummax()).min():,.0f}")
    print(
        f"Long / Short avg R: {t[t.side == 'LONG'].r.mean():+.3f} / {t[t.side == 'SHORT'].r.mean():+.3f}"
    )
    bins = [0, 1, 2, 3, 5, 10, np.inf]
    lab = ["<1x", "1-2x", "2-3x", "3-5x", "5-10x", ">10x"]
    b = t.assign(bucket=pd.cut(t.rvol, bins=bins, labels=lab, right=False)).groupby(
        "bucket", observed=True
    )
    print("Edge by opening RVOL (the key test - avg R should RISE with RVOL):")
    for k, g in b:
        print(
            f"   {k:>6}: {len(g):4d} trades  avg {g.r.mean():+.3f} R  win {(g.net_inr > 0).mean():.0%}  "
            f"net Rs {g.net_inr.sum():,.0f}"
        )
    m = t.assign(month=pd.to_datetime(t.day).dt.strftime("%Y-%m")).groupby("month").net_inr.sum()
    print("Monthly net       : " + ", ".join(f"{k} {v:+,.0f}" for k, v in m.items()))
    print("=" * 72)


# --------------------------------------------------------------------------- broker layer
_STATUS = {
    "complete": "complete",
    "completed": "complete",
    "filled": "complete",
    "traded": "complete",
    "open": "open",
    "pending": "open",
    "trigger pending": "open",
    "put order req received": "open",
    "validation pending": "open",
    "open pending": "open",
    "modified": "open",
    "after market order req received": "open",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "rejected": "rejected",
}


class LiveBroker:
    """Orders through OpenAlgo (sandbox or live - OpenAlgo's Analyze switch decides which pipe).

    Prices come from the WebSocket LTP feed when it is fresh, else from a REST quote. The feed
    callback runs on the SDK's socket thread and only writes one cache entry under a lock; all
    trading decisions stay on the main loop thread.
    """

    def __init__(self, cfg: Config, client=None):
        self.cfg = cfg
        if client is None:
            from openalgo import api

            client = api(api_key=cfg.api_key, host=cfg.host, ws_url=cfg.ws_url)
        self.c = client
        self._ticks = {}
        self._tick_lock = threading.Lock()
        self._subscribed = set()
        self._ws_up = False

    def _pause(self):
        time.sleep(self.cfg.api_sleep)

    # ---- live price feed
    def start_feed(self, symbols):
        """Subscribe symbols to the WebSocket LTP feed. Safe to call repeatedly; failures fall back to REST."""
        if self.cfg.feed != "ws":
            return
        new = [s for s in symbols if s not in self._subscribed]
        if not new:
            return
        try:
            if not self._ws_up:
                self.c.connect()
                self._ws_up = True
            ok = self.c.subscribe_ltp(
                [{"exchange": self.cfg.exchange, "symbol": s} for s in new],
                on_data_received=self._on_tick,
            )
        except Exception as e:
            log.warning(f"live price feed unavailable ({e}) - using REST quotes")
            return
        if ok:
            self._subscribed.update(new)
            log.info(f"live price feed on: {', '.join(new)}")
        else:
            log.warning("live price feed subscribe failed - using REST quotes")

    def _on_tick(self, msg):
        try:
            sym = msg.get("symbol")
            ltp = float(msg["data"]["ltp"])
        except (AttributeError, KeyError, TypeError, ValueError):
            return
        if sym and ltp > 0:
            with self._tick_lock:
                self._ticks[sym] = (ltp, time.time())

    def stop_feed(self):
        if self._ws_up:
            try:
                self.c.disconnect()
            except Exception as e:
                log.warning(f"feed disconnect: {e}")
            self._ws_up = False

    def quote(self, sym, book=False):
        """LTP from the live feed when fresh; a REST quote (with bid/ask) otherwise or when book=True."""
        if not book:
            with self._tick_lock:
                tick = self._ticks.get(sym)
            if tick and time.time() - tick[1] <= self.cfg.stale_sec:
                return {"ltp": tick[0], "bid": tick[0], "ask": tick[0]}
        self._pause()
        try:
            q = self.c.quotes(symbol=sym, exchange=self.cfg.exchange)
            d = q.get("data", {}) if isinstance(q, dict) else {}
            ltp = float(d.get("ltp") or 0)
            return {
                "ltp": ltp,
                "bid": float(d.get("bid") or ltp),
                "ask": float(d.get("ask") or ltp),
            }
        except Exception as e:
            log.warning(f"quote {sym} failed: {e}")
            return {"ltp": 0.0, "bid": 0.0, "ask": 0.0}

    def history(self, sym, interval, start, end):
        self._pause()
        try:
            return self.c.history(
                symbol=sym,
                exchange=self.cfg.exchange,
                interval=interval,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            return f"error: {e}"

    def place(self, sym, action, qty, price_type, price=0.0, trigger=0.0):
        self._pause()
        try:
            r = self.c.placeorder(
                strategy=self.cfg.strategy,
                symbol=sym,
                action=action,
                exchange=self.cfg.exchange,
                price_type=price_type,
                product=self.cfg.product,
                quantity=int(qty),
                price=price,
                trigger_price=trigger,
            )
        except Exception as e:
            log.error(f"placeorder {sym} failed: {e}")
            return None
        if not isinstance(r, dict) or r.get("status") != "success":
            log.error(f"order rejected by API ({sym} {action} {qty} {price_type}): {r}")
            return None
        return str(r.get("orderid"))

    def status(self, oid, sym=None, ltp=None):
        self._pause()
        try:
            r = self.c.orderstatus(order_id=oid, strategy=self.cfg.strategy)
            d = r.get("data", {}) if isinstance(r, dict) else {}
            st = _STATUS.get(str(d.get("order_status", "")).strip().lower(), "unknown")
            px = float(d.get("average_price") or 0) or float(d.get("price") or 0)
            return st, px
        except Exception as e:
            log.warning(f"orderstatus {oid} failed: {e}")
            return "unknown", 0.0

    def cancel(self, oid):
        self._pause()
        try:
            return self.c.cancelorder(order_id=oid, strategy=self.cfg.strategy)
        except Exception as e:
            log.warning(f"cancel {oid} failed: {e}")

    def position_qty(self, sym):
        """Signed open qty, or None if unreadable (None is NOT flat)."""
        self._pause()
        try:
            r = self.c.openposition(
                strategy=self.cfg.strategy,
                symbol=sym,
                exchange=self.cfg.exchange,
                product=self.cfg.product,
            )
            if not isinstance(r, dict) or r.get("status") != "success":
                return None
            return float(r.get("quantity", 0) or 0)
        except Exception:
            return None


class PaperBroker(LiveBroker):
    """Real quotes/history, simulated fills. LIMIT fills when LTP reaches the price, stop orders when LTP
    touches the trigger (filled at the worse of trigger/LTP), MARKET at the far side of the book."""

    def __init__(self, cfg: Config, client=None):
        super().__init__(cfg, client)
        self.orders, self.n, self.pos = {}, 0, {}

    def place(self, sym, action, qty, price_type, price=0.0, trigger=0.0):
        self.n += 1
        oid = f"PAPER{self.n}"
        o = {
            "sym": sym,
            "action": action,
            "qty": qty,
            "type": price_type,
            "price": price,
            "trigger": trigger,
            "status": "open",
            "avg": 0.0,
        }
        self.orders[oid] = o
        if price_type == "MARKET":
            q = self.quote(sym, book=True)
            self._fill(o, (q["ask"] if action == "BUY" else q["bid"]) or q["ltp"])
        elif price_type == "LIMIT":  # marketable limit fills now at the touch
            q = self.quote(sym, book=True)
            if action == "BUY" and 0 < q["ask"] <= price:
                self._fill(o, q["ask"])
            elif action == "SELL" and q["bid"] >= price > 0:
                self._fill(o, q["bid"])
        log.info(f"[PAPER] {sym} {action} {qty} {price_type} px={price} trg={trigger} -> {oid}")
        return oid

    def _fill(self, o, px):
        o.update(status="complete", avg=px)
        self.pos[o["sym"]] = self.pos.get(o["sym"], 0) + (
            o["qty"] if o["action"] == "BUY" else -o["qty"]
        )

    def status(self, oid, sym=None, ltp=None):
        o = self.orders.get(oid)
        if not o:
            return "unknown", 0.0
        if o["status"] == "open" and ltp:
            buy = o["action"] == "BUY"
            if o["type"] == "LIMIT" and (
                (buy and ltp <= o["price"]) or (not buy and ltp >= o["price"])
            ):
                self._fill(o, o["price"])
            elif o["type"] in ("SL", "SL-M") and (
                (buy and ltp >= o["trigger"]) or (not buy and ltp <= o["trigger"])
            ):
                self._fill(o, max(ltp, o["trigger"]) if buy else min(ltp, o["trigger"]))
        return o["status"], o["avg"]

    def cancel(self, oid):
        if oid in self.orders and self.orders[oid]["status"] == "open":
            self.orders[oid]["status"] = "cancelled"

    def position_qty(self, sym):
        return float(self.pos.get(sym, 0))


# --------------------------------------------------------------------------- live bot
class Bot:
    """Per-symbol state: WATCH -> ENTRY_SENT -> IN_POS -> DONE (or SKIPPED)."""

    def __init__(self, cfg: Config, broker=None):
        self.cfg = cfg
        self.b = broker or (PaperBroker(cfg) if cfg.paper else LiveBroker(cfg))
        self.day = None
        self.baselines = {}  # sym -> {atr, or_vol_avg, turnover_cr}
        self.book = {}  # sym -> plan + state fields
        self.scanned = False
        self.scan_started = None
        self.day_pnl = 0.0
        self.halted = False
        self._last = {}
        self._baseline_try = 0.0

    # ---- utils
    def _say(self, key, msg, every, level=logging.INFO):
        t = now_ist().timestamp()
        if t - self._last.get(key, 0) >= every:
            self._last[key] = t
            log.log(level, msg)

    def _risk_unit(self):
        return self.cfg.capital_inr * self.cfg.risk_pct

    def save(self):
        st = {
            "day": str(self.day),
            "day_pnl": self.day_pnl,
            "halted": self.halted,
            "scanned": self.scanned,
            "book": self.book,
        }
        tmp = self.cfg.state_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(st, fh, default=str, indent=1)
        os.replace(tmp, self.cfg.state_file)

    def load(self):
        if not os.path.exists(self.cfg.state_file):
            return
        try:
            with open(self.cfg.state_file) as fh:
                st = json.load(fh)
        except Exception as e:
            log.warning(f"could not read {self.cfg.state_file}: {e}")
            return
        if st.get("day") == str(now_ist().date()):
            self.day = now_ist().date()
            self.day_pnl, self.halted, self.scanned = st["day_pnl"], st["halted"], st["scanned"]
            self.book = st["book"]
            live = {
                s: p["state"]
                for s, p in self.book.items()
                if p["state"] in ("WATCH", "ENTRY_SENT", "IN_POS")
            }
            log.warning(
                f"RESUMED today's state from {self.cfg.state_file}: day P&L Rs {self.day_pnl:.0f}, active {live}"
            )

    def _active_symbols(self):
        return [s for s, p in self.book.items() if p["state"] in ("WATCH", "ENTRY_SENT", "IN_POS")]

    # ---- day roll & baselines
    def roll_day(self):
        d = now_ist().date()
        if d == self.day:
            return
        if self.day is not None:
            log.info(f"Day {self.day} closed: net Rs {self.day_pnl:.2f}")
        self.day, self.book, self.scanned, self.scan_started = d, {}, False, None
        self.day_pnl, self.halted, self.baselines = 0.0, False, {}
        self._baseline_try = 0.0

    def build_baselines(self):
        """Previous-sessions ATR, average OR volume and turnover for every symbol (excludes today)."""
        c = self.cfg
        now = now_ist()
        today = now.date()
        start = now - timedelta(days=int((max(c.rvol_days, c.atr_days) + 2) * 1.6) + 7)
        out, bad = {}, []
        for sym in c.symbols():
            df = self.b.history(sym, "5m", start, now)
            if not isinstance(df, pd.DataFrame) or df.empty:
                bad.append(sym)
                continue
            df5 = to_5m(normalize(df))
            df5 = df5[df5.index.date < today]
            tab = daily_table(df5, c)
            if tab.empty:
                bad.append(sym)
                continue
            # baselines for "the next session" = rolling values including the last completed day
            last = tab.iloc[-1]
            tab2 = tab.copy()
            pc = tab2["close"].shift()
            tr = pd.concat(
                [tab2["high"] - tab2["low"], (tab2["high"] - pc).abs(), (tab2["low"] - pc).abs()],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(c.atr_days, min_periods=c.atr_days).mean().iloc[-1]
            ov = tab2["or_vol"].rolling(c.rvol_days, min_periods=c.rvol_days).mean().iloc[-1]
            to = (tab2["close"] * tab2["volume"]).rolling(
                c.rvol_days, min_periods=c.rvol_days
            ).mean().iloc[-1] / 1e7
            if any(pd.isna(x) for x in (atr, ov, to)):
                bad.append(sym)
                continue
            out[sym] = {
                "atr": float(atr),
                "or_vol_avg": float(ov),
                "turnover_cr": float(to),
                "prev_close": float(last["close"]),
            }
        self.baselines = out
        log.info(
            f"baselines ready for {len(out)}/{len(c.symbols())} symbols"
            + (f"; no/short history: {','.join(bad)}" if bad else "")
        )

    # ---- morning scan
    def fetch_or(self):
        """Today's opening-range candle for each symbol with a baseline. Returns {sym: row} or None if
        the feed hasn't produced the OR bar yet for most symbols."""
        c = self.cfg
        now = now_ist()
        rows, missing = {}, 0
        for sym, base in self.baselines.items():
            df = self.b.history(sym, "5m", now, now)
            if not isinstance(df, pd.DataFrame) or df.empty:
                missing += 1
                continue
            df5 = to_5m(normalize(df))
            day = df5[df5.index.date == now.date()]
            orb = day[(day.index.time >= c.market_open) & (day.index.time < c.or_end)]
            if len(orb) != c.or_minutes // 5:
                missing += 1
                continue
            rows[sym] = {
                "or_open": orb["open"].iloc[0],
                "or_high": orb["high"].max(),
                "or_low": orb["low"].min(),
                "or_close": orb["close"].iloc[-1],
                "or_vol": orb["volume"].sum(),
                **base,
            }
        if missing > 0.5 * max(len(self.baselines), 1):
            return None
        return rows

    def scan(self, place=True):
        rows = self.fetch_or()
        if rows is None:
            return False
        cands, skipped = rank_candidates(rows, self.cfg)
        top_other = sorted(
            ((s, r["or_vol"] / r["or_vol_avg"]) for s, r in rows.items() if r["or_vol_avg"] > 0),
            key=lambda x: -x[1],
        )[:8]
        log.info("top opening RVOL today: " + ", ".join(f"{s} {v:.1f}x" for s, v in top_other))
        if not cands:
            log.info(
                "no stocks in play today (nothing passed the filters) - no trades. That is a valid outcome."
            )
        for cnd in cands:
            plan, why = make_plan(cnd, self.cfg, len(cands))
            if not plan:
                log.info(f"{cnd['symbol']}: in play (RVOL {cnd['rvol']:.1f}x) but skipped - {why}")
                continue
            log.info(
                f"IN PLAY {plan['symbol']:<11} {plan['side']:<5} RVOL {plan['rvol']:5.1f}x  OR "
                f"{plan['or_low']:.2f}-{plan['or_high']:.2f} ({plan['or_atr']:.2f} ATR)  trigger {plan['trigger']:.2f} "
                f"stop {plan['stop']:.2f} qty {plan['qty']}  risk Rs {plan['risk_inr']:.0f}"
                + (f" target {plan['target']:.2f}" if plan["target"] else " exit 15:10")
            )
            if place:
                self.book[plan["symbol"]] = {
                    **plan,
                    "state": "WATCH",
                    "entry_oid": None,
                    "stop_oid": None,
                    "entry_px": None,
                    "sent_at": None,
                    "exit_px": None,
                    "reason": None,
                }
        self.scanned = True
        self.save()
        if place:
            self.b.start_feed(self._active_symbols())
        return True

    # ---- per-symbol handlers
    def _open_risk_ok(self):
        return not self.halted and self.day_pnl > -self.cfg.daily_loss_limit_r * self._risk_unit()

    def on_watch(self, sym, p, q):
        c = self.cfg
        ltp = q["ltp"]
        long = p["side"] == "LONG"
        if now_ist().time() > c.entry_cutoff or not self._open_risk_ok():
            p["state"], p["reason"] = "SKIPPED", "cutoff/loss limit before trigger"
            return
        if not ((ltp >= p["trigger"]) if long else (ltp <= p["trigger"])):
            return
        chase = abs(ltp - p["trigger"]) / p["trigger"]
        if chase > c.max_chase_pct:
            p["state"], p["reason"] = (
                "SKIPPED",
                f"ran {chase:.2%} past trigger before we could enter",
            )
            log.info(f"{sym}: {p['reason']} - not chasing")
            return
        lim_cap = (
            p["trigger"] * (1 + c.max_chase_pct) if long else p["trigger"] * (1 - c.max_chase_pct)
        )
        px = round_tick(lim_cap, c.tick_size, "down" if long else "up")
        oid = self.b.place(sym, "BUY" if long else "SELL", p["qty"], "LIMIT", px)
        if oid:
            p.update(state="ENTRY_SENT", entry_oid=oid, sent_at=time.time(), limit_px=px)
            log.info(
                f"ENTRY {sym} {p['side']} {p['qty']} LIMIT {px:.2f} (ltp {ltp:.2f} crossed {p['trigger']:.2f})"
            )

    def on_entry_sent(self, sym, p, q):
        st, px = self.b.status(p["entry_oid"], sym, q["ltp"])
        if st != "complete" and (
            time.time() - p["sent_at"] > self.cfg.entry_timeout_sec
            or now_ist().time() >= self.cfg.squareoff_time
        ):
            self.b.cancel(p["entry_oid"])
            st, px = self._wait(p["entry_oid"], sym, q["ltp"])
            if st != "complete":
                pos = self.b.position_qty(sym)
                if pos:
                    log.warning(
                        f"{sym}: entry partially filled ({pos:+.0f}) - managing that quantity"
                    )
                    p["qty"] = int(abs(pos))
                    st, px = "complete", p.get("limit_px", p["trigger"])
                elif st in ("cancelled", "rejected") or pos == 0:
                    p["state"], p["reason"] = "SKIPPED", "entry limit not filled (no chase)"
                    log.info(f"{sym}: entry not filled -> cancelled")
                    return
                else:
                    self._say(
                        f"unk{sym}",
                        f"{sym}: entry state unclear ({st}) - rechecking",
                        30,
                        logging.WARNING,
                    )
                    return
        if st == "complete":
            p["entry_px"] = px or p.get("limit_px", p["trigger"])
            p["state"] = "IN_POS"
            self.place_stop(sym, p)
            log.info(
                f"FILLED {sym} {p['side']} {p['qty']} @ {p['entry_px']:.2f} | stop {p['stop']:.2f} "
                f"(order {p['stop_oid']})"
            )
        elif st in ("cancelled", "rejected"):
            p["state"], p["reason"] = "SKIPPED", f"entry {st} by broker"
            log.warning(f"{sym}: entry {st} by broker")

    def place_stop(self, sym, p):
        c = self.cfg
        long = p["side"] == "LONG"
        act = "SELL" if long else "BUY"
        if c.stop_order_type.upper() == "SL":
            lim = (
                p["stop"] * (1 - c.stop_limit_buf_pct)
                if long
                else p["stop"] * (1 + c.stop_limit_buf_pct)
            )
            oid = self.b.place(sym, act, p["qty"], "SL", round_tick(lim, c.tick_size), p["stop"])
        else:
            oid = self.b.place(sym, act, p["qty"], "SL-M", 0, p["stop"])
        p["stop_oid"] = oid
        if not oid:
            log.error(f"{sym}: broker STOP order failed - software stop only! Check the terminal.")

    def on_position(self, sym, p, q):
        c = self.cfg
        ltp = q["ltp"]
        long = p["side"] == "LONG"
        if now_ist().time() >= c.squareoff_time:
            return self.exit(sym, p, q, "EOD")
        touched = (ltp <= p["stop"]) if long else (ltp >= p["stop"])
        # The broker stop is polled every STATUS_EVERY_SEC, or at once when price reaches it, so a fast
        # loop on the live feed does not turn into a flood of order-status calls.
        if p["stop_oid"] and (
            touched or time.time() - p.get("stop_checked", 0) >= c.status_every_sec
        ):
            p["stop_checked"] = time.time()
            st, px = self.b.status(p["stop_oid"], sym, ltp)
            if st == "complete":
                return self.book_exit(sym, p, px or p["stop"], "STOP")
            if st in ("cancelled", "rejected"):
                log.error(f"{sym}: stop order {st} by broker - re-placing")
                self.place_stop(sym, p)
        # backup software stop: price well beyond the stop and the broker stop didn't fill
        beyond = (ltp <= p["stop"] * (1 - 0.002)) if long else (ltp >= p["stop"] * (1 + 0.002))
        if beyond or (not p["stop_oid"] and touched):
            return self.exit(sym, p, q, "STOP")
        if p["target"] and ((ltp >= p["target"]) if long else (ltp <= p["target"])):
            return self.exit(sym, p, q, "TARGET")
        r_now = (
            ((ltp - p["entry_px"]) if long else (p["entry_px"] - ltp))
            * p["qty"]
            / self._risk_unit()
        )
        self._say(
            f"pos{sym}",
            f"holding {sym} {p['side']} {p['qty']} @ {p['entry_px']:.2f} | ltp {ltp:.2f} "
            f"| {r_now:+.2f} R | stop {p['stop']:.2f}",
            120,
        )

    def _wait(self, oid, sym, ltp):
        deadline = time.time() + self.cfg.fill_wait_sec
        st, px = self.b.status(oid, sym, ltp)
        while st not in ("complete", "cancelled", "rejected") and time.time() < deadline:
            time.sleep(0.5)
            st, px = self.b.status(oid, sym, ltp)
        return st, px

    def exit(self, sym, p, q, reason):
        """Cancel the broker stop (confirmed), then market-exit what is really open."""
        if p["stop_oid"]:
            self.b.cancel(p["stop_oid"])
            st, px = self._wait(p["stop_oid"], sym, q["ltp"])
            if st == "complete":
                return self.book_exit(sym, p, px or p["stop"], "STOP")
            if st not in ("cancelled", "rejected"):
                self._say(
                    f"sc{sym}",
                    f"{sym}: stop cancel not confirmed ({st}) - retrying",
                    10,
                    logging.WARNING,
                )
                return
            p["stop_oid"] = None
        qty = p["qty"]
        pos = self.b.position_qty(sym)
        if pos is not None:
            if pos == 0:
                log.warning(f"{sym}: broker shows flat already - booking at last price")
                return self.book_exit(sym, p, q["ltp"], reason)
            qty = int(abs(pos))
        oid = self.b.place(sym, "SELL" if p["side"] == "LONG" else "BUY", qty, "MARKET")
        if not oid:
            self._say(
                f"xf{sym}", f"{sym}: {reason} exit order failed - retrying", 10, logging.ERROR
            )
            return
        st, px = self._wait(oid, sym, q["ltp"])
        if st == "rejected":
            self._say(f"xr{sym}", f"{sym}: {reason} exit REJECTED - retrying", 10, logging.ERROR)
            return
        if st != "complete":
            log.warning(f"{sym}: exit {oid} status '{st}' - booking at LTP, verify in the terminal")
        self.book_exit(sym, p, px or q["ltp"], reason)

    def book_exit(self, sym, p, px, reason):
        net, gross, fees = pnl_inr(p["side"], p["entry_px"], px, p["qty"], self.cfg)
        self.day_pnl += net
        p.update(state="DONE", exit_px=px, reason=reason)
        r = net / self._risk_unit()
        log.info(
            f"EXIT {reason} {sym} {p['side']} {p['qty']} @ {px:.2f} (entry {p['entry_px']:.2f}) | net Rs {net:.2f} "
            f"({r:+.2f} R, charges {fees:.2f}) | day Rs {self.day_pnl:.2f}"
        )
        new = not os.path.exists(self.cfg.trade_log)
        with open(self.cfg.trade_log, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(
                    [
                        "time",
                        "mode",
                        "symbol",
                        "side",
                        "rvol",
                        "qty",
                        "entry",
                        "exit",
                        "reason",
                        "gross",
                        "charges",
                        "net",
                        "r",
                    ]
                )
            w.writerow(
                [
                    now_ist().isoformat(timespec="seconds"),
                    self.cfg.trading_mode.upper(),
                    sym,
                    p["side"],
                    round(p["rvol"], 2),
                    p["qty"],
                    p["entry_px"],
                    px,
                    reason,
                    round(gross, 2),
                    round(fees, 2),
                    round(net, 2),
                    round(r, 3),
                ]
            )
        if self.day_pnl <= -self.cfg.daily_loss_limit_r * self._risk_unit() and not self.halted:
            self.halted = True
            log.warning(
                f"daily loss limit ({self.cfg.daily_loss_limit_r:g} R) hit - no new entries today"
            )

    # ---- main tick
    def tick(self):
        c = self.cfg
        self.roll_day()
        now = now_ist()
        if now.weekday() >= 5:
            self._say("wk", "weekend - waiting", 3600)
            return
        if (
            not self.baselines
            and now.time() < c.squareoff_time
            and time.time() - self._baseline_try > 600
        ):
            self._baseline_try = time.time()
            self.build_baselines()
        scan_at = (
            datetime.combine(now.date(), c.or_end, IST) + timedelta(seconds=c.scan_delay_sec)
        ).time()
        if not self.scanned and now.time() >= scan_at:
            if now.time() > c.entry_cutoff:
                self.scanned = True
                log.info("started after entry cutoff - no scan today")
            elif self.scan_started is None or time.time() - self.scan_started > 15:
                self.scan_started = time.time()
                if not self.scan():
                    late = (
                        datetime.combine(now.date(), c.or_end, IST)
                        + timedelta(minutes=c.scan_give_up_min)
                    ).time()
                    if now.time() > late:
                        log.error(
                            "opening-range bars never arrived from the history API - no trades today"
                        )
                        self.scanned = True
                    else:
                        self._say(
                            "orwait", "waiting for today's opening-range bars from the feed...", 30
                        )
        if not self.scanned and not self.book:
            self._say("pre", f"pre-market: scan at {scan_at:%H:%M:%S}", 600)
            return
        active = {
            s: p for s, p in self.book.items() if p["state"] in ("WATCH", "ENTRY_SENT", "IN_POS")
        }
        changed = False
        for sym, p in active.items():
            q = self.b.quote(sym)
            if q["ltp"] <= 0:
                self._say(f"nq{sym}", f"{sym}: no quote", 60, logging.WARNING)
                continue
            before = p["state"]
            if p["state"] == "WATCH":
                if now.time() >= c.squareoff_time:
                    p["state"], p["reason"] = "SKIPPED", "never triggered"
                else:
                    self.on_watch(sym, p, q)
            elif p["state"] == "ENTRY_SENT":
                self.on_entry_sent(sym, p, q)
            elif p["state"] == "IN_POS":
                self.on_position(sym, p, q)
            changed |= p["state"] != before
        if changed:
            self.save()
        if self.scanned and not active and now.time() >= c.squareoff_time:
            self._say(
                "done", f"session done: net Rs {self.day_pnl:.2f} - waiting for next day", 1800
            )

    def mode_check(self):
        """Refuse to start when the requested mode does not match OpenAlgo's Analyze switch."""
        c = self.cfg
        if c.paper:
            log.info(
                "TRADING_MODE=paper: fills are simulated inside this script and will NOT appear in "
                "OpenAlgo. Use TRADING_MODE=sandbox to see orders in OpenAlgo's Analyze mode."
            )
            return True
        try:
            r = self.b.c.analyzerstatus()
        except Exception as e:
            r = {"status": "error", "message": str(e)}
        if not isinstance(r, dict) or r.get("status") != "success":
            log.error(f"could not read OpenAlgo's Analyze mode ({r}) - not starting")
            return False
        sandbox = bool((r.get("data") or {}).get("analyze_mode"))
        if c.trading_mode == "sandbox" and not sandbox:
            log.error(
                "TRADING_MODE=sandbox but OpenAlgo is in LIVE mode - turn Analyze mode ON first. Not starting."
            )
            return False
        if c.trading_mode == "live":
            if sandbox:
                log.error(
                    "TRADING_MODE=live but OpenAlgo is in Analyze mode - orders would be simulated. Not starting."
                )
                return False
            if not c.confirm_live:
                log.error(
                    "TRADING_MODE=live sends real orders - set CONFIRM_LIVE=YES to confirm. Not starting."
                )
                return False
        log.info(
            f"OpenAlgo is in {'Analyze (sandbox)' if sandbox else 'LIVE'} mode - orders go to "
            f"{'the sandbox' if sandbox else 'the broker'}"
        )
        return True

    def startup_check(self):
        c = self.cfg
        syms = c.symbols()
        q = self.b.quote(syms[0], book=True)
        if q["ltp"] <= 0:
            log.error(f"quote for {syms[0]} failed - check HOST_SERVER / API key / broker login")
            return False
        log.info(f"quote OK: {syms[0]} ltp {q['ltp']:.2f}")
        if not c.paper:
            busy = []
            for s in syms:
                pos = self.b.position_qty(s)
                if pos is None:
                    log.error(f"could not read position for {s} - refusing to start")
                    return False
                if pos and not (s in self.book and self.book[s]["state"] == "IN_POS"):
                    busy.append(f"{s}({pos:+.0f})")
            if busy:
                log.error(
                    f"open {c.product} positions not owned by this bot's saved state: {busy} - close them first"
                )
                return False
        return True

    def run(self):
        c = self.cfg
        log.info(
            f"START sip_orb_nse mode={c.trading_mode.upper()} | {len(c.symbols())} symbols | capital "
            f"Rs {c.capital_inr:,.0f} | 1R = Rs {self._risk_unit():,.0f} | top {c.top_n} by RVOL >= {c.min_rvol:g}x "
            f"| stop={c.stop_mode} | exit {c.squareoff_time:%H:%M} | feed={c.feed}"
            + ("" if c.allow_short else " | longs only")
        )
        if not self.mode_check():
            sys.exit("trading mode check failed")
        self.load()
        if not self.startup_check():
            if not c.paper:
                sys.exit("startup check failed")
            log.warning("startup check failed - continuing in PAPER mode")
        if c.trading_mode == "live":
            log.warning(
                "LIVE: real orders. Broker-side stops protect open trades if this process dies, but the "
                "15:10 exit is done by the bot - if it is down, the broker's own MIS auto-square-off applies."
            )
        self.b.start_feed(self._active_symbols())  # resumed positions get live prices at once
        try:
            while True:
                try:
                    self.tick()
                    time.sleep(c.poll_sec)
                except KeyboardInterrupt:
                    active = [
                        s for s, p in self.book.items() if p["state"] in ("ENTRY_SENT", "IN_POS")
                    ]
                    log.info(f"stopped by user. open: {active or 'none'}")
                    if active:
                        log.warning(
                            "positions/orders still open at the broker - manage them in the terminal!"
                        )
                    self.save()
                    break
                except Exception as e:
                    log.exception(f"loop error: {e}")
                    time.sleep(5)
        finally:
            self.b.stop_feed()


# --------------------------------------------------------------------------- data loading for backtest
def load_openalgo(cfg: Config, days: int):
    from openalgo import api

    client = api(api_key=cfg.api_key, host=cfg.host)
    end = now_ist()
    data = {}
    # Historify (db) serves any window in one call and builds 5m from 1m; the broker API needs 30-day chunks.
    chunk = days if cfg.history_source == "db" else 30
    for sym in cfg.symbols():
        parts, cur = [], end - timedelta(days=days)
        while cur < end:
            nxt = min(cur + timedelta(days=chunk), end)
            time.sleep(cfg.api_sleep)
            try:
                df = client.history(
                    symbol=sym,
                    exchange=cfg.exchange,
                    interval="5m",
                    start_date=cur.strftime("%Y-%m-%d"),
                    end_date=nxt.strftime("%Y-%m-%d"),
                    source=cfg.history_source,
                )
            except Exception as e:
                df = f"error {e}"
            if isinstance(df, pd.DataFrame) and not df.empty:
                parts.append(normalize(df))
            cur = nxt + timedelta(days=1)
        if parts:
            data[sym] = pd.concat(parts)
            data[sym] = data[sym][~data[sym].index.duplicated()].sort_index()
            print(
                f"  {sym:<12} {len(data[sym]):6d} bars  {data[sym].index[0]:%Y-%m-%d} -> {data[sym].index[-1]:%Y-%m-%d}"
            )
        else:
            print(
                f"  {sym:<12} NO DATA (check symbol, or download it in Historify for HISTORY_SOURCE=db)"
            )
    return data


def load_csvdir(path):
    data = {}
    for fn in sorted(os.listdir(path)):
        if fn.lower().endswith(".csv"):
            data[os.path.splitext(fn)[0].upper()] = normalize(pd.read_csv(os.path.join(path, fn)))
    return data


# --------------------------------------------------------------------------- self-test (offline)
def selftest():
    """Checks the mechanics on synthetic data. It does NOT prove any edge - only that the code does what the
    rules say (no look-ahead, stops/gaps/costs handled, live state machine completes a day, mode gates hold)."""
    import tempfile

    global BOT_DIR
    BOT_DIR = tempfile.mkdtemp()
    rng = np.random.default_rng(7)
    cfg = Config()
    cfg.paper, cfg.api_sleep, cfg.fill_wait_sec, cfg.feed = True, 0.0, 0.0, "rest"
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # 1) synthetic market: 30 symbols x 60 sessions of 5m bars. On ~5% of days a symbol is "in play":
    #    5-8x opening volume and a drift in the direction of its opening candle. Otherwise random walk.
    days = pd.bdate_range("2026-03-02", periods=60)
    sess = pd.date_range("09:15", "15:25", freq="5min").time
    data, inplay = {}, set()
    for k in range(30):
        sym, px, rows = f"S{k:02d}", 500 + 50 * k, []
        for d in days:
            play = rng.random() < 0.05
            dirn = rng.choice([-1, 1])
            base_v = 20000 * (1 + k % 5)
            for i, t in enumerate(sess):
                mu = (dirn * 0.0012 if play else 0.0) if i > 0 else (dirn * 0.003 if play else 0.0)
                o = px
                c = o * (1 + mu + rng.normal(0, 0.0018))
                h, lo = (
                    max(o, c) * (1 + abs(rng.normal(0, 0.0007))),
                    min(o, c) * (1 - abs(rng.normal(0, 0.0007))),
                )
                v = (
                    base_v
                    * ((6 + rng.random() * 2) if (play and i == 0) else (3 if i == 0 else 1))
                    * (0.7 + 0.6 * rng.random())
                )
                rows.append((datetime.combine(d.date(), t), o, h, lo, c, v))
                px = c
            if play:
                inplay.add((sym, d.date()))
        data[sym] = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    print("\n1) cost model")
    f = charges_inr(100, 1000, 1000, cfg)
    check(f"Rs 1L round trip charges ~Rs 80 (got {f:.2f})", 70 < f < 90)
    n, g, fe = pnl_inr("SHORT", 1000, 990, 100, cfg)
    check("short P&L sign and charges", abs(g - 1000) < 1e-9 and n < g)

    print("2) selection has no look-ahead (baselines use prior days only)")
    tab = daily_table(to_5m(normalize(data["S00"])), cfg)
    d10 = tab.index[20]
    manual = tab["or_vol"].iloc[20 - cfg.rvol_days : 20].mean()
    check(
        "or_vol_avg on day 20 == mean of days 6..19",
        abs(tab.loc[d10, "or_vol_avg"] - manual) < 1e-6,
    )
    check(
        "first RVOL_DAYS days have no baseline",
        tab["or_vol_avg"].iloc[: cfg.rvol_days].isna().all(),
    )

    print("3) backtest mechanics")
    t = backtest(data, cfg, out_csv="sip_selftest.csv")
    check("some trades generated", len(t) > 0)
    if len(t):
        hit = sum((r.symbol, r.day) in inplay for r in t.itertuples()) / len(t)
        check(f"selected trades are mostly the planted in-play days ({hit:.0%})", hit > 0.7)
        stops = t[t.reason == "STOP"]
        check(
            "stop losses stay near -1R (<= -1.6R incl. gaps/slippage)",
            stops.empty or stops.r.min() > -1.6,
        )
        check(
            "exits never after square-off", all(x.time() <= cfg.squareoff_time for x in t.exit_time)
        )
    t0 = backtest(data, cfg, apply_rvol=False, quiet=True)
    if len(t) and len(t0):
        check(
            f"RVOL filter beats unfiltered on planted data ({t.r.mean():+.2f}R vs {t0.r.mean():+.2f}R)",
            t.r.mean() > t0.r.mean(),
        )

    print("4) gap-through-stop is filled at the open, not at the stop")
    bars = pd.DataFrame(
        {
            "open": [101, 101.5, 97.0],
            "high": [101.6, 101.8, 97.5],
            "low": [100.9, 101.2, 96.5],
            "close": [101.4, 101.3, 97.2],
            "volume": [1, 1, 1],
        },
        index=pd.DatetimeIndex(
            [datetime(2026, 3, 2, 9, 20), datetime(2026, 3, 2, 9, 25), datetime(2026, 3, 2, 9, 30)]
        ).tz_localize("Asia/Kolkata"),
    )
    plan = {"side": "LONG", "trigger": 101.05, "stop": 99.0, "target": None}
    res = simulate_trade(plan, bars, cfg)
    check(f"gap fill {res[3]:.2f} < stop 99.00", res and res[4] == "STOP" and res[3] < 97.0)

    print("5) live state machine on a simulated day (fake clock + fake broker)")
    ok_live = live_sim(data, cfg)
    check("live day: scan -> entry -> broker stop -> EOD/stop exit, all flat by 15:10", ok_live)

    print("6) trading-mode gates (sandbox and live never start against the wrong OpenAlgo mode)")

    class Gate:
        def __init__(self, analyze):
            self.analyze = analyze

        def analyzerstatus(self):
            return {"status": "success", "data": {"analyze_mode": self.analyze}}

    def gate(mode, analyze, confirm=False):
        g = Config()
        g.trading_mode, g.confirm_live, g.api_sleep = mode, confirm, 0.0
        return Bot(g, broker=LiveBroker(g, client=Gate(analyze))).mode_check()

    check("sandbox refuses while OpenAlgo is live", not gate("sandbox", False))
    check("sandbox starts with Analyze mode on", gate("sandbox", True))
    check("live refuses while Analyze mode is on", not gate("live", True, confirm=True))
    check("live refuses without CONFIRM_LIVE", not gate("live", False))
    check("live starts when confirmed and Analyze mode is off", gate("live", False, confirm=True))

    print("7) live price feed cache")
    fc = Config()
    fc.api_sleep, fc.stale_sec = 0.0, 5.0

    class NoRest:
        def quotes(self, symbol, exchange):
            return {"status": "success", "data": {"ltp": 1.0, "bid": 1.0, "ask": 1.0}}

    br = LiveBroker(fc, client=NoRest())
    br._on_tick({"symbol": "AAA", "data": {"ltp": 123.45}})
    check("fresh tick is used instead of a REST quote", br.quote("AAA")["ltp"] == 123.45)
    check(
        "book=True always asks REST (bid/ask for paper fills)",
        br.quote("AAA", book=True)["ltp"] == 1.0,
    )
    br._ticks["AAA"] = (123.45, time.time() - 60)
    check("stale tick falls back to REST", br.quote("AAA")["ltp"] == 1.0)
    br._on_tick({"symbol": "AAA", "data": {"ltp": "bad"}})
    check("malformed tick is ignored", br._ticks["AAA"][0] == 123.45)

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return ok


def live_sim(data, cfg):
    """Replay one synthetic session through Bot.tick() with a patched clock."""
    import tempfile

    global now_ist
    real_now = now_ist
    frames = {s: to_5m(normalize(df)) for s, df in data.items()}
    days = sorted(set(frames["S00"].index.date))
    # pick a day with an in-play candidate that triggers
    tmpdir = tempfile.mkdtemp()
    cfg2 = Config()
    cfg2.__dict__.update(cfg.__dict__)
    cfg2.universe = ",".join(frames)
    cfg2.state_file = os.path.join(tmpdir, "state.json")
    cfg2.trade_log = os.path.join(tmpdir, "trades.csv")
    cfg2.poll_sec = 0
    cfg2.status_every_sec = 0
    clock = {"t": None}

    class FakeClient:
        def quotes(self, symbol, exchange):
            df = frames[symbol]
            cur = df[df.index <= clock["t"] - timedelta(minutes=5)]  # last completed bar close
            ltp = float(cur["close"].iloc[-1]) if len(cur) else 0.0
            return {"status": "success", "data": {"ltp": ltp, "bid": ltp - 0.05, "ask": ltp + 0.05}}

        def history(self, symbol, exchange, interval, start_date, end_date):
            df = frames[symbol]
            s = pd.Timestamp(start_date).tz_localize("Asia/Kolkata")
            out = df[
                (df.index >= s) & (df.index <= clock["t"] - timedelta(minutes=5))
            ]  # completed bars only
            return out.reset_index().rename(columns={"index": "timestamp"})

    for d in days[30:]:
        clock["t"] = datetime.combine(d, dtime(9, 0), IST)
        now_ist = lambda: clock["t"]  # noqa: E731
        globals()["now_ist"] = now_ist
        bot = Bot(cfg2, broker=PaperBroker(cfg2, client=FakeClient()))
        while clock["t"].time() < dtime(15, 20):
            bot.tick()
            clock["t"] += timedelta(minutes=1)
        states = {s: p["state"] for s, p in bot.book.items()}
        if any(p["state"] == "DONE" for p in bot.book.values()):
            flat = all(bot.b.position_qty(s) == 0 for s in bot.book)
            done_ok = all(st in ("DONE", "SKIPPED") for st in states.values())
            globals()["now_ist"] = real_now
            print(f"     simulated {d}: {states} day P&L Rs {bot.day_pnl:.0f}")
            return flat and done_ok
    globals()["now_ist"] = real_now
    return False


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Stocks-in-Play Opening Range Breakout (NSE / OpenAlgo)"
    )
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument(
        "--days", type=int, default=None, help="history window for --backtest via OpenAlgo"
    )
    ap.add_argument("--csvdir", help="backtest from a folder of <SYMBOL>.csv files")
    ap.add_argument(
        "--scan", action="store_true", help="print today's stocks in play and exit (no orders)"
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    mode = _env("MODE", "run").lower()
    cfg = Config()

    if args.selftest or mode == "selftest":
        sys.exit(0 if selftest() else 1)
    if args.backtest or mode == "backtest":
        days = args.days or _env("BACKTEST_DAYS", 180, int)
        csvdir = args.csvdir or _env("CSV_DIR", "")
        print(f"Loading {len(cfg.symbols())} symbols...")
        data = load_csvdir(csvdir) if csvdir else load_openalgo(cfg, days)
        if not data:
            sys.exit("no data loaded")
        backtest(data, cfg)
        print(
            "\nBaseline for comparison - same rules WITHOUT the relative-volume filter (random-ish ORB):"
        )
        backtest(data, cfg, apply_rvol=False, out_csv="sip_orb_backtest_norvol.csv")
        print(
            "\nRead it like this: if the RVOL-filtered run is not clearly better than the baseline, and the"
        )
        print(
            "RVOL bucket table doesn't rise with RVOL, the edge is NOT present in this data - don't go live."
        )
        return
    if not cfg.api_key:
        sys.exit("Set OPENALGO_API_KEY first (the /python host injects it automatically).")
    bot = Bot(cfg)
    if args.scan or mode == "scan":
        bot.build_baselines()
        if not bot.scan(place=False):
            print("opening-range bars not available yet (run after 09:20 IST on a trading day)")
        return
    bot.run()


if __name__ == "__main__":
    main()
