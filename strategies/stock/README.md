# Intraday Stock Strategies

`intraday_strategy_lab.py` holds nine popular intraday stock strategies behind
one engine. The same file backtests them, removes the ones that do not beat
costs, and trades a surviving one on paper or live. Upload it once at `/python`
(exchange NSE) and switch between jobs with the `MODE` parameter.

## The strategies

| Name | Style | Entry | Exit |
| --- | --- | --- | --- |
| `orb` | Breakout | Close beyond the 09:15-09:30 range, before 11:00 | Other side of the range (max 2 ATR), target 2R |
| `vwap_pullback` | Trend | Above VWAP and EMA9 > EMA21, dips to VWAP or EMA21, closes above the prior high | Below the 3-bar low, target 2R |
| `vwap_reversion` | Reversion | Stretched 2 deviations from VWAP with a rejection bar | Beyond the bar, target VWAP |
| `rsi2_reversion` | Reversion | RSI(2) below 10 above EMA50 (above 90 below it) | 1.5 ATR stop, 1 ATR target, 6 bars max |
| `bb_reversion` | Reversion | Closes back inside the Bollinger band (20, 2) | Beyond the 2-bar extreme, target middle band |
| `supertrend` | Trend | Supertrend (10, 3) flips direction | Supertrend line, target 3R |
| `ema_cross` | Trend | EMA 9/21 cross on the same side of VWAP | 1.5 ATR stop, target 2R |
| `gap_fill` | Reversion | 0.5-2% opening gap whose first 15 minutes reverse | Beyond the day's extreme, target yesterday's close |
| `pdh_pdl_breakout` | Breakout | Close beyond the previous day's high or low, 09:45-13:30 | 1.5 ATR stop, target 2R |

All of them trade `MIS` on 5-minute bars, stop new entries at 14:30 and square
off at 15:10. Settings are the textbook defaults and are not tuned to the data.

## Step 1: analyze

Download 1-minute history for your stocks in Historify first, then run:

| Parameter | Value |
| --- | --- |
| `MODE` | `analyze` |
| `SYMBOLS` | Comma-separated NSE symbols. Defaults to 10 liquid NIFTY 50 stocks |
| `COST_PCT` | Your real round-trip cost in percent, slippage included. Default `0.10` |

It prints a leaderboard and writes `strategy_leaderboard.csv` and
`strategy_by_symbol.csv` next to the script. A strategy is marked **KEEP** only
if all of these hold, otherwise **REMOVE** with the reasons:

- at least 60 trades in total
- profitable after costs in the older 75% of days **and** the most recent 25%
- profit factor 1.1 or better
- profitable on at least half of the stocks

## Step 2: paper trade

Turn on **Analyze mode** in OpenAlgo, then run with `MODE=paper` and
`STRATEGY=<name>`. The script refuses to start in paper mode while OpenAlgo is
live, so a paper run can never send a real order.

## Step 3: live

Only after a few weeks of paper results that match the backtest. Turn Analyze
mode off and set `MODE=live`, `STRATEGY=<name>`, `CONFIRM_LIVE=YES`. It refuses
to start if the strategy is not marked KEEP in the latest leaderboard.

Live and paper runs also enforce `MAX_OPEN_POSITIONS` (3), `MAX_TRADES_PER_DAY`
(10) and `MAX_DAILY_LOSS` (5000 rupees), and use `CAPITAL_PER_TRADE` (100000
rupees of stock) to size each position.

## Read the win rate with care

Reversion strategies win most often because their targets are closer than
their stops. That is why the leaderboard ranks on the average result per trade
after costs, not on win rate. A strategy that wins 70% of the time can still
lose money.
