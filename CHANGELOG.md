# Changelog

All significant changes to the trading bot are documented here.
Each entry includes **what** changed, **why** it was needed, and the **observed problem** that motivated it.

Format: `[YYYY-MM-DD] commit-hash — short title`

---

## [2026-03-26] — Fix duplicate bot instances: PID path divergence, recycled-PID false stale, unsafe release

### Why
Telegram was logging `Conflict: terminated by other getUpdates request` continuously — two bot instances were polling simultaneously. This risks duplicate order placement and corrupted state.

### Root causes found

**Bug 1 — PID file resolved to two different paths (primary cause)**
`PID_FILE` was built with `os.path.dirname(__file__)`. When launched via launchd with an absolute path (`python3 /Users/.../main.py`), `__file__` is absolute and `dirname` returns the correct directory. When launched manually from the project directory (`python3 main.py`), `__file__` is `"main.py"` and `dirname` returns `""` — so the lock file resolved to `"data/bot.pid"` (CWD-relative) rather than `/Users/.../tradetheweather/data/bot.pid`. Each instance held a lock on a *different file* and neither saw the other.
- Fix: `_HERE = os.path.dirname(os.path.abspath(__file__))` — always resolves to the script's real directory regardless of invocation style.

**Bug 2 — PID recycling caused lock to block forever after SIGKILL**
If the bot was SIGKILLed (e.g., system shutdown escalation), `finally:` didn't run and the PID file stayed on disk. On the next launchd respawn, `os.kill(old_pid, 0)` succeeded if the OS had recycled that PID for an unrelated process — so the new bot instance saw "process alive, must be another bot" and exited. launchd then respawned every 30s, all failing — bot stayed dead indefinitely.
- Fix: After confirming the PID is alive, verify its command line contains `main.py` or `tradetheweather` via `ps`. If not, treat as stale and clear.

**Bug 3 — `_release_pid_lock()` could delete another instance's lock**
If the bot was signalled and began shutdown while launchd simultaneously respawned a new instance, the new instance could write its PID to the file before the old instance's `finally:` block ran `os.remove()` — deleting the new instance's lock.
- Fix: Read the PID file before removing; only delete if it still contains our own PID.

**Plist — ThrottleInterval too short, no ExitTimeout**
`ThrottleInterval=10` meant launchd could respawn within 10s of the previous exit. A slow shutdown (Telegram connection teardown, scheduler stop) could briefly overlap with the new instance's startup.
- Fix: Raise `ThrottleInterval` 10s → 30s. Add `ExitTimeout=30` so launchd sends SIGTERM and waits 30s before escalating to SIGKILL, giving `finally:` time to release the PID lock cleanly.

---

## [2026-03-26] 57a745e — Fix exit and entry logic: adverse stop, prob floor, weak-NO gate, hours cap

### Why
Post-trade analysis of recent losses identified four entry/exit patterns with negative expected value that were slipping through existing gates.

### Changes

**Exit — Adverse excursion stop (`pnl.py`)**
- Add Priority 4.5 adverse-excursion stop: exit NO position when YES price rises ≥50% above entry_NO price.
- *Problem it solves:* Trailing stop only arms after a gain. Positions that go adverse from entry (PHIL, DEN) held through full -$8.82 and -$2.25 moves. Gate would have saved $7.81 on those two alone.

**Entry — Hours cap tightened (`engine.py`)**
- Reduce `MAX_HOURS_TO_CLOSE` from 48h → 28h. Next-day contracts (30–40h) had 12% win rate vs 34% for same-day contracts.
- *Problem it solves:* NWS forecast skill beyond ~1 day is insufficient to overcome market friction. Bot was treating tomorrow like today.

**Entry — Minimum model probability (`engine.py`)**
- Add `MIN_OUR_PROB = 0.04`: reject when model assigns YES probability < 4%.
- *Problem it solves:* At extreme low probability, the estimate is bias-dominated rather than signal-driven. 22% win rate and -$9.11 total loss on 16 such trades.

**Entry — Weak-NO gate (`engine.py`)**
- Block NO entries at 25–40¢ when `our_prob > 0.15`.
- *Problem it solves:* Market pricing NO at 25–40¢ reflects genuine uncertainty. Model was ignoring this signal. 0% win rate, -$12 total loss on 6 such trades.

**Entry — Extreme low-prob size cap (`engine.py`)**
- When `our_prob < 0.05`, cap position at $1.50 regardless of other sizing rules.
- *Problem it solves:* Miscalibrated extreme outputs were being sized normally; limits damage when the model is least reliable.

**Retroactive simulation:** These gates together would have recovered $11.31 of recent losses and improved passing-trade win rate from 32% → 42%.

---

## [2026-03-26] 3fc0889 — Block temp_high/temp_low trades on non-today settlement dates

### Why
The NWS forecast pipeline only returns today's high/low. Scoring a tomorrow contract against today's data produces wrong signals and false edge readings.

### Changes
- Add `SAME_DAY_ONLY_MARKET_TYPES` gate in `_evaluate_market()` that hard-rejects `temp_high` and `temp_low` markets whose close time falls on any date other than today UTC.
- Rain and snow are exempt — storm-timing edge can legitimately span midnight.

---

## [2026-03-26] 5214303 — Fix starting_balance to include open position value, not just cash

### Why
The daily brake was triggering falsely when open positions settled and cash jumped, because the starting baseline was set from cash-only balance.

### Changes
- Starting balance now uses `portfolio_value` (cash + open position mark) from Kalshi's balance endpoint.
- *Problem it solves:* Settlement events caused cash to jump while open-position value dropped simultaneously — net flat, but the brake saw only the cash side and fired a false halt.

---

## [2026-03-24] 912be63 — Fix settlement accounting P0, session-halt logic, near-threshold gate, diagnostics

### Why
Multi-day audit (Mar 24) found three compounding bugs: settled positions weren't being detected, a halt on one market type was bleeding into unrelated types, and near-threshold markets were being traded when the forecast barely cleared the line.

### Changes

**P0 — Settlement detection (`orchestrator.py`)**
- Add `_detect_settled_positions()` called every 5 min to catch positions that expire naturally and record them via `record_exit(EXIT_EXPIRED)`.
- *Problem it solves:* Positions held to expiration stayed "open" in the tracker, masking ~$10 balance drop from daily brakes.

**P1 — Session-halt scope (`orchestrator.py`)**
- Fix `break → continue` so a `temp_low` halt doesn't block `temp_high`/`rain` all day.
- Add `can_trade_second_session()`: allows 1 high-conviction trade after 10am PT on non-halted market types.
- *Problem it solves:* One bad temp_low trade at 8am was silencing the entire day across all market types.

**P1 — Near-threshold fragile gate (`engine.py`)**
- Hard reject when raw forecast diff < 0.5°F (threshold too close to forecast).
- +8¢ edge surcharge when diff < 1.5°F.
- *Problem it solves:* Caught `KXLOWTPHIL` 32°F forecast vs 32.5°F threshold — near-certain to straddle, yet being traded.

**P2 — Missed opportunity logging**
- Add `_log_halted_missed_opportunity()` to record all blocked signals to `missed_opportunities.json` for post-session analysis.

---

## [2026-03-19] cdb6762 — Add weather thesis conflict engine: block contradictory positions on same event

### Why
Bot was placing contradictory positions on the same weather event (e.g., betting both YES and NO on temperature crossing different thresholds for the same city/day in conflicting directions).

### Changes
- Add `_weather_thesis()` + `_theses_conflict()` in `pnl.py`: detects direct, structural, and near-threshold contradictions (`CONFLICT_THRESH_F = 8°F`).
- Engine deduplication now keys on `(city, market_type, settlement_date)`.
- Selection pass checks inter-candidate conflicts before accepting.
- `check_correlation_limits()` checks new trade's thesis vs all open positions on same event key.
- Reinforcing ladders (same-direction or gap ≥ 8°F) are still allowed — only directional contradictions are blocked.

---

## [2026-03-18] e530cf3 — Add profit-tilt: operating profiles, trade tiers, quality-hold tolerance

### Why
All trades were being sized and held identically regardless of quality. High-conviction opportunities were being missed or undersized while low-quality trades got equal treatment.

### Changes
- New `operating_profile.py`: `protection_first` / `balanced` (default) / `profit_tilted` profiles controlled via `OPERATING_PROFILE` env var.
- Trade tier classification: `top_tier` / `standard` / `marginal` after conviction scoring. Top-tier requires high confidence, strong edge, signal agreement, low uncertainty, good liquidity, and no fragile flags.
- Top-tier size uplift: +25% contracts for `balanced` profile, +40% for `profit_tilted`, +0% for `protection_first`. All absolute caps enforced after uplift.
- Quality-based holding tolerance: high-quality positions get extra stall escalation cycles and fair-value grace time.
- Daily trade cap raised from 10 → 15.

---

## [2026-03-18] a350aaa — Tighten buy logic: low-price fragile gate, dangerous disagree classification, temp_low conservatism

### Why
Mar 18 audit showed a pattern: low-price entries (<20¢) with high model-vs-market disagreement and weak signal agreement were losing at a disproportionate rate. The classifier couldn't distinguish "market is wrong" from "model is miscalibrated."

### Changes
- `_classify_disagreement()`: splits large disagreement into `actionable` vs `dangerous` based on price, sigma, signal agreement, forecast freshness, and calibration level.
- `_low_price_fragile_gate()`: hard block for sub-20¢ entries with dangerous disagreement (disagree >35%, weak agreement, or extreme sigma). Requires ≥65% signal agreement as hard floor for low-price entries.
- Denver-style penalty leak fixed: hard minimum edge floor (12¢) for any warn/penalty temp_low regardless of raw edge size.
- Extra +5¢ surcharge when `temp_low` sigma > 10°F.
- `LOW_PRICE_ENTRY_SNAPSHOT` structured log line for every accepted low-price entry.
- *Retro: NYC-T27 gate would have blocked it (signal_agreement=50% < 65%).*

---

## [2026-03-18] 23d5bc3 — Fix multi-contract order logging and add 10-trade/day conviction cap

### Why
Logs were ambiguous about whether multi-contract orders were one order or many. No ranking or cap meant the bot placed any passing trade regardless of relative quality.

### Changes
- Add `ORDER_SUBMIT` / `ORDER_CONFIRMED` log lines that explicitly state "aggregated single order".
- Add `_conviction_score()`: multi-factor ranking (50% adjusted edge, 20% signal agreement, 15% model certainty, 10% liquidity, 5% confidence level).
- `get_recommendations()`: rank all deduplicated candidates by conviction, cap at 10/day, log full `CANDIDATE_RANKING`.
- `MAX_TRADES_PER_DAY` lowered from unlimited → 10.

---

## [2026-03-17] c812000 — Fix temp_low calibration: overnight window, bias correction, sigma widening

### Why
Three separate bugs caused the temp_low model to use wrong data for calibration — wrong time window for actual lows, bias correction code that existed but never ran, and sigma values based on no temp_low data at all.

### Changes

**P0 — Settlement error recording (`orchestrator.py`)**
- `actual_low` was using `min()` of all 24 hourly METARs, including daytime temps. Replaced with timezone-aware overnight window (18:00 prev day → 10:00 local) to match Kalshi's settlement period.
- `actual_high` uses 06:00–21:00 local daytime window.
- *Problem it solves:* Daytime lows (e.g., 45°F midday) were masking real overnight lows (e.g., 28°F) in calibration data.

**P1 — Calibration bias application (`engine.py`)**
- `_load_forecast_calibration()`: reads `forecast_errors.json` at score time (was never applied before).
- `_get_temp_low_bias_status()`: classifies cities as `ok/warn/penalty/block`.
- Apply `bias_correction = -mean_bias * 0.5` and `sigma_penalty = +4°F` at PENALTY level.
- *Problem it solves:* Cities like Austin (mean_err = -26°F) were being scored as if the forecast were accurate.

**P1 — sigma values for temp_low (`city_uncertainty.json`)**
- Populate with empirical MAE from 5 days of data. Phoenix/Austin/San Antonio now use 25–29°F sigma; Houston/Denver/Dallas use 18–21°F.
- *Problem it solves:* All cities were defaulting to 7°F sigma, wildly underestimating uncertainty in high-variance cities.

**P2 — Calibration edge surcharges and hard block**
- WARN (n≥2, mean<-8°F): +3¢ edge requirement.
- PENALTY (n≥3, mean<-15°F): +5¢ edge requirement.
- BLOCK (n≥4, mean<-25°F): reject trade entirely.

---

## [2026-03-17] 6f49fa3 — Fix stall audit open items and atomic history writes

### Changes
- P0: Move `_save()` out of `classify_stalled_positions` to after `check_profit_takes()` succeeds — prevents stall count overcounting if an exception is thrown mid-cycle.
- P1: Add `STALL_EXIT_MIN_MARK_CENTS = 10` floor so near-worthless positions (3¢ mark) are not force-exited for noise.
- P1: Carry `stall_alert_counts` forward at midnight rollover so multi-day stalled positions don't reset escalation to 0.
- P1: Skip orders not yet in `_known_resting_ids` in `_update_pending_buy_capital` to close race window where a fresh exit order gets counted as pending buy capital.

---

## [2026-03-17] 0721a46 — Fix stall-classification crash and classifier missing EXIT_STALLED

### Changes
- Fix `AttributeError` crash in `classify_stalled_positions()` when position lacked `entry_price`.
- Add `EXIT_STALLED` to classifier's known exit reason set so stalled exits generate lessons instead of being silently ignored.

---

## [2026-03-17] 418d1fa — Fix capital management review: stall dedup, time-aware EV, forced exit, pending staleness

### Changes
- Stall dedup: don't re-classify positions that already have a stall alert in the current cycle.
- Time-aware EV: weight hold EV by hours remaining so a 5% edge with 30 min left isn't treated equally to 5% edge with 18 hours left.
- Forced exit: `EXIT_STALLED` positions with 0 escalation cycles remaining now force-place an exit order.
- Pending staleness: expire pending buy tracking after 15 min if no fill confirmation received.

---

## [2026-03-17] 43d7458 — Improve capital management: effective deployable capital, pending buy tracking, stalled position detection

### Why
Bot was deploying capital it didn't actually have because pending buy orders weren't subtracted from available balance before placing new orders.

### Changes
- Replace raw-cash reserve check with `effective_available_capital` (cash minus reserve minus resting buy commitments).
- Add `DailyState.pending_buy_dollars`, updated every fill cycle from resting orders.
- Add `_update_pending_buy_capital()` excluding bot exit orders to avoid double-counting.
- Add `get_effective_deployable_capital()` with full breakdown for structured logging.
- Add `classify_stalled_positions()`: flags positions when ≥3 of: weak hold EV, weak exit EV, wide spread, poor liquidity, no favorable excursion, late and losing.

---

## [2026-03-17] 04c5c68 — Add trim-band entry guard and fair-value grace period to fix buy/sell coordination

### Why
Two coordination bugs were causing the bot to buy and immediately try to sell — burning spread twice for zero gain.

### Changes

**Trim-band entry guard (`engine.py`)**
- Reject in-band entries (70–99¢) unless edge clears an extra hurdle and confidence == "high". Entries at 90¢+ always rejected.
- *Problem it solves:* Fresh entries already inside a staged profit-taking band were being approved on raw edge, then immediately trimmed for no monetizable gain.

**Fair-value grace period (`pnl.py`)**
- Add `FAIR_VALUE_GRACE_MINUTES = 30`: suppress fair-value-only exits for 30 minutes after entry.
- *Problem it solves:* `_model_hold_value` anchors to market price then discounts by a risk buffer — so `hold_ev < exit_ev` by construction on any fresh position above ~37¢. The sell logic was undoing entries within the first 5-minute balance-refresh cycle.
- Emergency exits (thesis invalidation, trailing stop, salvage stop, daily brake) are not gated.

---

## [2026-03-17] 96bc35c — Fix false profit-target halt caused by Kalshi balance timing

### Why
Kalshi's available balance lags behind order placement. Adding open position cost to a lagged balance caused double-counting that falsely triggered the +5% profit target halt.

### Changes
- Switch profit target check to use `realized_pnl` (tracked internally, increments only on `record_exit`) instead of `_portfolio_value()`.
- Stop-loss/brake checks keep using `_portfolio_value()` — downside safety should remain conservative.

---

## [2026-03-16] ee843aa — Add entry snapshots, exit reason constants, MAE/fragile-trade tracking, and learning safeguards

### Why
Post-trade analysis lacked structured data — exit reasons were free-form strings, entry conditions weren't preserved, and the advisor was making strategy recommendations on accounts too small and datasets too thin to be reliable.

### Changes
- `pnl.py`: Add `EXIT_*` constants for stable exit reason strings. Add `low_water_mark` (MAE), `exit_reason`, full entry snapshot, and `fragile_flags` to `Position`. Compute fragile flags at entry.
- `classifier.py`: Fix win-rate in `classify_city_penalty()` to exclude scratches from denominator (scratches were suppressing apparent win rate for good cities).
- `advisor.py`: Add `SAFEGUARD_MIN_BALANCE_FOR_STRATEGY_CHANGE` and `SAFEGUARD_MIN_DAYS_FOR_PENALTY` to cap aggressive recommendations on thin accounts/data.

---

## [2026-03-16] b4bf313 — Fix EOD balance to include open positions, exclude UNKNOWN from best city

### Changes
- EOD report now shows total portfolio value (cash + open position cost basis) instead of cash-only.
- Best/worst city ranking excludes `UNKNOWN` and empty city keys so real cities surface in the analysis.

---

## [2026-03-15] c33d835 — Fix P&L calc to include open position value, not just cash balance

### Why
Cash balance alone understates portfolio value when capital is deployed. Halt thresholds were firing against a lower-than-actual baseline.

### Changes
- Add `_portfolio_value()`: sum of cash + open position cost basis.
- Used in `_check_rules()` and `get_summary()` so halt thresholds and reported P&L % reflect true account value.

---

## [2026-03-15] 46df730 — Overhaul signals, tracker, and add analysis modules

### Why
Signal modules were producing inconsistent output formats and the tracker lacked the structure needed to derive lessons from trades.

### Changes
- Refactor all 6 signal modules with improved logic and structured output.
- Add `cap_regime` / `suggested_cap` / `weights_version` to `AggregatedSignal`.
- Remove unused signal modules: `historical_analog`, `probability_surface`, `radar_analysis`, `weather_models`.
- Update aggregator with context-aware cap regimes and hot-reloadable `signal_weights.json`.
- Improve `PnLTracker` exit logic: staged profit-taking, trailing stops, thesis invalidation.
- Add analysis modules: `advisor`, `classifier`, `uncertainty_recalibrator`, `validation`.

---

## [2026-03-15] 5c3fd81 — Fix 7 production bugs from code review (P0/P1)

### Why
Code review after first live days found fundamental correctness bugs — wrong probability direction for temp_low, wrong timestamps for freshness, wrong bias sign, wrong position floor.

### Changes
- **P0** — Fix `temp_low` probability direction: was computing P(low > threshold) instead of P(low ≤ threshold).
- **P0** — Use real NWS `generatedAt` timestamp (not `datetime.now()`) for forecast freshness checks.
- **P0** — Invert station bias sign for temp_low: warm NWS bias now correctly increases YES probability.
- **P0** — Replace hardcoded $1 position floor with relative `max(0.10, 1% of budget)` to prevent oversizing on small accounts.
- **P1** — Fix NO-position exit price: now converts NO bid to YES ask (`100 - price`) so exit limit orders actually fill.
- **P1** — Add 2s delay before reconcile query to avoid false "exit may not have processed" warnings from Kalshi settlement lag.
- **P1** — Guard temp_low warming-trend cutoff: skip only when `current_temp <= threshold + 5.0`.

---

## [2026-03-15] d3b9295 — Remove unused advisor module and atomic PID lock on startup

### Changes
- Delete `src/advisor/` (auto_implementer, gpt_advisor) — nothing imported from it.
- `main.py`: atomic PID lock using `O_CREAT|O_EXCL` to prevent two bot instances starting in a race.

---

## [2026-03-15] 20d2009 — Fix bugs found during code review

### Changes
- Fix `_parse_ticker` bug: was checking `"LOWT"` twice, never matching `"LOW"` — broke temp_low position classification and exit logic.
- Fix city edge adjustment: engine looked for `"count"` key but history stores `"trades"` — bad-track-record filter never activated.
- Remove dead `format_recommendation` method, unused `TradeAnalysisEngine` import.
- Fix `send_learning_analysis` to use existing `orchestrator.history` instead of creating a new tracker instance.
- Fix inconsistent "Order filled" alert message.

---

## [2026-03-13] 8ba1b0f — Fix probability model: wider uncertainty, bucket markets, disable bad bias data

### Why
Initial uncertainty of 3°F produced overconfident probabilities. Bucket markets (B-suffix) were being scored with the wrong model. Bias correction was running on corrupted duplicate data.

### Changes
- `UNCERTAINTY_F`: 3.0 → 7.0°F.
- Bucket markets: compute P(temp in [threshold, threshold+2]) instead of P(temp > threshold).
- Disable warm-bias correction pending clean data accumulation — `forecast_errors.json` had duplicate writes inflating bias values (e.g., Denver +22.5°F, Phoenix -26°F).
- Fix duplicate writes: deduplicate by city+date before appending.
- Clear corrupted `forecast_errors.json`.

---

## [2026-03-13] f004376 — Fix three bugs that caused full account loss

### Why
Bot deployed the full account on a single day because position sizing, market timing, and city mapping were all wrong simultaneously.

### Changes
- **Market timing**: skip same-day markets after 18:00 UTC — today's high has already occurred, NWS returns tomorrow's forecast instead.
- **Position sizing**: hard cap $5/trade, 10% max position (was 20%), prevent 880-contract bets on penny markets.
- **City/series mapping**: fix KXHIGHTSFO→SAN_FRANCISCO (was mapped to LA), add SF/Austin/OKC/DC to `US_CITIES` with correct NWS grid coordinates.
- Raise `MIN_PRICE_CENTS` 10→15 and `MIN_HOURS_TO_CLOSE` 4→8.

---

## [2026-03-12] 532c394 — Add market timing and price floor guards to prevent bad trades

### Why
Bot placed 880-contract 1¢ bets on markets closing within hours, tying up the full account balance.

### Changes
- Skip markets closing within 4 hours (outcome already priced in).
- Skip markets priced below 10¢ (near-certain, no real trading edge).

---

## [2026-03-12] 2c12391 — Fix order placement: send only yes_price, not both prices

### Why
All order placements were failing with 400 Bad Request.

### Changes
- Kalshi API requires exactly one of `yes_price` or `no_price` per order — sending both caused rejection on every trade attempt.

---

## [2026-03-12] f0cb524 — Fix Kalshi RSA signature to use full API path

### Why
All API calls returning 401 Unauthorized.

### Changes
- RSA signature requires the full path `/trade-api/v2/...`, not just the endpoint suffix. Without this, all authenticated requests failed.

---

## [2026-03-12] 68c030c — Fix market scanning: correct Kalshi API format, ticker parsing, and forecast timing

### Changes
- `get_weather_markets`: fetch by known series tickers instead of broken keyword filter on generic `/markets` endpoint.
- `_parse_market`: add `SERIES_CITY_MAP` for direct series→city lookup; fix threshold regex for `-T96`/`-B95.5` suffix format.
- `get_liquidity`: parse `orderbook_fp.yes_dollars/no_dollars` format; convert decimal prices to cents.
- `get_forecast`: handle evening calls where `periods[0]` is nighttime (not daytime); correctly assign tonight's low vs tomorrow's high.

---

## [2026-03-12] b3b1219 — Initial commit: full bot codebase

### Summary
- Orchestrator, 12 signal modules, advisor, PnL tracker, Telegram bot.
- Kalshi RSA-PSS signing against `api.elections.kalshi.com`.
- NWS warm-bias correction using `forecast_errors.json`.
- Market-disagreement filter blocks trades where model prob ≥ 2× market price.
- Daily P&L rules: +5% profit target, -5% stop loss, 5 trades/day max.
- APScheduler: scan, balance refresh, EOD report, settlement error recording.
- Telegram approval workflow for GPT advisor suggestions.
