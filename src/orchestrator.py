"""
Orchestrator — the main coordinator.
Manages scheduled jobs, trade execution, and bot communication.
"""

import asyncio
import json
import logging
import os
from datetime import date, datetime, time, timedelta, timezone

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.kalshi.client import KalshiClient
from src.weather.pipeline import WeatherPipeline
from src.analysis.engine import TradeAnalysisEngine
from src.analysis.advisor import run_advisor_session
from src.analysis.uncertainty_recalibrator import run_recalibration_session
from src.tracker.pnl import PnLTracker
from src.tracker.history import DailyHistoryTracker, _get_season_for_date

logger = logging.getLogger(__name__)

SCAN_INTERVAL_MINUTES = 10
MAX_TRADES_PER_DAY = 999999  # unlimited — cash reserve enforces the real limit

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

# METAR station mapping for settlement error recording
CITY_STATIONS = {
    "LOS_ANGELES": "KLAX",
    "DENVER": "KDEN",
    "DALLAS": "KDFW",
    "SEATTLE": "KSEA",
    "NYC": "KNYC",
    "CHICAGO": "KORD",
    "SAN_FRANCISCO": "KSFO",
    "HOUSTON": "KHOU",
    "BOSTON": "KBOS",
    "PHILADELPHIA": "KPHL",
    "ATLANTA": "KATL",
    "DC": "KDCA",
    "SAN_ANTONIO": "KSAT",
    "AUSTIN": "KAUS",
    "OKLAHOMA_CITY": "KOKC",
    "PHOENIX": "KPHX",
    "MIAMI": "KMIA",
    "LAS_VEGAS": "KLAS",
    "MINNEAPOLIS": "KMSP",
    "NASHVILLE": "KBNA",
}

NWS_BASE = "https://api.weather.gov"
ERRORS_FILE = os.path.join(DATA_DIR, "forecast_errors.json")


class Orchestrator:
    def __init__(self):
        self.kalshi = KalshiClient()
        self.weather = WeatherPipeline()
        self.analysis = TradeAnalysisEngine(self.kalshi, self.weather)
        self.tracker = PnLTracker(self.kalshi, self.weather)
        self.history = DailyHistoryTracker()
        self.scheduler = AsyncIOScheduler()
        self.bot = None
        self._known_resting_ids: set = set()
        self._bot_exit_order_ids: set = set()
        # Maps exit order_id -> (position, exit_price_cents, contracts_to_exit)
        self._pending_exits: dict = {}
        self.started_at = datetime.now(timezone.utc)
        # Scan coordination
        self._scan_lock = asyncio.Lock()
        # Snapshot for fast-scan change detection: city -> {last_obs_time, forecast_age_h}
        self._last_scan_snapshot: dict = {}

    def set_bot(self, bot):
        self.bot = bot

    # -------------------------------------------------------------------------
    # Scheduler setup
    # -------------------------------------------------------------------------

    def start_scheduler(self):
        """Register all scheduled jobs and start the scheduler."""
        # Balance refresh every 5 minutes
        self.scheduler.add_job(
            self._balance_refresh_job,
            "interval",
            minutes=5,
            id="balance_refresh",
            name="Balance refresh + fill check",
        )

        # Market scan every 10 minutes (full)
        self.scheduler.add_job(
            self._scan_job,
            "interval",
            minutes=SCAN_INTERVAL_MINUTES,
            id="scan_job",
            name="Market scan",
        )

        # Fast event-driven scan every 2 minutes (targeted: final-window + fresh data only)
        self.scheduler.add_job(
            self._fast_scan_job,
            "interval",
            minutes=2,
            id="fast_scan",
            name="Fast signal scan",
        )

        # End-of-day close at 5pm PT
        self.scheduler.add_job(
            self._end_of_day_close_job,
            "cron",
            hour=17,
            minute=0,
            timezone="America/Los_Angeles",
            id="eod_close",
            name="End-of-day close",
        )

        # Midnight reset at 12:01am PT
        self.scheduler.add_job(
            self._midnight_reset_job,
            "cron",
            hour=0,
            minute=1,
            timezone="America/Los_Angeles",
            id="midnight_reset",
            name="Midnight reset",
        )

        # Settlement backfill at 9am PT
        self.scheduler.add_job(
            self._settlement_backfill_job,
            "cron",
            hour=9,
            minute=0,
            timezone="America/Los_Angeles",
            id="settlement_backfill",
            name="Settlement backfill",
        )

        # Morning briefing at 7am PT
        self.scheduler.add_job(
            self._morning_briefing_job,
            "cron",
            hour=7,
            minute=0,
            timezone="America/Los_Angeles",
            id="morning_briefing",
            name="Morning briefing",
        )

        # Uncertainty recalibration — weekly on Sunday at 2am PT
        # Runs after settlement errors have accumulated during the week.
        self.scheduler.add_job(
            self._uncertainty_recalibration_job,
            "cron",
            day_of_week="sun",
            hour=2,
            minute=0,
            timezone="America/Los_Angeles",
            id="uncertainty_recal",
            name="Uncertainty recalibration",
        )

        # Startup scan 15 seconds from now
        startup_scan_time = datetime.now(timezone.utc) + timedelta(seconds=15)
        self.scheduler.add_job(
            self._scan_job,
            "date",
            run_date=startup_scan_time,
            id="startup_scan",
            name="Startup scan",
        )

        self.scheduler.start()
        logger.info("Scheduler started with all jobs registered.")

    # -------------------------------------------------------------------------
    # Balance refresh + fill check
    # -------------------------------------------------------------------------

    def _update_pending_buy_capital(self, resting_orders: list):
        """
        Compute capital reserved by resting BUY orders (not exit orders placed by
        this bot) and write it into tracker.state.pending_buy_dollars.

        Kalshi order fields used:
          - action: "buy" | "sell"
          - side: "yes" | "no"
          - yes_price: limit price in cents for the YES side
          - remaining_count: unfilled contracts still resting

        Capital reserved = sum over resting buy orders of:
            (price_cents / 100) * remaining_count
        where price_cents is the price the buyer is paying (yes_price for YES buys,
        (100 - yes_price) for NO buys).

        Exit orders (those in _bot_exit_order_ids or _pending_exits) are excluded —
        they represent sells, not new capital deployment.
        """
        # P1-4 fix: exclude IDs that are registered exit orders OR were not seen in
        # a prior cycle. Any brand-new resting buy order that appeared since the last
        # _check_order_fills tick could be an exit order whose ID hasn't been
        # registered yet (exit_position returns the ID, but _update_pending_buy_capital
        # may run in run_scan concurrently). Skipping truly-new orders is conservative
        # (we may under-count pending buys by one cycle) but avoids over-counting capital.
        exit_ids = self._bot_exit_order_ids | set(self._pending_exits.keys())
        total_pending = 0.0
        for o in resting_orders:
            oid = o.get("order_id", "")
            if oid in exit_ids:
                continue
            # Skip orders not yet seen in a previous cycle — they could be unregistered exits
            if oid and oid not in self._known_resting_ids:
                continue
            action = o.get("action", "")
            if action != "buy":
                continue
            yes_price = o.get("yes_price", 0) or 0
            side = o.get("side", "yes")
            price_cents = yes_price if side == "yes" else (100 - yes_price)
            remaining = o.get("remaining_count", 0) or 0
            total_pending += (price_cents / 100.0) * remaining

        prev = self.tracker.state.pending_buy_dollars
        self.tracker.state.pending_buy_dollars = total_pending
        if abs(total_pending - prev) >= 0.01:
            logger.info(
                f"PENDING_BUY_CAPITAL updated: ${total_pending:.2f} "
                f"(was ${prev:.2f}, delta ${total_pending - prev:+.2f}) "
                f"from {len(resting_orders)} resting orders"
            )

    async def _balance_refresh_job(self):
        """Refresh balance, check fills, and enforce profit targets."""
        try:
            await self._check_order_fills()

            was_halted = self.tracker.state.trading_halted
            self.tracker.refresh_balance()

            # Alert if daily brake just changed
            new_brake = self.tracker.state.daily_brake_level
            if not was_halted and self.tracker.state.trading_halted:
                if self.bot:
                    await self.bot.send_alert(
                        f"TRADING HALTED\n{self.tracker.format_summary()}"
                    )

            # Check for stalled / capital-trap positions and alert if warranted
            try:
                stalled = self.tracker.classify_stalled_positions()
                for report in stalled:
                    # Only alert when the report says to (deduped by stall cycle count)
                    if not report.get("should_alert", False):
                        continue
                    # Always log monitor-only at WARNING; only Telegram-alert for actionable cases
                    cycle = report.get("stall_cycle", 1)
                    urgent_marker = " ★URGENT" if report.get("is_urgent") else ""
                    repeat_context = f" (repeat #{cycle})" if cycle > 1 else " (first detection)"
                    if self.bot and report["action"] in ("escalate_fair_value_exit", "attempt_partial_reduction"):
                        hrs = report["hours_left"]
                        hrs_str = f"{hrs:.1f}" if hrs is not None else "N/A"
                        await self.bot.send_alert(
                            f"CAPITAL TRAP [{report['action'].upper()}]{urgent_marker}{repeat_context}\n"
                            f"{report['ticker']} {report['side'].upper()} | "
                            f"age={report['age_minutes']:.0f}min "
                            f"mark={report['mark_cents']}¢ state={report['state']} "
                            f"hrs_left={hrs_str} | "
                            f"hold_ev={report['hold_ev']:.1f}¢ exit_ev={report['exit_ev']:.1f}¢ "
                            f"exitability={report['exitability_score']}/100 | "
                            f"flags={report['stall_flags']}"
                        )
            except Exception as e:
                logger.error(f"Stalled position check error: {e}")

            # Check per-position exits (profit takes, trailing stops, thesis invalidation)
            profit_takes = self.tracker.check_profit_takes()
            # Persist stall counts now that check_profit_takes succeeded — P0-2 fix:
            # saving here (not inside classify_stalled_positions) ensures counts are
            # only written when the full cycle completes without exception.
            self.tracker._save()
            for position, exit_price, contracts_to_exit, exit_reason in profit_takes:
                try:
                    if contracts_to_exit <= 0:
                        logger.warning(f"Skipping exit for {position.ticker}: contracts_to_exit={contracts_to_exit}")
                        continue

                    order = self.kalshi.exit_position(
                        position.ticker, position.side, contracts_to_exit, exit_price
                    )
                    order_data = order.get("order", {})
                    exit_order_id = order_data.get("order_id", "")
                    order_status = order_data.get("status", "")

                    if exit_order_id:
                        self._bot_exit_order_ids.add(exit_order_id)

                    if position.side == "yes":
                        pnl = (exit_price / 100 - position.entry_price / 100) * contracts_to_exit
                    else:
                        pnl = (position.entry_price / 100 - exit_price / 100) * contracts_to_exit

                    is_partial = contracts_to_exit < position.contracts

                    if order_status == "executed":
                        # Filled immediately — record and notify
                        self.tracker.record_exit(
                            position.order_id, exit_price, pnl, contracts_to_exit,
                            exit_reason=exit_reason,
                        )
                        reconciled = await self._reconcile_exit(
                            position.ticker, position.side, position.contracts, contracts_to_exit
                        )
                        msg = (
                            f"{'Partial exit' if is_partial else 'Profit take'}: {position.ticker} {position.side.upper()} "
                            f"{contracts_to_exit}/{position.contracts} contracts @ {exit_price}¢ | P&L: ${pnl:+.2f}"
                        )
                        if not reconciled:
                            msg += " (reconciliation warning)"
                        if self.bot:
                            await self.bot.send_alert(msg)
                        logger.info(msg)
                    elif exit_order_id:
                        # Order is resting — store pending with exit_reason so it
                        # is recorded when the order fills in _check_order_fills.
                        self._pending_exits[exit_order_id] = (
                            position, exit_price, contracts_to_exit, pnl, is_partial, exit_reason
                        )
                        logger.info(
                            f"Exit order resting: {position.ticker} {position.side.upper()} "
                            f"{contracts_to_exit}x @ {exit_price}¢ | order_id={exit_order_id} "
                            f"reason={exit_reason}"
                        )
                    else:
                        logger.warning(f"Exit order for {position.ticker} returned no order_id and status='{order_status}'")

                except Exception as e:
                    logger.error(f"Error exiting position {position.ticker}: {e}")

        except Exception as e:
            logger.error(f"Balance refresh job error: {e}")

    async def _check_order_fills(self):
        """
        Compare current resting orders against known set.
        Notify on fills (orders that disappeared from resting list).
        Resolves pending exit orders when they fill.
        Also updates pending_buy_dollars in tracker state for capital accounting.
        """
        try:
            current_resting = self.kalshi.get_orders(status="resting")
            current_ids = {o["order_id"] for o in current_resting if "order_id" in o}

            filled_ids = self._known_resting_ids - current_ids
            for oid in filled_ids:
                if oid in self._pending_exits:
                    # A bot-placed exit order just filled — record and notify.
                    # Unpack 6-tuple (position, exit_price, contracts, pnl, is_partial, exit_reason).
                    pending = self._pending_exits.pop(oid)
                    if len(pending) == 6:
                        position, exit_price, contracts_to_exit, pnl, is_partial, exit_reason = pending
                    else:
                        # Legacy 5-tuple (no exit_reason) — degrade gracefully
                        position, exit_price, contracts_to_exit, pnl, is_partial = pending
                        exit_reason = ""
                    self._bot_exit_order_ids.discard(oid)
                    self.tracker.record_exit(
                        position.order_id, exit_price, pnl, contracts_to_exit,
                        exit_reason=exit_reason,
                    )
                    msg = (
                        f"{'Partial exit' if is_partial else 'Profit take'}: {position.ticker} {position.side.upper()} "
                        f"{contracts_to_exit}/{position.contracts} contracts @ {exit_price}¢ | P&L: ${pnl:+.2f}"
                    )
                    if self.bot:
                        await self.bot.send_alert(msg)
                    logger.info(msg)
                elif oid in self._bot_exit_order_ids:
                    self._bot_exit_order_ids.discard(oid)
                else:
                    msg = f"Order filled: {oid}"
                    logger.info(msg)
                    if self.bot:
                        await self.bot.send_alert(msg)

            self._known_resting_ids = current_ids

            # Update pending buy capital for accurate deployable-capital accounting
            self._update_pending_buy_capital(current_resting)

        except Exception as e:
            logger.error(f"Check order fills error: {e}")

    # -------------------------------------------------------------------------
    # Market scan
    # -------------------------------------------------------------------------

    async def _scan_job(self):
        """Run a full market scan with a 180-second timeout, protected by scan lock."""
        if not self._scan_lock.locked():
            try:
                async with self._scan_lock:
                    await asyncio.wait_for(self.run_scan(), timeout=180)
            except asyncio.TimeoutError:
                logger.warning("Market scan timed out after 180s")
            except Exception as e:
                logger.error(f"Scan job error: {e}")
        else:
            logger.debug("Full scan skipped: scan already in progress")

    async def _fast_scan_job(self):
        """
        Lightweight 2-minute scan targeting high-signal events only:
          - Final-window markets (< 3h to close)
          - Cities with fresh forecast updates (detected via snapshot diff)
          - Cities with new METAR observations since last scan
        Skips full evaluation if trading is halted. Uses scan lock to prevent overlap.
        """
        if self.tracker.state.trading_halted:
            return
        if not self._scan_lock.locked():
            try:
                async with self._scan_lock:
                    await asyncio.wait_for(self._run_fast_scan(), timeout=45)
            except asyncio.TimeoutError:
                logger.warning("Fast scan timed out after 45s")
            except Exception as e:
                logger.error(f"Fast scan job error: {e}")
        else:
            logger.debug("Fast scan skipped: full scan in progress")

    async def _run_fast_scan(self):
        """
        Determines which cities have fresh signals and triggers a targeted full scan
        for those specific markets only. Updates _last_scan_snapshot.
        """
        now_utc = datetime.now(timezone.utc)
        hot_cities: set = set()

        # Always include cities with open positions that have < 3h left (final window)
        for pos in self.tracker.state.positions:
            if pos.status != "open":
                continue
            if pos.city:
                try:
                    market_data = self.kalshi.get_market(pos.ticker) or {}
                    ct_str = market_data.get("market", {}).get("close_time") or ""
                    if ct_str:
                        ct = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
                        if (ct - now_utc).total_seconds() < 3 * 3600:
                            hot_cities.add(pos.city)
                except Exception:
                    pass

        # Check for fresh data changes vs last snapshot
        for city, snap in list(self._last_scan_snapshot.items()):
            try:
                report = await asyncio.get_event_loop().run_in_executor(
                    None, lambda c=city: self.weather.get_full_report(c)
                )
                obs = report.get("recent_observations", [])
                new_obs_time = obs[0].get("timestamp", "") if obs else ""
                last_obs_time = snap.get("last_obs_time", "")

                fc_generated = report.get("forecast", {}).get("generated_at", "")
                last_fc = snap.get("last_forecast_generated", "")

                obs_changed = new_obs_time != last_obs_time and new_obs_time
                fc_changed = fc_generated != last_fc and fc_generated

                if obs_changed or fc_changed:
                    hot_cities.add(city)
                    logger.debug(
                        f"Fast scan hot: {city} "
                        f"obs_changed={obs_changed} fc_changed={fc_changed}"
                    )

                # Update snapshot
                self._last_scan_snapshot[city] = {
                    "last_obs_time": new_obs_time,
                    "last_forecast_generated": fc_generated,
                }
            except Exception:
                pass

        if not hot_cities:
            logger.debug("Fast scan: no hot cities detected")
            return

        logger.info(f"Fast scan: hot cities={hot_cities} — running targeted evaluation")

        # Run a targeted full scan (engine handles per-market filtering)
        daily_budget = self.tracker.state.current_balance
        trades_used = self.tracker.state.trades_placed
        open_position_cost = sum(
            p.cost_dollars for p in self.tracker.state.positions if p.status == "open"
        )
        try:
            recommendations = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.analysis.get_recommendations(
                    daily_budget, trades_used, open_position_cost
                ),
            )
        except Exception as e:
            logger.error(f"Fast scan recommendations error: {e}")
            return

        # Filter recommendations to hot cities only
        hot_recs = [r for r in recommendations if r.city in hot_cities]
        logger.info(f"Fast scan: {len(hot_recs)} hot recommendations from {hot_cities}")

        for rec in hot_recs:
            can, reason = self.tracker.can_trade()
            if not can:
                if self.tracker.state.goal_met and self.tracker.is_high_conviction_exception(rec):
                    self.tracker.record_goal_exception()
                else:
                    break
            corr_ok, corr_reason = self.tracker.check_correlation_limits(rec)
            if not corr_ok:
                logger.info(f"Fast scan SKIPPED {rec.ticker}: {corr_reason}")
                continue
            await self.execute_trade(rec)

    async def run_scan(self):
        """
        Gets trade recommendations and executes approved ones if trading is allowed.
        """
        logger.info("Starting market scan...")

        # Refresh pending buy capital immediately before evaluating can_trade().
        # The balance-refresh job runs every 5 min; the scan runs every 10 min.
        # Without this, pending_buy_dollars could be up to ~5 min stale, causing
        # can_trade() to see an overly optimistic effective-available-capital value
        # if buy orders filled or were placed since the last fill check.
        try:
            current_resting = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.kalshi.get_orders(status="resting")
            )
            self._update_pending_buy_capital(current_resting)
        except Exception as e:
            logger.warning(f"run_scan: could not refresh pending buy capital: {e}")

        daily_budget = self.tracker.state.current_balance
        trades_used = self.tracker.state.trades_placed
        open_position_cost = sum(
            p.cost_dollars for p in self.tracker.state.positions if p.status == "open"
        )

        try:
            recommendations = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.analysis.get_recommendations(
                    daily_budget, trades_used, open_position_cost
                ),
            )
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return

        logger.info(f"Scan found {len(recommendations)} recommendations.")

        # Update fast-scan snapshot with cities from current recommendations
        now_utc = datetime.now(timezone.utc)
        for rec in recommendations:
            city = rec.city
            if city and city not in self._last_scan_snapshot:
                self._last_scan_snapshot[city] = {
                    "last_obs_time": "",
                    "last_forecast_generated": "",
                }

        for rec in recommendations:
            can, reason = self.tracker.can_trade()
            if not can:
                # Check for high-conviction exception past daily profit halt
                if self.tracker.state.goal_met and self.tracker.is_high_conviction_exception(rec):
                    logger.info(
                        f"Goal-exception trade: {rec.ticker} qualifies "
                        f"(edge={rec.edge:.3f}, confidence={rec.confidence})"
                    )
                    self.tracker.record_goal_exception()
                else:
                    logger.info(f"Trading halted: {reason}")
                    break

            # Correlated exposure check
            corr_ok, corr_reason = self.tracker.check_correlation_limits(rec)
            if not corr_ok:
                logger.info(f"SKIPPED {rec.ticker}: correlation limit — {corr_reason}")
                continue

            await self.execute_trade(rec)

    async def execute_trade(self, rec):
        """Places an order for a recommendation and records it."""
        try:
            # Prevent duplicate orders: skip if a resting order already exists for this ticker
            try:
                resting = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.kalshi.get_orders(status="resting")
                )
                resting_orders = resting if isinstance(resting, list) else resting.get("orders", [])
                if any(o.get("ticker") == rec.ticker for o in resting_orders):
                    logger.info(
                        f"SKIP_DUPLICATE {rec.ticker}: resting order already exists — skipping to prevent duplicate"
                    )
                    return
            except Exception as e:
                logger.warning(f"Could not check resting orders before trade ({rec.ticker}): {e}")

            order_resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.kalshi.place_order(
                    ticker=rec.ticker,
                    side=rec.side,
                    count=rec.contracts,
                    price=rec.market_price,
                ),
            )
            order = order_resp.get("order", {})
            order_id = order.get("order_id", "unknown")

            self.tracker.record_trade(
                ticker=rec.ticker,
                order_id=order_id,
                side=rec.side,
                contracts=rec.contracts,
                entry_price=rec.market_price,
                cost_dollars=rec.cost_dollars,
                city=rec.city,
                market_type=rec.market_type,
                model_uncertainty=getattr(rec, "model_uncertainty", 0.3),
                entry_context=getattr(rec, "entry_context", None),
            )

            logger.info(
                f"Trade placed: {rec.ticker} {rec.side.upper()} x{rec.contracts} "
                f"@ {rec.market_price}¢ | order_id={order_id}"
            )

            if self.bot:
                await self.bot.send_trade_alert(rec)

        except Exception as e:
            logger.error(f"Failed to execute trade {rec.ticker}: {e}")
            if self.bot:
                await self.bot.send_alert(f"Trade failed: {rec.ticker} — {e}")

    # -------------------------------------------------------------------------
    # End-of-day close
    # -------------------------------------------------------------------------

    async def _end_of_day_close_job(self):
        """
        Records day history, analyzes the day, records settlement errors,
        runs the structured advisor, sends EOD report and learning analysis.
        """
        logger.info("Running end-of-day close job...")

        # Record and analyze the day
        try:
            self.history.record_day(self.tracker.state)
        except Exception as e:
            logger.error(f"Error recording day history: {e}")

        day_analysis = {}
        try:
            day_analysis = self.history.analyze_day(self.tracker.state)
        except Exception as e:
            logger.error(f"Error analyzing day: {e}")

        # Alert immediately on any P0 findings before anything else
        p0_count = day_analysis.get("p0_count", 0)
        if p0_count > 0 and self.bot:
            try:
                await self.bot.send_alert(
                    f"[P0 ALERT] {p0_count} P0 finding(s) detected in today's trades. "
                    f"Review structured_lessons in learning_log.json immediately."
                )
            except Exception as e:
                logger.error(f"Error sending P0 alert: {e}")

        # Record settlement errors
        try:
            await self._record_settlement_errors()
        except Exception as e:
            logger.error(f"Error recording settlement errors: {e}")

        # Run structured advisor (data-driven — no external LLM call)
        try:
            advisor_recs = await asyncio.get_event_loop().run_in_executor(
                None, run_advisor_session
            )
            # Alert on any P0 advisor findings requiring immediate action
            p0_recs = [r for r in advisor_recs if r.severity == "P0"]
            if p0_recs and self.bot:
                for rec in p0_recs[:3]:  # cap at 3 alerts
                    await self.bot.send_alert(
                        f"[ADVISOR P0] {rec.title}\n"
                        f"Action: {rec.recommended_action}\n"
                        f"Auto-apply: {rec.auto_apply_allowed}\n"
                        f"{rec.proposed_change_summary[:300]}"
                    )
        except Exception as e:
            logger.error(f"Error running advisor session: {e}")

        # Send EOD report
        try:
            if self.bot:
                await self.bot.send_eod_report(self.tracker.state)
        except Exception as e:
            logger.error(f"Error sending EOD report: {e}")

        # Send learning analysis
        try:
            if self.bot and day_analysis:
                insights = self.history.get_insights()
                await self.bot.send_learning_analysis(insights, self.tracker.state)
        except Exception as e:
            logger.error(f"Error sending learning analysis: {e}")

    async def _record_settlement_errors(self):
        """
        Fetches NWS actual observations for each city and compares to yesterday's
        forecast. Saves errors to data/forecast_errors.json.
        """
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        season = _get_season_for_date(date.today().isoformat())

        # Load existing errors
        try:
            with open(ERRORS_FILE) as f:
                data = json.load(f)
        except Exception:
            data = {"forecast_errors": [], "calibration": {}}

        errors = data.get("forecast_errors", [])
        calibration = data.get("calibration", {})

        # Deduplicate: remove any existing entries for today to avoid double-writing
        today_iso = date.today().isoformat()
        errors = [e for e in errors if e.get("date") != today_iso]

        http = httpx.Client(
            timeout=15.0,
            headers={"User-Agent": "TradeTheWeather/1.0 contact@tradetheweather.local"},
        )

        for city, station in CITY_STATIONS.items():
            try:
                url = f"{NWS_BASE}/stations/{station}/observations"
                resp = http.get(url, params={"limit": 24})
                resp.raise_for_status()
                features = resp.json().get("features", [])

                if not features:
                    continue

                # Extract temperature readings from the last 24 observations
                temps_f = []
                for feat in features:
                    props = feat.get("properties", {})
                    temp_c = props.get("temperature", {}).get("value")
                    if temp_c is not None:
                        temps_f.append(round(temp_c * 9 / 5 + 32, 1))

                if not temps_f:
                    continue

                actual_high = max(temps_f)
                actual_low = min(temps_f)

                # Get yesterday's NWS forecast for this city (from weather pipeline)
                # Use the city in US_CITIES if available
                from src.weather.pipeline import US_CITIES
                if city not in US_CITIES:
                    continue

                forecast = self.weather.get_forecast(city)
                if not forecast or "error" in forecast:
                    continue

                forecast_high = forecast.get("high_temp_f")
                forecast_low = forecast.get("low_temp_f")

                # Record high temp error
                if forecast_high is not None:
                    error = forecast_high - actual_high   # positive = ran too warm
                    record = {
                        "city": city,
                        "date": today_iso,
                        "market_type": "temp_high",
                        "forecast_value": forecast_high,
                        "actual_value": actual_high,
                        "error": round(error, 1),
                        "season": season,
                    }
                    errors.append(record)
                    logger.info(
                        f"Settlement error recorded: {city} temp_high "
                        f"forecast={forecast_high}°F actual={actual_high}°F "
                        f"error={error:+.1f}°F"
                    )

                # Record low temp error
                if forecast_low is not None:
                    error = forecast_low - actual_low
                    record = {
                        "city": city,
                        "date": today_iso,
                        "market_type": "temp_low",
                        "forecast_value": forecast_low,
                        "actual_value": actual_low,
                        "error": round(error, 1),
                        "season": season,
                    }
                    errors.append(record)
                    logger.info(
                        f"Settlement error recorded: {city} temp_low "
                        f"forecast={forecast_low}°F actual={actual_low}°F "
                        f"error={error:+.1f}°F"
                    )

            except Exception as e:
                logger.warning(f"Settlement error for {city}/{station}: {e}")
                continue

        http.close()

        # Trim to last 90 days to keep file manageable
        errors = errors[-90 * len(CITY_STATIONS):]

        # Update calibration summary
        city_type_errors: dict = {}
        for e in errors:
            key = f"{e['city']}/{e['market_type']}"
            city_type_errors.setdefault(key, []).append(e["error"])
        calibration = {
            k: {"mean": round(sum(v) / len(v), 2), "n": len(v)}
            for k, v in city_type_errors.items()
        }

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(ERRORS_FILE, "w") as f:
            json.dump({"forecast_errors": errors, "calibration": calibration}, f, indent=2)

    # -------------------------------------------------------------------------
    # Uncertainty recalibration (weekly)
    # -------------------------------------------------------------------------

    async def _uncertainty_recalibration_job(self):
        """
        Runs the rolling uncertainty recalibration and queues proposals into
        pending_approvals.json. Does NOT apply any changes automatically.
        """
        logger.info("Running weekly uncertainty recalibration session...")
        try:
            new_recs = await asyncio.get_event_loop().run_in_executor(
                None, run_recalibration_session
            )
            if new_recs and self.bot:
                await self.bot.send_alert(
                    f"[RECALIBRATION] {len(new_recs)} uncertainty update proposal(s) queued "
                    f"in pending_approvals.json. Review before applying."
                )
            logger.info(
                f"Uncertainty recalibration complete: {len(new_recs)} proposals queued"
            )
        except Exception as e:
            logger.error(f"Uncertainty recalibration error: {e}")

    # -------------------------------------------------------------------------
    # Midnight reset
    # -------------------------------------------------------------------------

    async def _midnight_reset_job(self):
        """Resets the daily state for a fresh trading day."""
        logger.info("Midnight reset: starting fresh day...")
        try:
            # Preserve stall counts for positions that are still open so multi-day
            # stalled positions continue to escalate rather than resetting to 0 (P1-3 fix).
            old_stall_counts = dict(self.tracker.state.stall_alert_counts)
            open_tickers = {p.ticker for p in self.tracker.state.positions}

            self.tracker.state = self.tracker._load_or_init()

            carried = {t: c for t, c in old_stall_counts.items() if t in open_tickers}
            if carried:
                self.tracker.state.stall_alert_counts.update(carried)
                logger.info(f"Midnight reset: carried stall counts for {list(carried.keys())}")

            logger.info(f"New day state initialized. Balance: ${self.tracker.state.starting_balance:.2f}")
        except Exception as e:
            logger.error(f"Midnight reset error: {e}")

    # -------------------------------------------------------------------------
    # Morning briefing
    # -------------------------------------------------------------------------

    async def _morning_briefing_job(self):
        """Sends morning briefing and applies history insights to analysis engine."""
        logger.info("Sending morning briefing...")
        try:
            # Refresh balance from Kalshi before reporting
            self.tracker.refresh_balance()

            # Build context for briefing
            context = {
                "balance": self.tracker.state.current_balance,
                "date": date.today().isoformat(),
                "open_markets": 0,  # populated if we do a quick scan
            }
            if self.bot:
                await self.bot.send_morning_briefing(context)

            # Load and apply history insights (uses confidence-weighted penalties)
            insights = self.history.get_insights()
            if hasattr(self.analysis, "apply_history_insights"):
                self.analysis.apply_history_insights(insights)
                logger.info(
                    f"Applied history insights: win_rate_7d={insights.win_rate_7d:.0%}, "
                    f"avoid_cities={insights.avoid_cities}, "
                    f"raise_edge_cities={insights.raise_edge_cities}, "
                    f"open_p0_count={insights.open_p0_count}"
                )
        except Exception as e:
            logger.error(f"Morning briefing error: {e}")

    # -------------------------------------------------------------------------
    # Settlement backfill
    # -------------------------------------------------------------------------

    async def _settlement_backfill_job(self):
        """
        Backfills settlement outcomes for positions older than 2 days
        that still have UNKNOWN status.

        Also backfills shadow_mode_log.json and missed_opportunities.json
        so the validation framework and opportunity tracker have resolved outcomes.
        """
        logger.info("Running settlement backfill...")
        settled_markets: dict = {}  # ticker -> winning side ("yes"|"no"), for shadow backfill
        try:
            history = self.history._load_history()
            cutoff = (date.today() - timedelta(days=2)).isoformat()
            updated = 0

            for record in history:
                if record.get("date", "") > cutoff:
                    continue  # too recent
                for pos in record.get("positions", []):
                    if pos.get("status") == "UNKNOWN":
                        # Try to look up fill status
                        ticker = pos.get("ticker", "")
                        try:
                            fills = self.kalshi.get_fills(limit=100)
                            matching = [f for f in fills if f.get("ticker") == ticker]
                            if matching:
                                pos["status"] = "closed"
                                pos["exit_price"] = matching[0].get("yes_price")
                                updated += 1
                        except Exception:
                            pass
                    # Collect settled tickers for shadow backfill.
                    # Only include positions with a decisive outcome (non-zero PnL).
                    # Scratch trades (pnl==0) have an ambiguous outcome and are excluded.
                    if pos.get("status") in ("closed", "expired"):
                        ticker = pos.get("ticker", "")
                        pnl = pos.get("pnl_dollars", 0.0)
                        side = pos.get("side", "")
                        if ticker and side and pnl != 0.0:
                            # Winning side = the side the bot held if it won, opposite if it lost.
                            won = pnl > 0
                            winning_side = side if won else ("no" if side == "yes" else "yes")
                            settled_markets[ticker] = winning_side

            if updated > 0:
                self.history._save_history(history)
                logger.info(f"Settlement backfill: updated {updated} positions")
            else:
                logger.info("Settlement backfill: no UNKNOWN positions to update")

            # Backfill shadow mode log outcomes
            if settled_markets:
                try:
                    from src.analysis.validation import backfill_shadow_outcomes
                    shadow_updated = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: backfill_shadow_outcomes(settled_markets),
                    )
                    if shadow_updated:
                        logger.info(f"Settlement backfill: updated {shadow_updated} shadow log entries")
                except Exception as e:
                    logger.warning(f"Shadow backfill error: {e}")

                # Backfill missed opportunities log
                try:
                    self.history.backfill_missed_opportunity_outcomes(settled_markets)
                except Exception as e:
                    logger.warning(f"Missed opportunity backfill error: {e}")

        except Exception as e:
            logger.error(f"Settlement backfill error: {e}")

    # -------------------------------------------------------------------------
    # Reconciliation
    # -------------------------------------------------------------------------

    async def _reconcile_exit(
        self,
        ticker: str,
        side: str,
        contracts: int,
        expected_reduction: int,
    ) -> bool:
        """
        Checks that the Kalshi position actually decreased after a sell order.
        Returns True if reconciled (position reduced), False if not.
        """
        try:
            # Brief delay so Kalshi's positions endpoint reflects the fill before we query.
            # Without this, an "executed" order may not appear settled yet (race condition).
            await asyncio.sleep(2)
            positions = await asyncio.get_event_loop().run_in_executor(
                None, self.kalshi.get_positions
            )
            for pos in positions:
                if pos.get("ticker") == ticker:
                    current_count = pos.get("position", 0)
                    logger.debug(f"Reconcile {ticker}: current_position={current_count}, expected_reduction={expected_reduction}")
                    if current_count > contracts - expected_reduction:
                        logger.warning(
                            f"Reconcile {ticker}: position={current_count} contracts but expected "
                            f"reduction of {expected_reduction} from {contracts} — exit may not have processed"
                        )
                        return False
                    return True

            # Position not found — fully closed, which is correct for a full exit
            logger.debug(f"Reconcile {ticker}: position not found (fully closed)")
            return True
        except Exception as e:
            logger.error(f"Reconcile exit error for {ticker}: {e}")
            return False

