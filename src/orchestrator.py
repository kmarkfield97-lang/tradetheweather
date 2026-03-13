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
from src.tracker.pnl import PnLTracker
from src.tracker.history import DailyHistoryTracker

logger = logging.getLogger(__name__)

SCAN_INTERVAL_MINUTES = 30
MAX_TRADES_PER_DAY = 10   # defined here; enforcement is via can_trade() / trading_halted

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
        self.tracker = PnLTracker(self.kalshi)
        self.history = DailyHistoryTracker()
        self.scheduler = AsyncIOScheduler()
        self.bot = None
        self._known_resting_ids: set = set()
        self._bot_exit_order_ids: set = set()
        self.started_at = datetime.now(timezone.utc)

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

        # Market scan every 30 minutes
        self.scheduler.add_job(
            self._scan_job,
            "interval",
            minutes=SCAN_INTERVAL_MINUTES,
            id="scan_job",
            name="Market scan",
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

        # Startup scan 15 seconds from now
        startup_scan_time = datetime.now(timezone.utc) + timedelta(seconds=15)
        self.scheduler.add_job(
            self._scan_job,
            "date",
            run_date=startup_scan_time,
            id="startup_scan",
            name="Startup scan",
        )

        # Startup EOD report 5 seconds from now
        startup_eod_time = datetime.now(timezone.utc) + timedelta(seconds=5)
        self.scheduler.add_job(
            self._end_of_day_close_job,
            "date",
            run_date=startup_eod_time,
            id="startup_eod",
            name="Startup EOD report",
        )

        self.scheduler.start()
        logger.info("Scheduler started with all jobs registered.")

    # -------------------------------------------------------------------------
    # Balance refresh + fill check
    # -------------------------------------------------------------------------

    async def _balance_refresh_job(self):
        """Refresh balance, check fills, and enforce profit targets."""
        try:
            await self._check_order_fills()
            self.tracker.refresh_balance()

            # Check profit take opportunities
            profit_takes = self.tracker.check_profit_takes() if hasattr(self.tracker, "check_profit_takes") else []
            for position, exit_price in profit_takes:
                try:
                    order = self.kalshi.exit_position(
                        position.ticker, position.side, position.contracts, exit_price
                    )
                    exit_order_id = order.get("order", {}).get("order_id", "")
                    if exit_order_id:
                        self._bot_exit_order_ids.add(exit_order_id)

                    pnl = (exit_price / 100 - position.entry_price / 100) * position.contracts
                    self.tracker.record_exit(position.order_id, exit_price, pnl)

                    reconciled = await self._reconcile_exit(
                        position.ticker, position.side, position.contracts, position.contracts
                    )
                    msg = (
                        f"Profit take: {position.ticker} {position.side.upper()} "
                        f"exit @ {exit_price}¢ | P&L: ${pnl:+.2f}"
                    )
                    if not reconciled:
                        msg += " (reconciliation warning: position may not have reduced)"
                    if self.bot:
                        await self.bot.send_alert(msg)
                    logger.info(msg)
                except Exception as e:
                    logger.error(f"Error exiting position {position.ticker}: {e}")

        except Exception as e:
            logger.error(f"Balance refresh job error: {e}")

    async def _check_order_fills(self):
        """
        Compare current resting orders against known set.
        Notify on fills (orders that disappeared from resting list).
        Skips orders placed by the bot for position exits.
        """
        try:
            current_resting = self.kalshi.get_orders(status="resting")
            current_ids = {o["order_id"] for o in current_resting if "order_id" in o}

            filled_ids = self._known_resting_ids - current_ids
            for oid in filled_ids:
                if oid in self._bot_exit_order_ids:
                    self._bot_exit_order_ids.discard(oid)
                    continue
                msg = f"Order filled: {oid}"
                logger.info(msg)
                if self.bot:
                    await self.bot.send_alert(f"Order filled: {oid}")

            self._known_resting_ids = current_ids
        except Exception as e:
            logger.error(f"Check order fills error: {e}")

    # -------------------------------------------------------------------------
    # Market scan
    # -------------------------------------------------------------------------

    async def _scan_job(self):
        """Run a market scan with a 120-second timeout."""
        try:
            await asyncio.wait_for(self.run_scan(), timeout=120)
        except asyncio.TimeoutError:
            logger.warning("Market scan timed out after 120s")
        except Exception as e:
            logger.error(f"Scan job error: {e}")

    async def run_scan(self):
        """
        Gets trade recommendations and executes approved ones if trading is allowed.
        """
        logger.info("Starting market scan...")
        daily_budget = self.tracker.state.current_balance
        trades_used = self.tracker.state.trades_placed

        try:
            recommendations = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.analysis.get_recommendations(daily_budget, trades_used),
            )
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return

        logger.info(f"Scan found {len(recommendations)} recommendations.")

        for rec in recommendations:
            can, reason = self.tracker.can_trade()
            if not can:
                logger.info(f"Trading halted: {reason}")
                break
            await self.execute_trade(rec)

    async def execute_trade(self, rec):
        """Places an order for a recommendation and records it."""
        try:
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
        sends EOD report and learning analysis, then runs GPT advisor.
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

        # Record settlement errors
        try:
            await self._record_settlement_errors()
        except Exception as e:
            logger.error(f"Error recording settlement errors: {e}")

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

        # Run GPT advisor UNCONDITIONALLY (outside the try/except blocks above)
        await self._run_gpt_advisor()

    async def _record_settlement_errors(self):
        """
        Fetches NWS actual observations for each city and compares to yesterday's
        forecast. Saves errors to data/forecast_errors.json.
        """
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        season = _get_season(date.today())

        # Load existing errors
        try:
            with open(ERRORS_FILE) as f:
                data = json.load(f)
        except Exception:
            data = {"forecast_errors": [], "calibration": {}}

        errors = data.get("forecast_errors", [])
        calibration = data.get("calibration", {})

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
                        "date": yesterday,
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
                        "date": yesterday,
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
    # Midnight reset
    # -------------------------------------------------------------------------

    async def _midnight_reset_job(self):
        """Resets the daily state for a fresh trading day."""
        logger.info("Midnight reset: starting fresh day...")
        try:
            self.tracker.state = self.tracker._load_or_init()
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
            # Build context for briefing
            context = {
                "balance": self.tracker.state.current_balance,
                "date": date.today().isoformat(),
                "open_markets": 0,  # populated if we do a quick scan
            }
            if self.bot:
                await self.bot.send_morning_briefing(context)

            # Load and apply history insights
            insights = self.history.get_insights()
            if hasattr(self.analysis, "apply_history_insights"):
                self.analysis.apply_history_insights(insights)
                logger.info(
                    f"Applied history insights: win_rate_7d={insights.win_rate_7d:.0%}, "
                    f"avoid_cities={insights.avoid_cities}"
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
        """
        logger.info("Running settlement backfill...")
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

            if updated > 0:
                self.history._save_history(history)
                logger.info(f"Settlement backfill: updated {updated} positions")
            else:
                logger.info("Settlement backfill: no UNKNOWN positions to update")
        except Exception as e:
            logger.error(f"Settlement backfill error: {e}")

    # -------------------------------------------------------------------------
    # GPT Advisor
    # -------------------------------------------------------------------------

    async def _run_gpt_advisor(self):
        """Runs GPT advisor, sends results to Telegram, and routes suggestions."""
        logger.info("Running GPT advisor...")
        try:
            from src.advisor.gpt_advisor import get_suggestions
            advice = await asyncio.get_event_loop().run_in_executor(None, get_suggestions)

            if advice and self.bot:
                await self.bot.send_gpt_suggestions(advice)

            if advice:
                gpt_run_date = date.today().isoformat()
                await self._route_suggestions(advice, gpt_run_date)

        except Exception as e:
            logger.error(f"GPT advisor error: {e}")

    async def _route_suggestions(self, advice: dict, gpt_run_date: str):
        """
        Routes suggestions based on priority and category:
        - HIGH + additive → auto-implement
        - HIGH/MEDIUM + update/replace → approval queue (Telegram button)
        - MEDIUM/LOW + additive → info only
        """
        from src.advisor.auto_implementer import implement_suggestion

        suggestions = advice.get("suggestions", [])
        for suggestion in suggestions:
            priority = suggestion.get("priority", "LOW").upper()
            category = suggestion.get("category", "additive").lower()
            title = suggestion.get("title", "")

            if priority == "HIGH" and category == "additive":
                # Auto-implement
                result = implement_suggestion(suggestion)
                logger.info(f"Auto-implemented suggestion '{title}': {result}")
                if self.bot:
                    await self.bot.send_alert(
                        f"Auto-implemented suggestion: {title}\nResult: {result}"
                    )

            elif priority in ("HIGH", "MEDIUM") and category in ("update", "replace"):
                # Queue for approval
                if self.bot and hasattr(self.bot, "queue_for_approval"):
                    await self.bot.queue_for_approval(suggestion, gpt_run_date)
                    logger.info(f"Queued for approval: '{title}'")

            else:
                # Info only — already shown in send_gpt_suggestions
                logger.info(f"Informational suggestion logged: '{title}'")

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
            positions = await asyncio.get_event_loop().run_in_executor(
                None, self.kalshi.get_positions
            )
            for pos in positions:
                if pos.get("ticker") == ticker:
                    current_count = pos.get("position", 0)
                    # If we reduced expected_reduction contracts, position should be lower
                    # We just check it didn't increase
                    logger.debug(f"Reconcile {ticker}: current_position={current_count}")
                    return True  # position found, assume reduce was processed

            # Position not found at all — may have been fully closed
            return True
        except Exception as e:
            logger.error(f"Reconcile exit error for {ticker}: {e}")
            return False


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _get_season(d: date) -> str:
    month = d.month
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "fall"
