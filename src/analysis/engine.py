"""
Trade analysis engine.
Scores weather markets using NWS forecasts, historical trends, station bias,
temperature trajectory, forecast update shock, orderbook microstructure,
and Kalshi market pricing. Only surfaces high-confidence, liquid opportunities.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.kalshi.client import KalshiClient
from src.weather.pipeline import WeatherPipeline
from src.signals import aggregator as signal_aggregator
from src.signals import (
    station_bias,
    temperature_trajectory,
    forecast_update,
    metar_latency,
    orderbook_microstructure,
    market_implied_prob,
    threshold_clustering,
    position_sizing as position_sizer,
)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

# ── Hard filter constants ──────────────────────────────────────────────────────
MIN_HOURS_TO_CLOSE = 3      # skip markets closing in < 3 hours (unless high liquidity)
MAX_HOURS_TO_CLOSE = 48     # skip markets > 48 hours out — NWS skill is negligible
MAX_SPREAD = 15             # maximum acceptable spread (cents)
MAX_SPREAD_PCT = 0.25       # or 25% of contract price, whichever is more restrictive
MIN_EXECUTABLE_LIQUIDITY = 25.0   # require $25 within 5¢ of mid price
MIN_PRICE_CENTS = 15        # skip markets priced below 15¢ (near-certain outcomes)
MIN_PRICE_CENTS_HIGH_EDGE = 10    # allow down to 10¢ if edge is extremely large (>20¢)
MAX_TRADES_PER_DAY = 999999

# ── Edge thresholds (dynamic by liquidity tier) ────────────────────────────────
MIN_EDGE_HIGH_LIQ = 0.05    # 5¢ — deep orderbook
MIN_EDGE_NORMAL = 0.07      # 7¢ — normal
MIN_EDGE_LOW_LIQ = 0.10     # 10¢ — thin orderbook
EXTREME_DISAGREEMENT_EDGE = 0.15   # if market_implied says extreme disagreement, require 15¢

# ── Position sizing caps ───────────────────────────────────────────────────────
MAX_POSITION_PCT = 0.10     # 10% of daily budget per trade
MAX_DOLLARS_PER_TRADE = 5.0 # hard cap: never risk more than $5 on one trade

# ── Default forecast uncertainty fallback ─────────────────────────────────────
DEFAULT_UNCERTAINTY_F = 7.0

# ── Monte Carlo settings ───────────────────────────────────────────────────────
MC_PATHS = 2000             # number of simulated temperature paths


@dataclass
class TradeRecommendation:
    ticker: str
    market_title: str
    side: str                    # "yes" or "no"
    our_probability: float       # our estimated probability (0–1)
    market_price: int            # kalshi price in cents
    edge: float                  # our_prob - market_implied_prob
    confidence: str              # "high" / "medium" / "low"
    contracts: int               # number of contracts to buy
    cost_dollars: float          # total cost in dollars
    reasoning: str               # human-readable explanation
    city: str
    market_type: str             # "temp_high" / "temp_low" / "rain" / "snow"
    forecast_summary: str
    alerts: list[str] = field(default_factory=list)
    signal_notes: list[str] = field(default_factory=list)


class TradeAnalysisEngine:
    def __init__(self, kalshi: KalshiClient, weather: WeatherPipeline):
        self.kalshi = kalshi
        self.weather = weather
        self._uncertainty_cache: dict = {}   # city:market_type -> sigma_f
        self._avoid_cities: set = set()
        self._city_edge_adjustments: dict = {}
        self._load_uncertainty_cache()

    def _load_uncertainty_cache(self):
        """Load per-city forecast uncertainty from data/city_uncertainty.json."""
        path = os.path.join(DATA_DIR, "city_uncertainty.json")
        try:
            with open(path) as f:
                raw = json.load(f)
            # Keys are "CITY:market_type" or "CITY:market_type:season"
            # Use the non-season keys as the primary lookup
            self._uncertainty_cache = {
                k: float(v) for k, v in raw.items()
                if k.count(":") == 1   # skip season-specific keys
            }
        except Exception:
            self._uncertainty_cache = {}

    def _get_city_uncertainty(self, city: str, market_type: str) -> float:
        """
        Returns forecast sigma (°F) for city+market_type.
        Falls back to DEFAULT_UNCERTAINTY_F if unknown.
        """
        key = f"{city.upper()}:{market_type}"
        return self._uncertainty_cache.get(key, DEFAULT_UNCERTAINTY_F)

    def _get_dynamic_sigma(
        self, city: str, market_type: str, hours_to_close: Optional[float]
    ) -> float:
        """
        Dynamic forecast uncertainty based on time remaining to settlement.

        24h+ remaining → city baseline (typically 6–8°F)
        12–24h         → 80% of baseline
        same-day morning (6–12h) → 60% of baseline
        afternoon (3–6h) → 45% of baseline
        final hours (< 3h) → 30% of baseline (tight)
        """
        base = self._get_city_uncertainty(city, market_type)
        if hours_to_close is None or hours_to_close >= 24:
            return base
        elif hours_to_close >= 12:
            return base * 0.80
        elif hours_to_close >= 6:
            return base * 0.60
        elif hours_to_close >= 3:
            return base * 0.45
        else:
            return max(1.5, base * 0.30)

    def apply_history_insights(self, insights):
        """
        Apply historical performance insights to adjust future trade behavior.
        Called by orchestrator morning briefing.
        """
        if not insights:
            return
        try:
            self._avoid_cities = set(getattr(insights, "avoid_cities", []))
            perf = getattr(insights, "performance_by_city", {})
            self._city_edge_adjustments = {
                city: data for city, data in perf.items()
            }
            logger.info(
                f"History insights applied: avoid_cities={self._avoid_cities}, "
                f"city_adjustments={list(self._city_edge_adjustments.keys())}"
            )
        except Exception as e:
            logger.warning(f"apply_history_insights error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def get_recommendations(
        self,
        daily_budget: float,
        trades_used: int = 0,
        open_position_cost: float = 0.0,
    ) -> list[TradeRecommendation]:
        """
        Scans all open Kalshi weather markets, scores them, and returns
        ranked trade recommendations that fit within the daily rules.
        """
        trades_remaining = MAX_TRADES_PER_DAY - trades_used
        if trades_remaining <= 0:
            return []

        markets = self.kalshi.get_weather_markets()
        recommendations = []

        for market in markets:
            ticker = market.get("ticker", "")
            try:
                rec = self._evaluate_market(market, daily_budget, open_position_cost)
                if rec:
                    recommendations.append(rec)
                else:
                    logger.debug(f"REJECTED {ticker}: no recommendation returned")
            except Exception as e:
                logger.warning(f"REJECTED {ticker}: exception during evaluation — {e}")
                continue

        # Sort by edge descending (best opportunity first)
        recommendations.sort(key=lambda r: r.edge, reverse=True)

        # Deduplicate: only one trade per city+market_type to avoid contradictory positions.
        seen: set = set()
        deduped = []
        for rec in recommendations:
            key = (rec.city, rec.market_type)
            if key not in seen:
                seen.add(key)
                deduped.append(rec)

        return deduped[:trades_remaining]

    # ─────────────────────────────────────────────────────────────────────────
    # Market evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_market(
        self, market: dict, daily_budget: float, open_position_cost: float = 0.0
    ) -> Optional[TradeRecommendation]:
        ticker = market.get("ticker", "")
        title = market.get("title", "")

        # Parse market type and city from ticker/title
        parsed = self._parse_market(ticker, title)
        if not parsed:
            logger.debug(f"REJECTED {ticker}: could not parse market")
            return None
        city, market_type, threshold, is_bucket = parsed

        # Skip cities with bad historical performance
        if city in self._avoid_cities:
            logger.debug(f"REJECTED {ticker}: {city} in avoid_cities list")
            return None

        # ── Time filter ────────────────────────────────────────────────────────
        close_time = market.get("close_time") or market.get("expiration_time")
        hours_left: Optional[float] = None
        if close_time:
            try:
                ct = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                now_utc = datetime.now(timezone.utc)
                hours_left = (ct - now_utc).total_seconds() / 3600

                # Far-future filter: NWS skill negligible beyond 48 hours
                if hours_left > MAX_HOURS_TO_CLOSE:
                    logger.debug(
                        f"REJECTED {ticker}: {hours_left:.1f}h to close > {MAX_HOURS_TO_CLOSE}h max"
                    )
                    return None

                # If market settles today after noon PT (18:00 UTC) the high has occurred
                if ct.date() == now_utc.date() and now_utc.hour >= 18:
                    logger.debug(f"REJECTED {ticker}: same-day market after 18:00 UTC")
                    return None
            except Exception:
                pass

        # ── Liquidity filter ───────────────────────────────────────────────────
        liquidity = self.kalshi.get_liquidity(ticker)
        if not liquidity["is_liquid"]:
            logger.debug(f"REJECTED {ticker}: illiquid")
            return None

        spread = liquidity.get("spread")
        yes_price = liquidity.get("best_yes_price")
        no_price = liquidity.get("best_no_price")

        if yes_price is not None and no_price is not None:
            mid_price = (yes_price + no_price) / 2.0
        else:
            mid_price = (yes_price if yes_price is not None else no_price if no_price is not None else 50)

        # Spread filter: tighter of absolute or % of mid
        if spread:
            spread_pct_limit = max(1, round(mid_price * MAX_SPREAD_PCT))
            effective_max_spread = min(MAX_SPREAD, spread_pct_limit)
            if spread > effective_max_spread:
                logger.debug(
                    f"REJECTED {ticker}: spread={spread}¢ > limit={effective_max_spread}¢"
                )
                return None

        # Executable liquidity check (need $25 within 5¢ of mid)
        exec_liq = self._executable_liquidity(liquidity, window_cents=5)
        if exec_liq < MIN_EXECUTABLE_LIQUIDITY:
            # Allow short-time-remaining markets with any liquidity as exception
            if hours_left is None or hours_left > MIN_HOURS_TO_CLOSE:
                logger.debug(
                    f"REJECTED {ticker}: executable_liquidity=${exec_liq:.1f} < ${MIN_EXECUTABLE_LIQUIDITY}"
                )
                return None

        # Near-close liquidity exception: allow < 3h if some liquidity exists
        if hours_left is not None and hours_left < MIN_HOURS_TO_CLOSE:
            if exec_liq < 5.0:
                logger.debug(f"REJECTED {ticker}: {hours_left:.1f}h left and exec_liq=${exec_liq:.1f} too thin")
                return None
            # Otherwise allow (good liquidity near close is a valid late trade)

        # ── Weather data ───────────────────────────────────────────────────────
        report = self.weather.get_full_report(city)
        if not report["forecast"] or "error" in report.get("forecast", {}):
            logger.debug(f"REJECTED {ticker}: weather data error for {city}")
            return None

        # Weather data integrity: check observation freshness
        obs = report.get("recent_observations", [])
        if obs:
            latest_obs_time = obs[0].get("timestamp")
            if latest_obs_time:
                try:
                    obs_dt = datetime.fromisoformat(latest_obs_time.replace("Z", "+00:00"))
                    obs_age_h = (datetime.now(timezone.utc) - obs_dt).total_seconds() / 3600
                    if obs_age_h > 6:
                        logger.warning(
                            f"{ticker}: weather data is {obs_age_h:.1f}h old — reducing confidence"
                        )
                        # Don't reject, but the metar_latency signal will penalise
                except Exception:
                    pass

        # ── Score the market ───────────────────────────────────────────────────
        sigma = self._get_dynamic_sigma(city, market_type, hours_left)
        score_result = self._score_market(
            market_type, threshold, report, city, is_bucket, sigma, hours_left
        )
        if score_result is None:
            return None
        base_prob, reasoning, forecast_summary = score_result

        # ── Determine best side using base probability ─────────────────────────
        if yes_price is None or no_price is None:
            return None
        # Reject zero-priced contracts — can't compute edge or odds
        if yes_price <= 0 or no_price <= 0:
            return None

        market_yes_prob = yes_price / 100.0
        market_no_prob = no_price / 100.0

        yes_edge = base_prob - market_yes_prob
        no_edge = (1 - base_prob) - market_no_prob

        if yes_edge >= no_edge:
            side = "yes"
            price = yes_price
        else:
            side = "no"
            price = no_price

        # ── Collect and aggregate signals ──────────────────────────────────────
        signal_inputs = []

        # Station bias
        try:
            sig = station_bias.compute(report, threshold, market_type)
            signal_inputs.append(sig)
        except Exception as e:
            logger.debug(f"station_bias signal error: {e}")

        # Temperature trajectory (now uses corrected temp_trend sign)
        if market_type in ("temp_high", "temp_low") and threshold is not None:
            try:
                sig = temperature_trajectory.compute(report, threshold, market_type)
                signal_inputs.append(sig)
            except Exception as e:
                logger.debug(f"temp_trajectory signal error: {e}")

        # Forecast update shock
        try:
            sig = forecast_update.compute(report, threshold, market_type)
            signal_inputs.append(sig)
        except Exception as e:
            logger.debug(f"forecast_update signal error: {e}")

        # METAR latency (current station temp vs threshold)
        if market_type in ("temp_high", "temp_low") and threshold is not None:
            try:
                sig = metar_latency.compute(report, threshold, market_type)
                signal_inputs.append(sig)
            except Exception as e:
                logger.debug(f"metar_latency signal error: {e}")

        # Orderbook microstructure
        try:
            sig = orderbook_microstructure.compute(liquidity, base_prob, side)
            signal_inputs.append(sig)
        except Exception as e:
            logger.debug(f"orderbook signal error: {e}")

        # Market implied probability (extreme disagreement guard)
        extreme_disagreement = False
        try:
            mip_sig = market_implied_prob.compute(liquidity, base_prob, side)
            signal_inputs.append(mip_sig)
            # Flag if extreme disagreement — will require wider edge
            if mip_sig.get("confidence", 1.0) <= 0.2:
                extreme_disagreement = True
        except Exception as e:
            logger.debug(f"market_implied_prob signal error: {e}")

        # Threshold clustering (confidence modifier only)
        threshold_cluster_confidence = 1.0
        try:
            tc_sig = threshold_clustering.compute(threshold, market_type)
            threshold_cluster_confidence = tc_sig.get("confidence", 1.0)
            # Don't append as probability adjuster — it's a confidence tier modifier
        except Exception as e:
            logger.debug(f"threshold_clustering signal error: {e}")

        # Aggregate all signals
        agg = signal_aggregator.aggregate(signal_inputs)

        # Apply capped probability adjustment
        adj = max(-0.20, min(0.20, agg.prob_adjustment))
        our_prob = max(0.01, min(0.99, base_prob + adj))

        # Recompute edge with adjusted probability
        if side == "yes":
            edge = our_prob - market_yes_prob
        else:
            edge = (1 - our_prob) - market_no_prob

        # Determine minimum edge threshold for this market
        total_volume = liquidity.get("total_volume", 0)
        if total_volume >= 500:
            min_edge_req = MIN_EDGE_HIGH_LIQ
        elif total_volume >= 100:
            min_edge_req = MIN_EDGE_NORMAL
        else:
            min_edge_req = MIN_EDGE_LOW_LIQ

        # Raise edge requirement if market implied probability is in extreme disagreement
        if extreme_disagreement:
            min_edge_req = max(min_edge_req, EXTREME_DISAGREEMENT_EDGE)

        # Historical performance edge adjustment
        city_perf = self._city_edge_adjustments.get(city)
        if city_perf:
            win_rate = city_perf.get("win_rate", 0.5)
            trade_count = city_perf.get("trades", 0)
            if trade_count >= 10 and win_rate < 0.40:
                # Bad track record — require more edge
                min_edge_req = max(min_edge_req, 0.10)

        if edge < min_edge_req:
            logger.debug(
                f"REJECTED {ticker}: edge={edge:.3f} < min_edge={min_edge_req:.3f} "
                f"(vol={total_volume}, extreme_disagree={extreme_disagreement})"
            )
            return None

        # ── Price floor ────────────────────────────────────────────────────────
        effective_min_price = MIN_PRICE_CENTS
        if edge >= 0.20:
            effective_min_price = MIN_PRICE_CENTS_HIGH_EDGE  # loosen for huge edge
        if price < effective_min_price:
            logger.debug(
                f"REJECTED {ticker}: price={price}¢ < min_price={effective_min_price}¢"
            )
            return None

        # ── Position sizing via position_sizer module ─────────────────────────
        sizing: Optional[dict] = None
        try:
            sizing = position_sizer.compute(
                our_prob=our_prob,
                side=side,
                price=price,
                edge=edge,
                daily_budget=daily_budget,
                open_position_cost=open_position_cost,
                signal_agreement=agg.signal_agreement,
                model_uncertainty=agg.model_uncertainty,
                liquidity=liquidity,
            )
            position_dollars = sizing["position_dollars"]
        except Exception as e:
            logger.warning(f"position_sizer error for {ticker}: {e}")
            # Fallback: inline 1/4 Kelly
            prob_win = our_prob if side == "yes" else (1 - our_prob)
            prob_lose = 1 - prob_win
            odds = (100 - price) / price
            kelly = max(0, (prob_win * odds - prob_lose) / odds)
            position_dollars = max(0.50, min(daily_budget * kelly * 0.25, MAX_DOLLARS_PER_TRADE))

        # Apply absolute caps
        max_position = min(daily_budget * MAX_POSITION_PCT, MAX_DOLLARS_PER_TRADE)
        position_dollars = min(position_dollars, max_position)
        position_dollars = max(0.50, round(position_dollars, 2))

        contracts = math.floor(position_dollars / (price / 100))
        if contracts < 1:
            logger.debug(f"REJECTED {ticker}: position_dollars=${position_dollars:.2f} rounds to 0 contracts")
            return None

        # Hard safety cap
        actual_cost = contracts * (price / 100)
        if actual_cost > MAX_DOLLARS_PER_TRADE:
            contracts = math.floor(MAX_DOLLARS_PER_TRADE / (price / 100))
            actual_cost = contracts * (price / 100)
        if contracts < 1:
            return None

        # ── Confidence tier ────────────────────────────────────────────────────
        # Round-number thresholds get one tier lower
        if edge >= 0.15:
            confidence = "high" if threshold_cluster_confidence >= 0.5 else "medium"
        elif edge >= 0.10:
            confidence = "medium" if threshold_cluster_confidence >= 0.3 else "low"
        else:
            confidence = "low"

        alerts = [a["event"] for a in report.get("alerts", [])]

        # ── Build full reasoning string ─────────────────────────────────────────
        signal_notes = agg.notes or []
        sizing_note = ""
        if sizing is not None:
            try:
                sizing_note = (
                    f" | Kelly={sizing.get('kelly_raw', 0):.3f} "
                    f"agreement={agg.signal_agreement:.0%} "
                    f"uncertainty={agg.model_uncertainty:.2f}"
                )
            except Exception:
                pass

        full_reasoning = (
            f"{reasoning} | sigma={sigma:.1f}°F | "
            f"base_prob={base_prob:.3f} adj={adj:+.3f} final={our_prob:.3f}"
            f"{sizing_note}"
        )

        logger.info(
            f"TRADE_SIGNAL {ticker} | city={city} type={market_type} threshold={threshold} | "
            f"side={side} price={price}¢ edge={edge:.3f} | "
            f"base_prob={base_prob:.3f} signal_adj={adj:+.3f} final_prob={our_prob:.3f} | "
            f"sigma={sigma:.1f}°F active_signals={agg.active_signals} "
            f"agreement={agg.signal_agreement:.0%} | "
            f"contracts={contracts} cost=${actual_cost:.2f} confidence={confidence} | "
            f"signals={signal_notes}"
        )

        return TradeRecommendation(
            ticker=ticker,
            market_title=title,
            side=side,
            our_probability=round(our_prob, 3),
            market_price=price,
            edge=round(edge, 3),
            confidence=confidence,
            contracts=contracts,
            cost_dollars=round(actual_cost, 2),
            reasoning=full_reasoning,
            city=city,
            market_type=market_type,
            forecast_summary=forecast_summary,
            alerts=alerts,
            signal_notes=signal_notes,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Executable liquidity helper
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _executable_liquidity(liquidity: dict, window_cents: int = 5) -> float:
        """
        Estimates dollar liquidity within `window_cents` of mid price.
        Uses total_volume as proxy when detailed orderbook depth is unavailable.
        """
        total_volume = liquidity.get("total_volume", 0)
        spread = liquidity.get("spread") or 0
        # If spread is very tight (< window_cents), almost all volume is executable
        if spread <= window_cents:
            return total_volume
        # Otherwise, estimate a fraction based on how much spread eats into the window
        depth_fraction = max(0.1, window_cents / max(spread, 1))
        return total_volume * depth_fraction

    # ─────────────────────────────────────────────────────────────────────────
    # Market parser
    # ─────────────────────────────────────────────────────────────────────────

    SERIES_CITY_MAP = {
        "KXHIGHTPHX": "PHOENIX", "KXHIGHTHOU": "HOUSTON", "KXHIGHTMIN": "MINNEAPOLIS",
        "KXHIGHTDAL": "DALLAS", "KXHIGHTLV": "LAS_VEGAS", "KXHIGHTSATX": "SAN_ANTONIO",
        "KXHIGHTBOS": "BOSTON", "KXHIGHTNOLA": "NEW_ORLEANS", "KXHIGHTSFO": "SAN_FRANCISCO",
        "KXHIGHTSEA": "SEATTLE", "KXHIGHTDC": "DC",
        "KXHIGHTATL": "ATLANTA", "KXHIGHTOKC": "OKLAHOMA_CITY",
        "KXLOWTCHI": "CHICAGO", "KXLOWTDEN": "DENVER", "KXLOWTNYC": "NYC",
        "KXLOWTPHIL": "PHILADELPHIA", "KXLOWTMIA": "MIAMI", "KXLOWTLAX": "LOS_ANGELES",
        "KXLOWTAUS": "AUSTIN",
        "KXRAINNYC": "NYC", "KXRAINHOU": "HOUSTON", "KXRAINCHIM": "CHICAGO",
        "KXRAINSEA": "SEATTLE",
    }

    def _parse_market(self, ticker: str, title: str) -> Optional[tuple]:
        """
        Attempts to extract (city, market_type, threshold, is_bucket) from ticker/title.
        Returns None if we can't parse it confidently.
        Threshold is returned as float to preserve precision (e.g. 67.5°F).
        """
        import re
        ticker_upper = ticker.upper()
        title_lower = title.lower()

        city = None
        for prefix, mapped_city in self.SERIES_CITY_MAP.items():
            if ticker_upper.startswith(prefix):
                city = mapped_city
                break

        if not city:
            for c in ["NYC", "LOS_ANGELES", "CHICAGO", "HOUSTON", "PHOENIX",
                       "PHILADELPHIA", "SAN_ANTONIO", "SAN_DIEGO", "DALLAS",
                       "MIAMI", "ATLANTA", "BOSTON", "SEATTLE", "DENVER",
                       "MINNEAPOLIS", "NEW_ORLEANS", "LAS_VEGAS", "KANSAS_CITY",
                       "CLEVELAND", "NASHVILLE"]:
                if c.replace("_", "") in ticker_upper or c.replace("_", " ").lower() in title_lower:
                    city = c
                    break
        if not city:
            city_aliases = {
                "new york": "NYC", "los angeles": "LOS_ANGELES",
                "chicago": "CHICAGO", "atlanta": "ATLANTA", "boston": "BOSTON",
                "seattle": "SEATTLE", "denver": "DENVER", "miami": "MIAMI",
                "houston": "HOUSTON", "phoenix": "PHOENIX", "dallas": "DALLAS",
                "minneapolis": "MINNEAPOLIS", "philadelphia": "PHILADELPHIA",
                "san antonio": "SAN_ANTONIO", "new orleans": "NEW_ORLEANS",
                "las vegas": "LAS_VEGAS",
            }
            for alias, mapped in city_aliases.items():
                if alias in title_lower:
                    city = mapped
                    break
        if not city:
            return None

        market_type = None
        series_prefix = next((p for p in self.SERIES_CITY_MAP if ticker_upper.startswith(p)), "")
        if "HIGH" in series_prefix or "HIGHT" in series_prefix:
            market_type = "temp_high"
        elif "LOW" in series_prefix or "LOWT" in series_prefix:
            market_type = "temp_low"
        elif "RAIN" in series_prefix:
            market_type = "rain"
        elif "SNOW" in series_prefix:
            market_type = "snow"
        elif any(kw in title_lower for kw in ["high temp", "high temperature", "maximum temperature", "max temp", "maximum temp"]):
            market_type = "temp_high"
        elif any(kw in title_lower for kw in ["low temp", "low temperature", "lowest temperature", "minimum temp"]):
            market_type = "temp_low"
        elif any(kw in title_lower for kw in ["rain", "precipitation", "precip"]):
            market_type = "rain"
        elif any(kw in title_lower for kw in ["snow", "snowfall", "snowstorm"]):
            market_type = "snow"
        else:
            return None

        is_bucket = False
        threshold: Optional[float] = None

        # Use float to preserve precision (e.g. 67.5°F bucket thresholds)
        ticker_thresh = re.search(r"-([TB])([\d.]+)(?:-|$)", ticker_upper)
        if ticker_thresh:
            is_bucket = (ticker_thresh.group(1) == "B")
            threshold = float(ticker_thresh.group(2))
        else:
            m = re.search(r"(\d+)\s*°", title)
            if m:
                threshold = float(m.group(1))
            else:
                nums = re.findall(r"\b(\d{2,3})\b", title)
                if nums:
                    threshold = float(nums[0])

        return city, market_type, threshold, is_bucket

    # ─────────────────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _score_market(
        self,
        market_type: str,
        threshold: Optional[float],
        report: dict,
        city: str = "",
        is_bucket: bool = False,
        sigma: float = DEFAULT_UNCERTAINTY_F,
        hours_to_close: Optional[float] = None,  # reserved for future time-decay logic
    ) -> Optional[tuple[float, str, str]]:
        """
        Returns (probability, reasoning, forecast_summary) or None if market type unsupported.
        probability is our estimated P(YES) for the market.
        """
        _ = hours_to_close  # accepted for future use; sigma already incorporates time
        forecast = report.get("forecast", {})
        hourly = report.get("hourly", [])
        moon = report.get("moon", {})
        temp_trend = report.get("temp_trend")
        alerts = report.get("alerts", [])

        high_temp = forecast.get("high_temp_f")
        low_temp = forecast.get("low_temp_f")
        precip_chance = forecast.get("precip_chance") or 0
        short_fc = forecast.get("short_forecast", "")
        detailed_fc = forecast.get("detailed_forecast", "")

        forecast_summary = f"{short_fc} | High: {high_temp}°F Low: {low_temp}°F | Precip: {precip_chance}%"

        if market_type == "temp_high" and threshold is not None and high_temp is not None:
            # Temperature trajectory adjustment (replaces ad-hoc trend_adj)
            trend_adj = 0.0
            if temp_trend is not None:
                # Modest nudge: 1°F per 2 hrs of trend
                trend_adj = temp_trend * 2.0
                trend_adj = max(-3.0, min(3.0, trend_adj))

            corrected_high = high_temp + trend_adj

            if is_bucket:
                # Bucket market: P = CDF(threshold+2) - CDF(threshold)
                p_above_lo = self._sigmoid((corrected_high - threshold) / sigma)
                p_above_hi = self._sigmoid((corrected_high - (threshold + 2.0)) / sigma)
                prob = p_above_lo - p_above_hi
            else:
                # Monte Carlo for threshold markets when sigma and hours are known
                prob = self._monte_carlo_prob(corrected_high, threshold, sigma, "above")

            trend_str = f"{temp_trend:+.3f}" if temp_trend is not None else "N/A"
            reasoning = (
                f"NWS high {high_temp}°F + trend_adj {trend_adj:+.1f}°F = {corrected_high:.1f}°F "
                f"vs threshold {threshold}°F (gap: {corrected_high - threshold:+.1f}°F). "
                f"{'Bucket market. ' if is_bucket else ''}"
                f"Dynamic sigma={sigma:.1f}°F. "
                f"Trend: {trend_str}°F/hr."
            )
            return prob, reasoning, forecast_summary

        elif market_type == "temp_low" and threshold is not None and low_temp is not None:
            trend_adj = 0.0
            if temp_trend is not None:
                trend_adj = temp_trend * 2.0
                trend_adj = max(-3.0, min(3.0, trend_adj))

            corrected_low = low_temp + trend_adj

            if is_bucket:
                p_above_lo = self._sigmoid((corrected_low - threshold) / sigma)
                p_above_hi = self._sigmoid((corrected_low - (threshold + 2.0)) / sigma)
                prob = p_above_lo - p_above_hi
            else:
                prob = self._monte_carlo_prob(corrected_low, threshold, sigma, "above")

            trend_str = f"{temp_trend:+.3f}" if temp_trend is not None else "N/A"
            reasoning = (
                f"NWS low {low_temp}°F + trend_adj {trend_adj:+.1f}°F = {corrected_low:.1f}°F "
                f"vs threshold {threshold}°F (gap: {corrected_low - threshold:+.1f}°F). "
                f"{'Bucket market. ' if is_bucket else ''}"
                f"Dynamic sigma={sigma:.1f}°F. "
                f"Trend: {trend_str}°F/hr."
            )
            return prob, reasoning, forecast_summary

        elif market_type == "rain":
            prob, reasoning = self._score_rain(precip_chance, hourly, moon, alerts, short_fc)
            return prob, reasoning, forecast_summary

        elif market_type == "snow":
            prob, reasoning = self._score_snow(
                precip_chance, hourly, detailed_fc, alerts, moon, low_temp
            )
            return prob, reasoning, forecast_summary

        return None

    def _score_rain(
        self,
        precip_chance: float,
        hourly: list,
        moon: dict,
        alerts: list,
        short_fc: str,
    ) -> tuple[float, str]:
        """
        Improved rain probability using hourly independence model.
        P(any rain) = 1 - product(1 - P_hour_i) across settlement window hours.
        """
        moon_phase = moon.get("phase_name", "")

        # Hourly independence model: P(any rain in window) = 1 - prod(1 - p_i)
        if hourly:
            hourly_probs = [
                h.get("precip_chance", 0) / 100.0
                for h in hourly[:12]  # next 12 hours
                if h.get("precip_chance") is not None
            ]
            if hourly_probs:
                no_rain_product = 1.0
                for p in hourly_probs:
                    no_rain_product *= (1 - p)
                hourly_any_rain = 1.0 - no_rain_product
                # Blend: 60% hourly model, 40% point-in-time NWS forecast
                base_prob = 0.60 * hourly_any_rain + 0.40 * (precip_chance / 100.0)
            else:
                base_prob = precip_chance / 100.0
        else:
            base_prob = precip_chance / 100.0

        # Active rain/flood alerts
        has_rain_alert = any(
            "rain" in a["event"].lower() or "flood" in a["event"].lower()
            for a in alerts
        )
        if has_rain_alert:
            base_prob = min(1.0, base_prob + 0.10)

        prob = min(1.0, max(0.01, base_prob))

        # Moon phase note only (removed probability modifier — no empirical backing)
        reasoning = (
            f"NWS precipitation probability: {precip_chance}%. "
            f"Hourly independence model: {prob:.0%}. "
            f"Moon phase: {moon_phase}. "
            f"{'Active rain/flood alerts. ' if has_rain_alert else ''}"
            f"Forecast: {short_fc}."
        )
        return prob, reasoning

    def _score_snow(
        self,
        precip_chance: float,
        hourly: list,
        detailed_fc: str,
        alerts: list,
        moon: dict,
        low_temp: Optional[float],  # reserved for future below-freezing guard
    ) -> tuple[float, str]:
        """
        Improved snow probability using hourly sub-freezing + precip check.
        """
        _ = low_temp  # used indirectly via hourly temps; kept for API consistency
        snow_keywords = ["snow", "flurr", "blizzard", "winter storm", "sleet", "freezing"]
        has_snow_fc = any(kw in detailed_fc.lower() for kw in snow_keywords)
        snow_alert = any(
            "snow" in a["event"].lower() or "winter" in a["event"].lower()
            for a in alerts
        )

        # Count hours with both precipitation chance and sub-freezing temps
        snow_hours = 0
        total_hours_checked = 0
        for h in hourly[:12]:
            temp = h.get("temp_f")
            precip = h.get("precip_chance", 0) or 0
            fc_text = (h.get("short_forecast") or "").lower()
            total_hours_checked += 1
            if temp is not None and temp <= 34 and precip >= 20:
                snow_hours += 1
            elif any(kw in fc_text for kw in snow_keywords) and precip >= 10:
                snow_hours += 1

        if snow_hours > 0 and total_hours_checked > 0:
            snow_hour_fraction = snow_hours / total_hours_checked
            base_prob = (precip_chance / 100.0) * snow_hour_fraction
        elif has_snow_fc:
            base_prob = precip_chance / 100.0 * 0.5
        else:
            base_prob = 0.05

        if snow_alert:
            base_prob = min(1.0, base_prob + 0.15)

        prob = min(1.0, max(0.01, base_prob))

        reasoning = (
            f"Snow keywords in forecast: {'yes' if has_snow_fc else 'no'}. "
            f"Precip chance: {precip_chance}%. "
            f"Sub-freezing precip hours: {snow_hours}/{total_hours_checked}. "
            f"Winter weather alerts: {'yes' if snow_alert else 'no'}. "
            f"Moon: {moon.get('phase_name', 'unknown')}."
        )
        return prob, reasoning

    # ─────────────────────────────────────────────────────────────────────────
    # Monte Carlo probability estimation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _monte_carlo_prob(
        expected: float,
        threshold: float,
        sigma: float,
        direction: str = "above",
        n_paths: int = MC_PATHS,  # reserved for true MC extension
    ) -> float:
        """
        Monte Carlo estimate of P(final temperature direction threshold).
        Currently uses analytical sigmoid (Gaussian CDF) for speed; n_paths is
        reserved for a future true-MC extension with non-normal path distributions.
        """
        _ = n_paths
        # Analytical Gaussian CDF approximation via sigmoid is equivalent
        # and much faster than drawing random samples.
        # P(X > threshold) where X ~ N(expected, sigma^2)
        z = (expected - threshold) / sigma
        if direction == "above":
            return TradeAnalysisEngine._sigmoid(z)
        else:
            return 1.0 - TradeAnalysisEngine._sigmoid(z)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Logistic sigmoid, returns value in (0, 1)."""
        return 1.0 / (1.0 + math.exp(-x))

