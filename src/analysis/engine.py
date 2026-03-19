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
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

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
MAX_TRADES_PER_DAY = 10  # hard cap: only take the top 10 highest-conviction trades per day

# ── Edge thresholds (dynamic by liquidity tier) ────────────────────────────────
MIN_EDGE_HIGH_LIQ = 0.05    # 5¢ — deep orderbook
MIN_EDGE_NORMAL = 0.07      # 7¢ — normal
MIN_EDGE_LOW_LIQ = 0.10     # 10¢ — thin orderbook
EXTREME_DISAGREEMENT_EDGE = 0.15   # if market_implied says extreme disagreement, require 15¢

# ── Position sizing caps ───────────────────────────────────────────────────────
MAX_POSITION_PCT = 0.10     # 10% of daily budget per trade
MAX_DOLLARS_PER_TRADE = 5.0 # hard cap: never risk more than $5 on one trade

# ── Trim-band entry guard ──────────────────────────────────────────────────────
# Fresh entries at prices already inside a sell trim band generate little
# monetizable edge because sell logic fires almost immediately.
#
# For each band (min¢, max¢, trim_frac):
#   upside_before_trim  = band_min - entry_price  (0 if already inside band)
#   trim_drag_cents     = trim_frac * (entry_price - spread/2)  (approximate cost
#                         of selling trim_frac of the position back near entry)
#   tradable_edge_cents = raw_edge_cents - SLIPPAGE_TOTAL_CENTS - trim_drag_cents
#
# Extra edge hurdles required to allow fresh entry inside or near a trim band.
# These are *additional* cents above the normal min_edge_req.
TRIM_BAND_EXTRA_EDGE_70_79 = 0.07   # 7¢ extra: band 70–79¢ (25% trim fires quickly)
TRIM_BAND_EXTRA_EDGE_80_89 = 0.12   # 12¢ extra: band 80–89¢ (40% trim, rare situations)
TRIM_BAND_EXTRA_EDGE_90_PLUS = 9.99 # effectively reject: 90¢+ band (65–85% trim)

# "Near-band" warning zone: flag entries within this many cents below a trim band.
# These won't be rejected outright but will be logged with a warning.
TRIM_BAND_NEAR_WARN_CENTS = 5

# Total round-trip slippage estimate used in tradable-edge calculation (cents).
# Entry slippage + exit slippage (both half-spread, rounded up).
TRIM_BAND_SLIPPAGE_TOTAL = 4  # cents

# ── Default forecast uncertainty fallback ─────────────────────────────────────
DEFAULT_UNCERTAINTY_F = 7.0

# ── Temp_low calibration bias protection thresholds ───────────────────────────
# Applied per-city when recent forecast_errors show systematic cold bias.
# Bias = forecast_value - actual_value, so strongly negative = forecasts too cold.
#
# WARNING level  (-8°F mean): log and apply modest edge increase (+3¢)
# PENALTY level  (-15°F mean): widen sigma +4°F, increase edge req +5¢,
#                              apply partial bias correction (50% of mean error)
# BLOCK level    (-25°F mean): block temp_low trade for that city entirely
#                              (requires MIN_BLOCK_SAMPLES to avoid 1-sample overreaction)
TEMP_LOW_BIAS_WARN_F   = -8.0    # mean error threshold for warning
TEMP_LOW_BIAS_PENALTY_F = -15.0  # mean error threshold for penalty
TEMP_LOW_BIAS_BLOCK_F   = -25.0  # mean error threshold for blocking
MIN_WARN_SAMPLES    = 2          # minimum samples to issue a warning
MIN_PENALTY_SAMPLES = 3          # minimum samples to apply penalty
MIN_BLOCK_SAMPLES   = 4          # minimum samples to trigger a block
TEMP_LOW_PENALTY_SIGMA_ADD_F = 4.0   # extra sigma added at PENALTY level
TEMP_LOW_PENALTY_EDGE_ADD    = 0.05  # extra edge required (cents as fraction) at PENALTY level
TEMP_LOW_WARN_EDGE_ADD       = 0.03  # extra edge required at WARN level
TEMP_LOW_BIAS_CORRECTION_FRAC = 0.5  # fraction of mean bias to subtract from forecast low

# ── Low-price + large-disagreement combined gate ───────────────────────────────
# A contract priced below LOW_PRICE_GATE_CENTS is "low-price fragile":
#   - high delta variance → a 5¢ move is a 25%+ swing on a 20¢ contract
#   - any disagreement with the market is amplified; errors compound quickly
#
# When the contract is also in large model-vs-market disagreement we treat the
# combination as "dangerous disagreement" rather than "actionable opportunity"
# UNLESS hard confirming evidence is present (see _classify_disagreement).
#
# LOW_PRICE_GATE_CENTS          : price threshold defining "low-price" (¢)
# LOW_PRICE_LARGE_DISAGREE_EDGE : extra edge required when low-price + LARGE_OPP
# LOW_PRICE_EXTREME_DISAGREE_EDGE: extra edge when low-price + EXTREME_OPP
# LOW_PRICE_MAX_DISAGREEMENT    : hard block if disagreement > this and no
#                                 confirming signals (danger zone)
# LOW_PRICE_MIN_SIGNAL_AGREEMENT: minimum signal agreement required for a
#                                 low-price + large-disagreement trade
# LOW_PRICE_MAX_SIGMA_F         : maximum allowed sigma (°F) for low-price trades
LOW_PRICE_GATE_CENTS            = 20    # contracts priced < 20¢ are "low-price fragile"
LOW_PRICE_LARGE_DISAGREE_EDGE   = 0.10  # 10¢ extra above normal min_edge
LOW_PRICE_EXTREME_DISAGREE_EDGE = 0.20  # 20¢ extra — nearly always rejects
LOW_PRICE_MAX_DISAGREEMENT      = 0.35  # hard block above this if disagree=dangerous
LOW_PRICE_MIN_SIGNAL_AGREEMENT  = 0.65  # need 65%+ signal alignment
LOW_PRICE_MAX_SIGMA_F           = 10.0  # reject if model uncertainty > 10°F

# ── Temp_low conservatism under poor calibration ──────────────────────────────
# Beyond the per-level edge surcharges, apply additional gating when the
# combination of calibration quality + sigma is extreme.
#
# TEMP_LOW_EXTREME_SIGMA_F      : sigma above this triggers an extra edge req
# TEMP_LOW_EXTREME_SIGMA_EDGE   : extra edge added when sigma is extreme
# TEMP_LOW_POOR_CALIB_MIN_EDGE  : hard minimum edge for any temp_low with warn+
#                                 calibration (overrides the liquidity-tier minimum)
TEMP_LOW_EXTREME_SIGMA_F        = 10.0  # °F
TEMP_LOW_EXTREME_SIGMA_EDGE     = 0.05  # +5¢ extra when sigma > 10°F
TEMP_LOW_POOR_CALIB_MIN_EDGE    = 0.12  # 12¢ minimum for warn/penalty temp_low

# ── Monte Carlo settings ───────────────────────────────────────────────────────
MC_PATHS = 2000             # number of simulated temperature paths

# ── City timezone map (IANA) ───────────────────────────────────────────────────
CITY_TIMEZONES: dict = {
    "NYC": "America/New_York",
    "PHILADELPHIA": "America/New_York",
    "BOSTON": "America/New_York",
    "DC": "America/New_York",
    "MIAMI": "America/New_York",
    "ATLANTA": "America/New_York",
    "CLEVELAND": "America/New_York",
    "NASHVILLE": "America/Chicago",
    "CHICAGO": "America/Chicago",
    "HOUSTON": "America/Chicago",
    "DALLAS": "America/Chicago",
    "SAN_ANTONIO": "America/Chicago",
    "AUSTIN": "America/Chicago",
    "OKLAHOMA_CITY": "America/Chicago",
    "NEW_ORLEANS": "America/Chicago",
    "KANSAS_CITY": "America/Chicago",
    "MINNEAPOLIS": "America/Chicago",
    "DENVER": "America/Denver",
    "PHOENIX": "America/Phoenix",   # no DST
    "LOS_ANGELES": "America/Los_Angeles",
    "SAN_FRANCISCO": "America/Los_Angeles",
    "SAN_DIEGO": "America/Los_Angeles",
    "LAS_VEGAS": "America/Los_Angeles",
    "SEATTLE": "America/Los_Angeles",
}

# Local hour (0-23) after which temp_high is effectively decided for that city.
# This is the approximate end of the solar heating window.
CITY_HIGH_PEAK_HOUR: dict = {
    "PHOENIX": 19, "LAS_VEGAS": 19, "LOS_ANGELES": 18, "SAN_DIEGO": 18,
    "SAN_FRANCISCO": 17,  # marine influence kills afternoon heating early
    "SEATTLE": 17,
}
DEFAULT_HIGH_PEAK_HOUR = 19  # 7pm local — sensible default for interior cities

# Kalshi weather series prefix → city key mapping (shared with tracker/pnl.py)
SERIES_CITY_MAP: dict = {
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
    model_uncertainty: float = 0.3   # from signal aggregation (used in hold_ev at exit)
    # ── Per-signal breakdown for offline analysis / future weight optimization ──
    signal_breakdown: list[dict] = field(default_factory=list)
    weights_version: str = "fallback"   # version of signal_weights.json active at score time
    # ── Full decision context (populated by _evaluate_market) ────────────────
    # Carries all fields needed to reconstruct the entry decision snapshot.
    # Passed through to record_trade() as entry_context so diagnostics have
    # the full picture of why the trade was entered.
    entry_context: dict = field(default_factory=dict)


class TradeAnalysisEngine:
    def __init__(self, kalshi: KalshiClient, weather: WeatherPipeline):
        self.kalshi = kalshi
        self.weather = weather
        self._uncertainty_cache: dict = {}   # city:market_type -> sigma_f
        self._avoid_cities: set = set()
        self._city_edge_adjustments: dict = {}
        self._load_uncertainty_cache()
        # Lazy-init history tracker reference for missed-opportunity logging
        self._history_tracker = None

    def _get_history_tracker(self):
        if self._history_tracker is None:
            from src.tracker.history import DailyHistoryTracker
            self._history_tracker = DailyHistoryTracker()
        return self._history_tracker

    def _record_missed_opportunity(
        self,
        ticker: str,
        city: str,
        market_type: str,
        rejection_reason: str,
        our_prob: float,
        market_price_cents: int,
        edge_cents: float,
    ):
        """Best-effort logging of rejected candidates for missed-opportunity analysis."""
        try:
            self._get_history_tracker().record_missed_opportunity(
                ticker=ticker,
                city=city,
                market_type=market_type,
                rejection_reason=rejection_reason,
                our_prob=our_prob,
                market_price_cents=market_price_cents,
                edge_cents=edge_cents,
            )
        except Exception as e:
            logger.debug(f"missed_opportunity log failed for {ticker}: {e}")

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

    def _load_forecast_calibration(self) -> dict:
        """
        Load the calibration summary from forecast_errors.json.
        Returns dict keyed by "CITY/market_type" → {"mean": float, "n": int}.
        Returns empty dict on any failure (never raises).
        """
        path = os.path.join(DATA_DIR, "forecast_errors.json")
        try:
            with open(path) as f:
                data = json.load(f)
            return data.get("calibration", {})
        except Exception:
            return {}

    def _get_temp_low_bias_status(
        self, city: str, calibration: dict
    ) -> tuple[str, float, int]:
        """
        Returns (level, mean_error, n_samples) for temp_low calibration of a city.
        level is one of: "ok", "warn", "penalty", "block"
        mean_error is the mean forecast error (forecast − actual); negative = forecast ran cold.
        """
        key = f"{city.upper()}/temp_low"
        entry = calibration.get(key)
        if not entry:
            return "ok", 0.0, 0
        mean_err = float(entry.get("mean", 0.0))
        n = int(entry.get("n", 0))
        if n >= MIN_BLOCK_SAMPLES and mean_err <= TEMP_LOW_BIAS_BLOCK_F:
            return "block", mean_err, n
        if n >= MIN_PENALTY_SAMPLES and mean_err <= TEMP_LOW_BIAS_PENALTY_F:
            return "penalty", mean_err, n
        if n >= MIN_WARN_SAMPLES and mean_err <= TEMP_LOW_BIAS_WARN_F:
            return "warn", mean_err, n
        return "ok", mean_err, n

    def _get_city_uncertainty(self, city: str, market_type: str) -> float:
        """
        Returns forecast sigma (°F) for city+market_type.
        Falls back to DEFAULT_UNCERTAINTY_F if unknown.
        """
        key = f"{city.upper()}:{market_type}"
        return self._uncertainty_cache.get(key, DEFAULT_UNCERTAINTY_F)

    @staticmethod
    def _classify_disagreement(
        disagreement: float,
        market_price_cents: int,
        sigma_f: float,
        signal_agreement: float,
        mip_verdict: str,
        bias_level: str,
        market_type: str,
        forecast_fresh: bool,
    ) -> tuple[str, list[str]]:
        """
        Classifies a model-vs-market disagreement as either "actionable" or "dangerous".

        Actionable disagreement = we likely have a real edge (market is behind our data).
        Dangerous disagreement  = the market is probably right and our model is wrong.

        Returns (classification, reasons) where classification is one of:
          "actionable"  — proceed (subject to normal edge checks)
          "dangerous"   — apply extra scrutiny / higher edge / hard block

        Confirming signals required to call disagreement "actionable":
          1. Fresh forecast (<3h old) — our data is likely newer than market
          2. High signal agreement (≥65%) — multiple independent signals agree
          3. Low uncertainty (sigma ≤ 10°F) — model is confident

        Any of the following push toward "dangerous":
          - Low-price contract (< 20¢)     — high delta variance amplifies errors
          - Extreme sigma (> 10°F)         — model itself is very uncertain
          - Poor calibration (warn+)       — historical forecast errors are severe
          - Very large disagreement (>35%) — burden of proof is on our model
        """
        is_low_price = (market_price_cents < LOW_PRICE_GATE_CENTS)
        is_extreme_sigma = (sigma_f > LOW_PRICE_MAX_SIGMA_F)
        is_poor_calib = (bias_level in ("warn", "penalty", "block"))
        is_very_large_disagree = (disagreement > 0.35)

        danger_flags: list[str] = []
        if is_low_price:
            danger_flags.append(f"low_price({market_price_cents}¢)")
        if is_extreme_sigma:
            danger_flags.append(f"extreme_sigma({sigma_f:.1f}°F)")
        if is_poor_calib:
            danger_flags.append(f"poor_calib({bias_level})")
        if is_very_large_disagree:
            danger_flags.append(f"very_large_disagree({disagreement:.0%})")

        # Confirming signals for "actionable".
        # For a low-price trade, signal agreement is REQUIRED — it is not optional.
        # Without strong signal alignment a low-price large-disagreement trade is too fragile.
        confirm_count = 0
        confirm_flags: list[str] = []
        has_signal_agreement = (signal_agreement >= LOW_PRICE_MIN_SIGNAL_AGREEMENT)
        if forecast_fresh:
            confirm_count += 1
            confirm_flags.append("fresh_forecast")
        if has_signal_agreement:
            confirm_count += 1
            confirm_flags.append(f"signal_agreement({signal_agreement:.0%})")
        if not is_extreme_sigma:
            confirm_count += 1
            confirm_flags.append(f"low_sigma({sigma_f:.1f}°F)")

        # Classification rule:
        #  - No danger flags → actionable
        #  - Low-price or extreme-sigma danger: MUST have signal agreement + at least one other
        #  - Other danger flags: need ≥2 confirms
        n_danger = len(danger_flags)
        if n_danger == 0:
            classification = "actionable"
            reasons = confirm_flags
        elif is_low_price or is_extreme_sigma:
            # Signal agreement is a hard requirement for low-price or extreme-sigma trades.
            # When BOTH low-price AND extreme-sigma fire together, require all 3 confirms.
            both_severe = is_low_price and is_extreme_sigma
            required = 3 if both_severe else 2
            if not has_signal_agreement:
                classification = "dangerous"
                reasons = danger_flags + [f"missing_signal_agreement({signal_agreement:.0%}<{LOW_PRICE_MIN_SIGNAL_AGREEMENT:.0%})"]
            elif confirm_count >= required:
                classification = "actionable"
                reasons = confirm_flags + [f"danger_flags_cleared: {danger_flags}"]
            else:
                classification = "dangerous"
                reasons = danger_flags + [f"only {confirm_count}/{required} confirms: {confirm_flags}"]
        else:
            required_confirms = min(n_danger, 3)
            if confirm_count >= required_confirms:
                classification = "actionable"
                reasons = confirm_flags + [f"danger_flags_cleared: {danger_flags}"]
            else:
                classification = "dangerous"
                reasons = danger_flags + [f"only {confirm_count}/{required_confirms} confirms: {confirm_flags}"]

        return classification, reasons

    def _low_price_fragile_gate(
        self,
        ticker: str,
        price: int,
        edge: float,
        min_edge_req: float,
        disagreement: float,
        mip_verdict: str,
        disagree_classification: str,
        signal_agreement: float,
        sigma_f: float,
        market_type: str,
        bias_level: str,
    ) -> tuple[bool, float, str]:
        """
        Gate for low-price (< LOW_PRICE_GATE_CENTS) entries with large disagreement.

        Returns (allow, new_min_edge_req, rejection_reason).
        If allow=True, new_min_edge_req reflects any surcharge applied.
        If allow=False, rejection_reason explains why.

        This gate is only called when price < LOW_PRICE_GATE_CENTS.
        It is a hard veto for genuinely dangerous combinations.
        """
        is_large_disagree = (mip_verdict in ("LARGE_OPP", "EXTREME_OPP"))
        is_extreme_disagree = (mip_verdict == "EXTREME_OPP")

        # Hard block: dangerous disagreement above the maximum allowed threshold
        if (disagree_classification == "dangerous"
                and disagreement > LOW_PRICE_MAX_DISAGREEMENT):
            reason = (
                f"LOW_PRICE_FRAGILE_BLOCK {ticker}: price={price}¢ < {LOW_PRICE_GATE_CENTS}¢ "
                f"disagree={disagreement:.0%} > max={LOW_PRICE_MAX_DISAGREEMENT:.0%} "
                f"classification=dangerous mip={mip_verdict} "
                f"agreement={signal_agreement:.0%} sigma={sigma_f:.1f}°F calib={bias_level}"
            )
            return False, min_edge_req, reason

        # Hard block: signal agreement too low for a low-price trade with disagreement
        if (is_large_disagree and disagree_classification == "dangerous"
                and signal_agreement < LOW_PRICE_MIN_SIGNAL_AGREEMENT):
            reason = (
                f"LOW_PRICE_FRAGILE_BLOCK {ticker}: price={price}¢ low_price+dangerous_disagree "
                f"signal_agreement={signal_agreement:.0%} < min={LOW_PRICE_MIN_SIGNAL_AGREEMENT:.0%} "
                f"mip={mip_verdict} disagree={disagreement:.0%}"
            )
            return False, min_edge_req, reason

        # Hard block: extreme sigma on a low-price trade (model itself is deeply uncertain)
        if sigma_f > LOW_PRICE_MAX_SIGMA_F and is_large_disagree:
            reason = (
                f"LOW_PRICE_FRAGILE_BLOCK {ticker}: price={price}¢ sigma={sigma_f:.1f}°F "
                f"> max={LOW_PRICE_MAX_SIGMA_F}°F with mip={mip_verdict} disagree={disagreement:.0%}"
            )
            return False, min_edge_req, reason

        # Edge surcharge: apply extra edge requirements for any low-price large-disagree combo
        if is_extreme_disagree:
            extra = LOW_PRICE_EXTREME_DISAGREE_EDGE
        elif is_large_disagree:
            extra = LOW_PRICE_LARGE_DISAGREE_EDGE
        else:
            extra = 0.0

        new_min = min_edge_req + extra
        return True, new_min, ""

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

    def _same_day_cutoff_check(
        self,
        city: str,
        market_type: str,
        threshold: Optional[float],
        hours_left: Optional[float],
        report: Optional[dict],
    ) -> tuple[bool, str]:
        """
        Weather-aware same-day rejection logic.

        For temp_high: reject once local time is past the city's heating peak hour,
        UNLESS current observed temp is already near or past the threshold (station edge).

        For temp_low: reject only after local time > 08:00 AND warming trend is positive
        (the low has clearly passed).

        For rain/snow: no time-based cutoff — precipitation can happen any hour.

        Returns (should_reject, reason).
        """
        now_utc = datetime.now(timezone.utc)

        if market_type in ("rain", "snow"):
            # Precipitation markets: never reject on time alone (timing is the edge)
            return False, ""

        tz_name = CITY_TIMEZONES.get(city)
        if not tz_name:
            # Unknown city timezone — fall back to conservative 18:00 UTC rule
            if now_utc.hour >= 18:
                return True, "unknown city tz, fallback 18:00 UTC rule"
            return False, ""

        try:
            local_now = now_utc.astimezone(ZoneInfo(tz_name))
        except Exception:
            if now_utc.hour >= 18:
                return True, f"tz parse failed for {tz_name}, fallback 18:00 UTC"
            return False, ""

        local_hour = local_now.hour

        if market_type == "temp_high":
            peak_hour = CITY_HIGH_PEAK_HOUR.get(city, DEFAULT_HIGH_PEAK_HOUR)
            if local_hour < peak_hour:
                return False, ""  # still within heating window

            # Past peak hour — check if current obs already crossed threshold
            # (station latency edge: market hasn't priced the crossing yet)
            if report and threshold is not None:
                obs = report.get("recent_observations", [])
                if obs:
                    current_temp = obs[0].get("temp_f")
                    if current_temp is not None and current_temp >= threshold - 1.0:
                        # Close enough — valid late trade (station edge)
                        return False, ""

            return True, (
                f"temp_high past city peak hour {peak_hour}:00 local "
                f"(city={city} local_hour={local_hour})"
            )

        elif market_type == "temp_low":
            # Low temp markets settle on overnight low — reject only after warming has begun
            if local_hour < 8:
                return False, ""  # overnight / early morning — low not yet recorded
            # After 8am: check if trend is warming (low has passed)
            if report:
                temp_trend = report.get("temp_trend")
                obs = report.get("recent_observations", [])
                if obs and temp_trend is not None:
                    current_temp = obs[0].get("temp_f")
                    if current_temp is not None and threshold is not None:
                        if current_temp <= threshold and temp_trend > 0:
                            # Currently at/below threshold and warming — low already hit
                            # This is a valid station-edge observation; DON'T reject
                            return False, ""
                    if temp_trend > 0.5 and local_hour >= 9 and current_temp is not None and threshold is not None and current_temp <= threshold + 5.0:
                        # Warming fast after 9am AND temp is within 5°F of threshold —
                        # the low has likely already occurred and is now rising.
                        # If current_temp is well above threshold, the low hasn't been
                        # reached yet, so don't reject.
                        return True, (
                            f"temp_low: warming trend {temp_trend:+.2f}°F/hr after 09:00 local "
                            f"(local_hour={local_hour}, current={current_temp}°F, threshold={threshold}°F)"
                        )
            # Without trend data, allow until 10:00 local
            if local_hour >= 10:
                return True, f"temp_low: past 10:00 local without trend confirmation (local_hour={local_hour})"
            return False, ""

        return False, ""

    def apply_history_insights(self, insights):
        """
        Apply historical performance insights to adjust future trade behavior.
        Uses confidence-weighted city penalties instead of a blunt avoid-list.
        Called by orchestrator morning briefing.
        """
        if not insights:
            return
        try:
            # Avoid-cities: only those with classifier action="avoid" AND sufficient sample
            self._avoid_cities = set(getattr(insights, "avoid_cities", []))

            # Build edge-penalty map: city -> extra cents required
            # Sources: (a) raise_edge_cities from classifier, (b) legacy win-rate check
            self._city_edge_adjustments = {}

            raise_edge = getattr(insights, "raise_edge_cities", {})
            for city, penalty_cents in raise_edge.items():
                self._city_edge_adjustments[city] = {
                    "edge_penalty_cents": penalty_cents,
                    "source": "classifier",
                }

            # Backward-compat: also check raw win-rate from performance_by_city
            # for any city not already covered by the classifier
            perf = getattr(insights, "performance_by_city", {})
            for city, data in perf.items():
                if city in self._city_edge_adjustments:
                    continue  # already covered by classifier
                win_rate = data.get("win_rate", 0.5)
                trade_count = data.get("trades", 0)
                if trade_count >= 10 and win_rate < 0.40:
                    self._city_edge_adjustments[city] = {
                        "edge_penalty_cents": 3.0,
                        "source": "win_rate_fallback",
                    }

            logger.info(
                f"History insights applied: avoid_cities={self._avoid_cities}, "
                f"raise_edge_cities={list(self._city_edge_adjustments.keys())}"
            )
        except Exception as e:
            logger.warning(f"apply_history_insights error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Conviction scoring
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _conviction_score(rec: "TradeRecommendation") -> float:
        """
        Multi-factor conviction score used to rank candidates when more than
        MAX_TRADES_PER_DAY qualify.  Higher is better.

        Components (each normalised to a 0–1 contribution):
          • adjusted edge (50%) — primary monetisable-edge driver
          • signal_agreement (20%) — how many signals agree
          • model certainty (15%) — inverse of model_uncertainty
          • liquidity (10%) — executable $ near mid
          • confidence tier (5%) — high/medium/low discrete penalty
        """
        ctx = rec.entry_context or {}

        # 1. Adjusted edge: use tradable_edge_cents when available (accounts for trim band
        #    discounts), otherwise fall back to raw edge.
        tradable_cents = ctx.get("tradable_edge_cents")
        if tradable_cents is not None:
            edge_component = max(0.0, tradable_cents / 100.0)
        else:
            edge_component = max(0.0, rec.edge)

        # 2. Signal agreement (0–1 already)
        agreement = ctx.get("signal_agreement", 0.5)

        # 3. Model certainty — invert uncertainty so higher certainty = higher score
        uncertainty = ctx.get("model_uncertainty", rec.model_uncertainty)
        certainty = max(0.0, 1.0 - uncertainty)

        # 4. Liquidity — normalise $0–$200 executable depth to 0–1
        exec_liq = ctx.get("exec_liq", 0.0) or 0.0
        liquidity_norm = min(1.0, exec_liq / 200.0)

        # 5. Confidence tier
        conf_bonus = {"high": 1.0, "medium": 0.6, "low": 0.2}.get(rec.confidence, 0.4)

        score = (
            0.50 * edge_component
            + 0.20 * agreement
            + 0.15 * certainty
            + 0.10 * liquidity_norm
            + 0.05 * conf_bonus
        )
        return round(score, 6)

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
        Scans all open Kalshi weather markets, scores them, and returns the
        top MAX_TRADES_PER_DAY highest-conviction recommendations that fit
        within the daily rules.

        Selection process:
          1. Evaluate every market; collect all candidates that pass filters.
          2. Deduplicate to one trade per (city, market_type).
          3. Rank survivors by _conviction_score (multi-factor: edge, agreement,
             certainty, liquidity, confidence tier).
          4. Cap at (MAX_TRADES_PER_DAY - trades_used); log skipped candidates.
        """
        trades_remaining = MAX_TRADES_PER_DAY - trades_used
        if trades_remaining <= 0:
            logger.info(
                f"DAILY_CAP_REACHED trades_used={trades_used} "
                f"max={MAX_TRADES_PER_DAY} — no further trades today"
            )
            return []

        markets = self.kalshi.get_weather_markets()
        candidates = []

        for market in markets:
            ticker = market.get("ticker", "")
            try:
                rec = self._evaluate_market(market, daily_budget, open_position_cost)
                if rec:
                    candidates.append(rec)
                else:
                    logger.debug(f"REJECTED {ticker}: no recommendation returned")
            except Exception as e:
                logger.warning(f"REJECTED {ticker}: exception during evaluation — {e}")
                continue

        # Deduplicate: only one trade per city+market_type to avoid contradictory positions.
        seen: set = set()
        deduped = []
        for rec in sorted(candidates, key=lambda r: r.edge, reverse=True):
            key = (rec.city, rec.market_type)
            if key not in seen:
                seen.add(key)
                deduped.append(rec)

        # Rank by conviction score (multi-factor)
        for rec in deduped:
            rec._conviction = self._conviction_score(rec)  # type: ignore[attr-defined]
        deduped.sort(key=lambda r: getattr(r, "_conviction", 0.0), reverse=True)

        # Log full ranked candidate list
        logger.info(
            f"CANDIDATE_RANKING trades_used={trades_used} "
            f"cap={MAX_TRADES_PER_DAY} candidates={len(deduped)}"
        )
        for rank, rec in enumerate(deduped, start=1):
            score = getattr(rec, "_conviction", 0.0)
            ctx = rec.entry_context or {}
            marker = "SELECT" if rank <= trades_remaining else "SKIP_CAP"
            agreement = ctx.get("signal_agreement", 0.5)
            uncertainty = ctx.get("model_uncertainty", rec.model_uncertainty)
            exec_liq = ctx.get("exec_liq", 0) or 0
            logger.info(
                f"  RANK_{rank:02d} [{marker}] {rec.ticker} "
                f"conviction={score:.4f} edge={rec.edge:.3f} "
                f"agreement={agreement:.2f} "
                f"certainty={1.0 - uncertainty:.2f} "
                f"liq=${exec_liq:.0f} "
                f"confidence={rec.confidence} "
                f"city={rec.city} type={rec.market_type}"
            )

        selected = deduped[:trades_remaining]
        skipped = deduped[trades_remaining:]
        if skipped:
            skipped_tickers = [r.ticker for r in skipped]
            logger.info(
                f"SKIPPED_BY_CAP {len(skipped)} candidates not executed "
                f"(daily limit {MAX_TRADES_PER_DAY} trades): {skipped_tickers}"
            )

        return selected

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

        # Skip cities the classifier has marked as persistent underperformers
        # (only after sufficient sample size — see classifier.py thresholds)
        if city in self._avoid_cities:
            logger.info(f"REJECTED {ticker}: {city} in classifier avoid_cities (confidence-weighted)")
            return None

        # ── Time filter ────────────────────────────────────────────────────────
        close_time = market.get("close_time") or market.get("expiration_time")
        hours_left: Optional[float] = None
        _is_same_day = False
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

                # Pre-report same-day check (without observation data).
                # If it rejects, we defer final decision until after the report is fetched
                # so station-edge exceptions (obs already past threshold) can override.
                if ct.date() == now_utc.date():
                    _is_same_day = True
                    should_reject, cutoff_reason = self._same_day_cutoff_check(
                        city, market_type, threshold, hours_left, None
                    )
                    if should_reject:
                        # Don't hard-reject yet — re-check after report fetch
                        logger.debug(
                            f"{ticker}: pre-report same-day flag — {cutoff_reason} "
                            f"(will re-check with observations)"
                        )
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

        # Weather data integrity: check observation and forecast freshness
        now_utc = datetime.now(timezone.utc)
        stale_station = False
        obs = report.get("recent_observations", [])
        if obs:
            latest_obs_time = obs[0].get("timestamp")
            if latest_obs_time:
                try:
                    obs_dt = datetime.fromisoformat(latest_obs_time.replace("Z", "+00:00"))
                    obs_age_h = (now_utc - obs_dt).total_seconds() / 3600
                    if obs_age_h > 4.0 and (hours_left is None or hours_left > 2.0):
                        # Stale METAR outside final window — reject new entries
                        logger.debug(
                            f"REJECTED {ticker}: station observation is {obs_age_h:.1f}h old "
                            f"(limit 4h outside final 2h window)"
                        )
                        return None
                    elif obs_age_h > 1.0:
                        stale_station = True
                        logger.debug(
                            f"{ticker}: station observation is {obs_age_h:.1f}h old — "
                            f"metar signals penalized"
                        )
                except Exception:
                    pass

        # Forecast freshness: reject if NWS forecast is older than 4h (outside final 2h)
        forecast_generated_at = report.get("forecast", {}).get("generated_at")
        if forecast_generated_at and (hours_left is None or hours_left > 2.0):
            try:
                fc_dt = datetime.fromisoformat(forecast_generated_at.replace("Z", "+00:00"))
                fc_age_h = (now_utc - fc_dt).total_seconds() / 3600
                if fc_age_h > 4.0:
                    logger.debug(
                        f"REJECTED {ticker}: NWS forecast is {fc_age_h:.1f}h old "
                        f"(limit 4h outside final 2h window)"
                    )
                    return None
            except Exception:
                pass

        # ── Post-report same-day cutoff (with observation data) ───────────────
        # Now we have fresh obs — re-run the same-day check so station-edge
        # exceptions (observed temp already near/past threshold) can override.
        if _is_same_day:
            should_reject, cutoff_reason = self._same_day_cutoff_check(
                city, market_type, threshold, hours_left, report
            )
            if should_reject:
                logger.debug(f"REJECTED {ticker}: same-day cutoff (post-obs) — {cutoff_reason}")
                return None

        # ── Score the market ───────────────────────────────────────────────────
        sigma = self._get_dynamic_sigma(city, market_type, hours_left)
        calibration = self._load_forecast_calibration() if market_type == "temp_low" else {}
        score_result = self._score_market(
            market_type, threshold, report, city, is_bucket, sigma, hours_left,
            calibration=calibration,
        )
        if score_result is None:
            return None
        base_prob, reasoning, forecast_summary = score_result

        # ── Temp_low calibration block / extra-edge gate ───────────────────────
        if market_type == "temp_low" and calibration:
            bias_level, mean_bias, n_bias = self._get_temp_low_bias_status(city, calibration)
            if bias_level == "block":
                logger.warning(
                    f"REJECTED {ticker}: temp_low BLOCKED for {city} — "
                    f"calib mean_err={mean_bias:+.1f}°F n={n_bias} "
                    f"(threshold: {TEMP_LOW_BIAS_BLOCK_F}°F, min_samples: {MIN_BLOCK_SAMPLES})"
                )
                self._record_missed_opportunity(
                    ticker, city, market_type,
                    f"temp_low_calib_block(mean={mean_bias:+.1f}F,n={n_bias})",
                    base_prob, yes_price if yes_price else 0, 0.0,
                )
                return None

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

        # Temperature trajectory (with path-to-threshold projection)
        if market_type in ("temp_high", "temp_low") and threshold is not None:
            try:
                sig = temperature_trajectory.compute(
                    report, threshold, market_type,
                    hours_to_close=hours_left,
                )
                signal_inputs.append(sig)
            except Exception as e:
                logger.debug(f"temp_trajectory signal error: {e}")

        # Forecast update shock
        try:
            sig = forecast_update.compute(report, threshold, market_type)
            signal_inputs.append(sig)
        except Exception as e:
            logger.debug(f"forecast_update signal error: {e}")

        # METAR latency (with age tiers and local-hour context)
        if market_type in ("temp_high", "temp_low") and threshold is not None:
            try:
                # Derive local hour from city timezone for time-of-day context
                metar_local_hour: Optional[int] = None
                tz_name = CITY_TIMEZONES.get(city)
                if tz_name:
                    try:
                        metar_local_hour = datetime.now(timezone.utc).astimezone(ZoneInfo(tz_name)).hour
                    except Exception:
                        pass
                sig = metar_latency.compute(
                    report, threshold, market_type,
                    stale_station=stale_station,
                    local_hour=metar_local_hour,
                )
                signal_inputs.append(sig)
            except Exception as e:
                logger.debug(f"metar_latency signal error: {e}")

        # Orderbook microstructure
        try:
            sig = orderbook_microstructure.compute(liquidity, base_prob, side)
            signal_inputs.append(sig)
        except Exception as e:
            logger.debug(f"orderbook signal error: {e}")

        # Market implied probability (extreme disagreement guard + diagnosis)
        extreme_disagreement = False
        mip_verdict = "NONE"
        raw_disagreement = 0.0
        try:
            mip_sig = market_implied_prob.compute(
                liquidity, base_prob, side,
                forecast_report=report.get("forecast"),
                hours_left=hours_left,
            )
            signal_inputs.append(mip_sig)
            # Read verdict and raw_disagreement directly from the signal dict.
            # Fall back to note-string parsing for compatibility with older versions.
            if "verdict" in mip_sig:
                mip_verdict = mip_sig["verdict"]
            else:
                mip_note = mip_sig.get("note", "")
                for _v in ("EXTREME_WARN", "EXTREME_OPP", "LARGE_WARN", "LARGE_OPP"):
                    if _v in mip_note:
                        mip_verdict = _v
                        break
            if "raw_disagreement" in mip_sig:
                raw_disagreement = mip_sig["raw_disagreement"]
            else:
                _best_yes = liquidity.get("best_yes_price")
                if _best_yes is not None:
                    raw_disagreement = abs(base_prob - _best_yes / 100.0)
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

        # Build structured signal breakdown for offline analysis
        # (stored on the recommendation and in shadow_mode_log.json)
        signal_breakdown = [
            {
                "name":             s.get("note", f"signal_{i}"),
                "prob_adjustment":  round(s.get("prob_adjustment", 0.0), 4),
                "confidence":       round(s.get("confidence", 0.5), 3),
                "note":             s.get("note", ""),
            }
            for i, s in enumerate(signal_inputs)
        ]

        # Aggregate all signals
        agg = signal_aggregator.aggregate(signal_inputs)

        # Apply contextual probability adjustment cap (regime-aware, not flat ±20¢)
        adj_cap = self._compute_adj_cap(market_type, hours_left, agg, report, threshold)
        adj = max(-adj_cap, min(adj_cap, agg.prob_adjustment))
        logger.debug(
            f"{ticker}: signal_cap={adj_cap:.2f} raw_adj={agg.prob_adjustment:+.4f} "
            f"clamped_adj={adj:+.4f} agreement={agg.signal_agreement:.0%} "
            f"active={agg.active_signals}"
        )
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

        # Historical performance edge adjustment (confidence-weighted, from classifier)
        city_adj = self._city_edge_adjustments.get(city)
        if city_adj:
            penalty_cents = city_adj.get("edge_penalty_cents", 0.0)
            if penalty_cents > 0:
                min_edge_req = max(min_edge_req, min_edge_req + penalty_cents / 100.0)
                logger.debug(
                    f"{ticker}: city={city} edge penalty +{penalty_cents:.0f}¢ "
                    f"(source={city_adj.get('source', '?')}) → min_edge={min_edge_req:.3f}"
                )

        # Temp_low calibration edge surcharge (warn/penalty levels)
        # Also enforce hard minimum and extreme-sigma penalty for poor calibration.
        bias_level_for_gate = "ok"
        if market_type == "temp_low" and calibration:
            bias_level_for_gate, mean_bias, n_bias = self._get_temp_low_bias_status(city, calibration)
            if bias_level_for_gate == "penalty":
                min_edge_req += TEMP_LOW_PENALTY_EDGE_ADD
                logger.info(
                    f"{ticker}: temp_low PENALTY edge surcharge +{TEMP_LOW_PENALTY_EDGE_ADD*100:.0f}¢ "
                    f"calib mean_err={mean_bias:+.1f}°F n={n_bias} → min_edge={min_edge_req:.3f}"
                )
            elif bias_level_for_gate == "warn":
                min_edge_req += TEMP_LOW_WARN_EDGE_ADD
                logger.info(
                    f"{ticker}: temp_low WARN edge surcharge +{TEMP_LOW_WARN_EDGE_ADD*100:.0f}¢ "
                    f"calib mean_err={mean_bias:+.1f}°F n={n_bias} → min_edge={min_edge_req:.3f}"
                )

            # Hard minimum for any temp_low with degraded calibration (warn or worse)
            if bias_level_for_gate in ("warn", "penalty") and min_edge_req < TEMP_LOW_POOR_CALIB_MIN_EDGE:
                logger.info(
                    f"{ticker}: temp_low poor-calib hard min_edge floor "
                    f"{min_edge_req:.3f} → {TEMP_LOW_POOR_CALIB_MIN_EDGE:.3f} "
                    f"(calib={bias_level_for_gate})"
                )
                min_edge_req = TEMP_LOW_POOR_CALIB_MIN_EDGE

            # Extra surcharge when sigma is extreme (model is deeply uncertain)
            if sigma > TEMP_LOW_EXTREME_SIGMA_F:
                min_edge_req += TEMP_LOW_EXTREME_SIGMA_EDGE
                logger.info(
                    f"{ticker}: temp_low extreme sigma surcharge +{TEMP_LOW_EXTREME_SIGMA_EDGE*100:.0f}¢ "
                    f"sigma={sigma:.1f}°F > {TEMP_LOW_EXTREME_SIGMA_F}°F → min_edge={min_edge_req:.3f}"
                )

        # ── Low-price + large-disagreement combined gate ───────────────────────
        # Determine whether forecast is fresh (for disagree classification)
        _forecast_fresh = False
        _fc_generated = report.get("forecast", {}).get("generated_at")
        if _fc_generated:
            try:
                _fc_dt = datetime.fromisoformat(_fc_generated.replace("Z", "+00:00"))
                _fc_age_h = (datetime.now(timezone.utc) - _fc_dt).total_seconds() / 3600
                _forecast_fresh = (_fc_age_h < 3.0)
            except Exception:
                pass

        disagree_classification = "actionable"
        disagree_reasons: list[str] = []
        is_low_price_entry = (price < LOW_PRICE_GATE_CENTS)

        if mip_verdict in ("LARGE_OPP", "EXTREME_OPP", "LARGE_WARN", "EXTREME_WARN"):
            disagree_classification, disagree_reasons = self._classify_disagreement(
                disagreement=raw_disagreement,
                market_price_cents=price,
                sigma_f=sigma,
                signal_agreement=agg.signal_agreement,
                mip_verdict=mip_verdict,
                bias_level=bias_level_for_gate,
                market_type=market_type,
                forecast_fresh=_forecast_fresh,
            )
            logger.info(
                f"{ticker}: disagree_classify={disagree_classification} "
                f"mip={mip_verdict} disagree={raw_disagreement:.0%} "
                f"price={price}¢ sigma={sigma:.1f}°F "
                f"agreement={agg.signal_agreement:.0%} fresh_fc={_forecast_fresh} "
                f"reasons={disagree_reasons}"
            )

        # Low-price fragile gate (only applies when price < LOW_PRICE_GATE_CENTS)
        if is_low_price_entry:
            _lp_allow, min_edge_req, _lp_reason = self._low_price_fragile_gate(
                ticker=ticker,
                price=price,
                edge=edge,
                min_edge_req=min_edge_req,
                disagreement=raw_disagreement,
                mip_verdict=mip_verdict,
                disagree_classification=disagree_classification,
                signal_agreement=agg.signal_agreement,
                sigma_f=sigma,
                market_type=market_type,
                bias_level=bias_level_for_gate,
            )
            if not _lp_allow:
                logger.warning(_lp_reason)
                self._record_missed_opportunity(
                    ticker=ticker, city=city, market_type=market_type,
                    rejection_reason="low_price_fragile_block",
                    our_prob=our_prob, market_price_cents=price, edge_cents=edge * 100,
                )
                return None

        if edge < min_edge_req:
            gap = min_edge_req - edge
            rejection_reason = (
                f"edge={edge:.3f}_below_min={min_edge_req:.3f}"
                if not extreme_disagreement
                else f"extreme_disagreement_edge={edge:.3f}_below_min={min_edge_req:.3f}"
            )
            if gap <= 0.03:
                # Close miss — log at INFO and record as missed opportunity candidate
                logger.info(
                    f"CLOSE_REJECT {ticker}: edge={edge:.3f} just below min={min_edge_req:.3f} "
                    f"(gap={gap:.3f}) city={city} type={market_type} "
                    f"extreme_disagree={extreme_disagreement}"
                )
                self._record_missed_opportunity(
                    ticker=ticker, city=city, market_type=market_type,
                    rejection_reason=f"close_miss_{rejection_reason}",
                    our_prob=our_prob, market_price_cents=price, edge_cents=edge * 100,
                )
            else:
                logger.debug(
                    f"REJECTED {ticker}: edge={edge:.3f} < min_edge={min_edge_req:.3f} "
                    f"(vol={total_volume}, extreme_disagree={extreme_disagreement})"
                )
                # Only record missed opportunity if edge was actually meaningful (>3¢)
                if edge >= 0.03:
                    self._record_missed_opportunity(
                        ticker=ticker, city=city, market_type=market_type,
                        rejection_reason=rejection_reason,
                        our_prob=our_prob, market_price_cents=price, edge_cents=edge * 100,
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

        # ── Confidence tier (computed early for trim-band guard) ──────────────
        # Round-number thresholds get one tier lower (threshold_cluster_confidence
        # was computed above during signal aggregation).
        if edge >= 0.15:
            confidence = "high" if threshold_cluster_confidence >= 0.5 else "medium"
        elif edge >= 0.10:
            confidence = "medium" if threshold_cluster_confidence >= 0.3 else "low"
        else:
            confidence = "low"

        # ── Trim-band entry guard ──────────────────────────────────────────────
        # A fresh position opened inside a staged profit-taking band will be
        # immediately subject to a trim, making the monetizable edge much smaller
        # than the raw edge figure. Reject or warn depending on which band we're in.
        _spread_for_guard = int(spread) if spread else 5  # fallback 5¢ if unavailable
        _tb_allow, _tb_zone, _tb_info = self._trim_band_entry_check(
            ticker=ticker,
            price=price,
            edge=edge,
            min_edge_req=min_edge_req,
            confidence=confidence,
            spread=_spread_for_guard,
        )
        if _tb_zone != "clear":
            if not _tb_allow:
                logger.info(
                    f"TRIM_BAND_REJECT {ticker}: {_tb_info['reason_detail']} | "
                    f"upside_before_trim={_tb_info['upside_before_trim_cents']}¢ "
                    f"tradable_edge={_tb_info['tradable_edge_cents']:.1f}¢"
                )
                self._record_missed_opportunity(
                    ticker=ticker, city=city, market_type=market_type,
                    rejection_reason=f"trim_band_reject_{_tb_zone}",
                    our_prob=our_prob, market_price_cents=price,
                    edge_cents=edge * 100,
                )
                return None
            elif _tb_zone == "near_trim_band":
                logger.warning(
                    f"TRIM_BAND_WARN {ticker}: {_tb_info['reason_detail']} | "
                    f"upside_before_trim={_tb_info['upside_before_trim_cents']}¢ "
                    f"tradable_edge={_tb_info['tradable_edge_cents']:.1f}¢"
                )
            else:
                logger.info(
                    f"TRIM_BAND_ACCEPT {ticker}: {_tb_info['reason_detail']} | "
                    f"upside_before_trim={_tb_info['upside_before_trim_cents']}¢ "
                    f"tradable_edge={_tb_info['tradable_edge_cents']:.1f}¢"
                )

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
            if price <= 0:
                return None
            prob_win = our_prob if side == "yes" else (1 - our_prob)
            prob_lose = 1 - prob_win
            odds = (100 - price) / price
            kelly = max(0, (prob_win * odds - prob_lose) / odds)
            position_dollars = min(daily_budget * kelly * 0.25, MAX_DOLLARS_PER_TRADE)

        # Apply absolute caps; use relative minimum (1% of budget, floor $0.10) not fixed $0.50
        max_position = min(daily_budget * MAX_POSITION_PCT, MAX_DOLLARS_PER_TRADE)
        min_position = max(0.10, daily_budget * 0.01)
        position_dollars = min(position_dollars, max_position)
        position_dollars = max(min_position, round(position_dollars, 2))

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
            f"mip={mip_verdict} disagree={raw_disagreement:.0%} "
            f"disagree_class={disagree_classification} "
            f"low_price={is_low_price_entry} | "
            f"contracts={contracts} cost=${actual_cost:.2f} confidence={confidence} | "
            f"weights_ver={agg.weights_version} signals={signal_notes}"
        )

        # ── Low-price diagnostic snapshot (logged for every accepted low-price entry) ──
        if is_low_price_entry:
            _mkt_implied_prob = liquidity.get("best_yes_price")
            _mkt_implied_str = f"{_mkt_implied_prob / 100.0:.4f}" if _mkt_implied_prob else "N/A"
            logger.info(
                f"LOW_PRICE_ENTRY_SNAPSHOT {ticker} | "
                f"price={price}¢ base_prob={base_prob:.4f} our_prob={our_prob:.4f} "
                f"market_implied={_mkt_implied_str} "
                f"disagree={raw_disagreement:.0%} mip={mip_verdict} "
                f"disagree_class={disagree_classification} "
                f"sigma={sigma:.1f}°F agreement={agg.signal_agreement:.0%} "
                f"calib={bias_level_for_gate} edge={edge:.3f} min_edge={min_edge_req:.3f} "
                f"reasons={disagree_reasons}"
            )

        # ── Shadow-mode evaluation log ─────────────────────────────────────────
        # Always log every scored recommendation (executed or not) so the
        # validation framework can compare candidate configs offline.
        try:
            from src.analysis.validation import log_shadow_evaluation
            log_shadow_evaluation(
                ticker=ticker,
                city=city,
                market_type=market_type,
                our_prob=our_prob,
                market_price_cents=price,
                edge_cents=round(edge * 100, 2),
                signal_breakdown=signal_breakdown,
                weights_version=agg.weights_version,
                context={
                    "hours_left":             hours_left,
                    "base_prob":              round(base_prob, 4),
                    "signal_adj":             round(adj, 4),
                    "cap_regime":             agg.cap_regime,
                    "signal_agreement":       round(agg.signal_agreement, 3),
                    "model_uncertainty":      round(agg.model_uncertainty, 3),
                    "sigma_f":                round(sigma, 2),
                    "extreme_disagreement":   extreme_disagreement,
                    "mip_verdict":            mip_verdict,
                    "raw_disagreement":       round(raw_disagreement, 4),
                    "disagree_classification": disagree_classification,
                    "is_low_price_entry":     is_low_price_entry,
                    "calib_bias_level":       bias_level_for_gate,
                },
            )
        except Exception as _shadow_err:
            logger.debug(f"shadow_log failed for {ticker}: {_shadow_err}")

        # ── Build entry context snapshot ──────────────────────────────────────
        # Contains every field needed to reconstruct the decision at entry time.
        # Stored on the Position via record_trade(entry_context=...) so that
        # post-trade diagnostics and the classifier have the full picture.
        entry_ctx: dict = {
            "our_prob":            round(our_prob, 4),
            "base_prob":           round(base_prob, 4),
            "signal_adj":          round(adj, 4),
            "edge":                round(edge, 4),
            "sigma_f":             round(sigma, 2),
            "hours_left":          hours_left,
            "cap_regime":          agg.cap_regime,
            "signal_agreement":    round(agg.signal_agreement, 3),
            "model_uncertainty":   round(agg.model_uncertainty, 3),
            "extreme_disagreement": extreme_disagreement,
            # ── Disagreement classification ──────────────────────────────────
            "mip_verdict":         mip_verdict,
            "raw_disagreement":    round(raw_disagreement, 4),
            "disagree_classification": disagree_classification,
            "disagree_reasons":    disagree_reasons,
            # ── Low-price entry diagnostics ──────────────────────────────────
            "is_low_price_entry":  is_low_price_entry,
            "calib_bias_level":    bias_level_for_gate,
            "spread":              (liquidity.get("spread") or
                                    (liquidity.get("best_ask_price", 50) -
                                     liquidity.get("best_bid_price", 50))),
            "exec_liq":            round(exec_liq, 2),
            "weights_version":     agg.weights_version,
            "signal_breakdown":    signal_breakdown,
            "trim_band_zone":      _tb_zone,
            "tradable_edge_cents": _tb_info["tradable_edge_cents"],
        }

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
            model_uncertainty=round(agg.model_uncertainty, 3),
            signal_breakdown=signal_breakdown,
            weights_version=agg.weights_version,
            entry_context=entry_ctx,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Trim-band entry guard
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _trim_band_entry_check(
        ticker: str,
        price: int,
        edge: float,
        min_edge_req: float,
        confidence: str,
        spread: int,
    ) -> tuple[bool, str, dict]:
        """
        Evaluates whether a *fresh* entry at `price` is economically sound given
        the sell logic's staged profit-taking bands.

        Sell logic trims positions at:
          70–79¢: 25 %   80–89¢: 40 %   90–95¢: 65 %   96–99¢: 85 %

        A brand-new position opened inside one of those bands will be immediately
        subject to a trim — the monetizable upside is much smaller than the raw
        edge suggests.

        Returns:
          (allow: bool, reason: str, info: dict)

        `info` contains all fields required for the structured log entry:
          - price, edge_cents, tradable_edge_cents, upside_before_trim_cents,
            trim_band, trim_fraction, zone, reason_detail
        """
        from src.tracker.pnl import PROFIT_TAKE_BANDS  # avoid circular import at module level

        edge_cents = round(edge * 100, 2)
        slippage = TRIM_BAND_SLIPPAGE_TOTAL

        # Determine which band (if any) the entry price falls into, or is near.
        in_band: Optional[tuple] = None
        near_band: Optional[tuple] = None
        for band in PROFIT_TAKE_BANDS:
            band_min, band_max, trim_frac = band
            if band_min <= price <= band_max:
                in_band = band
                break
            # "near" = within TRIM_BAND_NEAR_WARN_CENTS below the band
            if band_min - TRIM_BAND_NEAR_WARN_CENTS <= price < band_min:
                near_band = band
                # Don't break — a higher band might actually contain the price

        if in_band is None and near_band is None:
            # Well below all trim bands — no coordination issue
            info = {
                "ticker": ticker,
                "price": price,
                "edge_cents": edge_cents,
                "tradable_edge_cents": edge_cents,
                "upside_before_trim_cents": PROFIT_TAKE_BANDS[0][0] - price,
                "trim_band": None,
                "trim_fraction": 0.0,
                "zone": "clear",
                "reason_detail": "price below all trim bands",
            }
            return True, "clear", info

        # ── Inside a trim band ────────────────────────────────────────────────
        if in_band is not None:
            band_min, band_max, trim_frac = in_band

            # Upside before the trim fires = 0 (already inside the band).
            # After trimming trim_frac of contracts the remaining position can
            # run to 100¢; the drag is the profit left on the trimmed fraction.
            # Conservative estimate: trimmed contracts are sold back at roughly
            # entry_price (worst case: sell logic fires at the band floor).
            # That means we lose ~trim_frac * spread/2 from entry on those
            # contracts, plus slippage on the full round trip.
            trim_drag = trim_frac * (spread / 2.0)
            tradable_edge_cents = round(edge_cents - slippage - trim_drag, 2)

            # Determine the extra edge hurdle for this band
            if band_min >= 90:
                extra_hurdle = TRIM_BAND_EXTRA_EDGE_90_PLUS * 100  # convert to cents
                zone = "trim_90_plus"
            elif band_min >= 80:
                extra_hurdle = TRIM_BAND_EXTRA_EDGE_80_89 * 100
                zone = "trim_80_89"
            else:
                extra_hurdle = TRIM_BAND_EXTRA_EDGE_70_79 * 100
                zone = "trim_70_79"

            required_edge_cents = round((min_edge_req * 100) + extra_hurdle, 2)
            allow = (
                edge_cents >= required_edge_cents
                and confidence == "high"  # band entries require high confidence
            )

            reason_detail = (
                f"entry_price={price}¢ inside trim band [{band_min}–{band_max}¢] "
                f"(trim_frac={trim_frac:.0%}); "
                f"edge={edge_cents:.1f}¢ tradable={tradable_edge_cents:.1f}¢ "
                f"required={required_edge_cents:.1f}¢ confidence={confidence}; "
                f"{'ACCEPTED' if allow else 'REJECTED'}"
            )
            info = {
                "ticker": ticker,
                "price": price,
                "edge_cents": edge_cents,
                "tradable_edge_cents": tradable_edge_cents,
                "upside_before_trim_cents": 0,
                "trim_band": f"{band_min}-{band_max}",
                "trim_fraction": trim_frac,
                "zone": zone,
                "required_edge_cents": required_edge_cents,
                "reason_detail": reason_detail,
            }
            return allow, zone, info

        # ── Near a trim band (within TRIM_BAND_NEAR_WARN_CENTS below it) ─────
        band_min, band_max, trim_frac = near_band  # type: ignore[misc]
        upside_before_trim = band_min - price  # positive: 1–5¢ headroom

        # Tradable edge = raw edge - slippage. No trim drag yet (trim hasn't
        # fired), but upside before trim is slim so log as a warning.
        tradable_edge_cents = round(edge_cents - slippage, 2)

        zone = "near_trim_band"
        reason_detail = (
            f"entry_price={price}¢ within {TRIM_BAND_NEAR_WARN_CENTS}¢ of "
            f"trim band [{band_min}–{band_max}¢] (trim_frac={trim_frac:.0%}); "
            f"upside_before_trim={upside_before_trim}¢ "
            f"edge={edge_cents:.1f}¢ tradable={tradable_edge_cents:.1f}¢; WARN_ALLOWED"
        )
        info = {
            "ticker": ticker,
            "price": price,
            "edge_cents": edge_cents,
            "tradable_edge_cents": tradable_edge_cents,
            "upside_before_trim_cents": upside_before_trim,
            "trim_band": f"{band_min}-{band_max}",
            "trim_fraction": trim_frac,
            "zone": zone,
            "reason_detail": reason_detail,
        }
        # Allow near-band entries but emit a warning for post-trade review
        return True, zone, info

    # ─────────────────────────────────────────────────────────────────────────
    # Contextual signal adjustment cap
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_adj_cap(
        market_type: str,
        hours_left: Optional[float],
        agg,
        report: dict,
        threshold: Optional[float],
    ) -> float:
        """
        Returns the appropriate ±cap for signal probability adjustment.

        Regimes:
          noisy:            signal agreement < 0.6 AND < 3 active signals  → ±0.08
          normal (default):                                                 → ±0.12
          high-confidence:  agreement >= 0.8, active >= 3, hours_left < 6  → ±0.18
          final-threshold:  hours_left < 1, obs temp within 1°F of thresh  → ±0.22
        """
        active = agg.active_signals
        agreement = agg.signal_agreement

        # Noisy regime — few, conflicting signals
        if agreement < 0.6 and active < 3:
            return 0.08

        # Final-hour threshold-crossing regime
        if hours_left is not None and hours_left < 1.0 and threshold is not None:
            obs = report.get("recent_observations", [])
            if obs:
                cur = obs[0].get("temp_f")
                if cur is not None and abs(cur - threshold) <= 1.5:
                    return 0.22

        # High-confidence live-data regime
        if agreement >= 0.8 and active >= 3 and hours_left is not None and hours_left < 6.0:
            return 0.18

        return 0.12

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

    def _parse_market(self, ticker: str, title: str) -> Optional[tuple]:
        """
        Attempts to extract (city, market_type, threshold, is_bucket) from ticker/title.
        Returns None if we can't parse it confidently.
        Threshold is returned as float to preserve precision (e.g. 67.5°F).
        """
        ticker_upper = ticker.upper()
        title_lower = title.lower()

        city = None
        for prefix, mapped_city in SERIES_CITY_MAP.items():
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
        series_prefix = next((p for p in SERIES_CITY_MAP if ticker_upper.startswith(p)), "")
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
        hours_to_close: Optional[float] = None,
        calibration: Optional[dict] = None,
    ) -> Optional[tuple[float, str, str]]:
        """
        Returns (probability, reasoning, forecast_summary) or None if market type unsupported.
        probability is our estimated P(YES) for the market.
        """
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

            # ── Calibration bias correction for temp_low ──────────────────────
            # If the forecast source has a systematic cold bias (forecasts too low),
            # correct the forecast upward by a fraction of the observed mean error.
            # mean_error = forecast − actual, so negative = forecast ran too cold.
            # Correction: subtract the fractional bias from corrected_low
            # (making the effective forecast warmer, consistent with the observation).
            bias_level = "ok"
            mean_bias = 0.0
            n_bias_samples = 0
            bias_correction = 0.0
            sigma_penalty = 0.0

            if calibration:
                bias_level, mean_bias, n_bias_samples = self._get_temp_low_bias_status(
                    city, calibration
                )
                if bias_level in ("penalty", "block"):
                    # Apply partial bias correction (raise effective forecast)
                    # mean_bias is negative, so subtracting it raises the forecast
                    bias_correction = -mean_bias * TEMP_LOW_BIAS_CORRECTION_FRAC
                    sigma_penalty = TEMP_LOW_PENALTY_SIGMA_ADD_F
                elif bias_level == "warn":
                    bias_correction = -mean_bias * TEMP_LOW_BIAS_CORRECTION_FRAC * 0.5

            effective_sigma = sigma + sigma_penalty
            corrected_low = low_temp + trend_adj + bias_correction

            if is_bucket:
                # Bucket: P(threshold <= low <= threshold+2) = P(low <= threshold+2) - P(low <= threshold)
                p_above_lo = self._sigmoid((corrected_low - threshold) / effective_sigma)
                p_above_hi = self._sigmoid((corrected_low - (threshold + 2.0)) / effective_sigma)
                prob = p_above_lo - p_above_hi
            else:
                # temp_low YES wins when low_temp <= threshold, so P(YES) = P(X <= threshold) = "below"
                prob = self._monte_carlo_prob(corrected_low, threshold, effective_sigma, "below")

            trend_str = f"{temp_trend:+.3f}" if temp_trend is not None else "N/A"
            reasoning = (
                f"NWS low {low_temp}°F + trend_adj {trend_adj:+.1f}°F"
                f"{f' + bias_correction {bias_correction:+.1f}°F' if bias_correction else ''}"
                f" = {corrected_low:.1f}°F "
                f"vs threshold {threshold}°F (gap: {corrected_low - threshold:+.1f}°F). "
                f"{'Bucket market. ' if is_bucket else ''}"
                f"Dynamic sigma={sigma:.1f}°F"
                f"{f' + penalty {sigma_penalty:+.1f}°F = {effective_sigma:.1f}°F' if sigma_penalty else ''}. "
                f"Trend: {trend_str}°F/hr. "
                f"Calib [{city}/temp_low]: level={bias_level} mean_err={mean_bias:+.1f}°F n={n_bias_samples}."
            )
            logger.info(
                f"TEMP_LOW_SCORE | city={city} ticker_threshold={threshold}°F "
                f"nws_low={low_temp}°F trend_adj={trend_adj:+.1f}°F bias_correction={bias_correction:+.1f}°F "
                f"corrected_low={corrected_low:.1f}°F sigma={sigma:.1f}°F sigma_penalty={sigma_penalty:+.1f}°F "
                f"effective_sigma={effective_sigma:.1f}°F prob={prob:.3f} "
                f"calib_level={bias_level} mean_bias={mean_bias:+.1f}°F n={n_bias_samples}"
            )
            return prob, reasoning, forecast_summary

        elif market_type == "rain":
            prob, reasoning = self._score_rain(
                precip_chance, hourly, moon, alerts, short_fc, hours_to_close
            )
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
        hours_left: Optional[float] = None,
    ) -> tuple[float, str]:
        """
        Rain probability using a clustered hourly model (reduces naive independence bias).

        Model:
          - Identify storm clusters: consecutive hours with precip_chance >= 20%.
          - Treat each cluster as a single correlated event with P = max(cluster) * 0.85.
          - Independent hours outside clusters multiply as (1 - p_i).
          - Only include hourly slots within the settlement window (hours_left).
          - Storm-system persistence bonus if weather text suggests organized event.
          - Wet-bulb ambiguity penalty if low_temp near freezing.
        """
        moon_phase = moon.get("phase_name", "")

        # Filter hourly data to settlement window
        max_hours = min(12, int(hours_left) + 1) if hours_left is not None else 12
        relevant_hourly = [h for h in hourly[:max_hours] if h.get("precip_chance") is not None]

        has_rain_alert = any(
            "rain" in a["event"].lower() or "flood" in a["event"].lower()
            for a in alerts
        )

        if relevant_hourly:
            # Identify clusters of consecutive rainy hours (precip_chance >= 20%)
            in_cluster = False
            cluster_probs: list[float] = []
            isolated_no_rains: list[float] = []
            cluster_contributions: list[float] = []

            i = 0
            while i < len(relevant_hourly):
                p = relevant_hourly[i].get("precip_chance", 0) / 100.0
                if p >= 0.20:
                    # Start or continue a cluster
                    cluster_probs.append(p)
                    in_cluster = True
                    # Look ahead to collect consecutive rainy hours
                    j = i + 1
                    while j < len(relevant_hourly):
                        p_next = relevant_hourly[j].get("precip_chance", 0) / 100.0
                        if p_next >= 0.20:
                            cluster_probs.append(p_next)
                            j += 1
                        else:
                            break
                    # Cluster event probability: max of cluster * 0.85 (correlated, not independent)
                    cluster_p_event = min(0.97, max(cluster_probs) * 0.85)
                    cluster_contributions.append(cluster_p_event)
                    cluster_probs = []
                    i = j
                else:
                    # Isolated dry or low-chance hour
                    isolated_no_rains.append(1 - p)
                    i += 1

            # P(no rain from isolated hours) = product of isolated no-rains
            no_rain_isolated = 1.0
            for nr in isolated_no_rains:
                no_rain_isolated *= nr

            # P(no rain from any cluster) = product of (1 - cluster_p_event)
            no_rain_clusters = 1.0
            for cp in cluster_contributions:
                no_rain_clusters *= (1 - cp)

            hourly_any_rain = 1.0 - (no_rain_isolated * no_rain_clusters)

            # Blend: 60% clustered hourly model, 40% NWS point forecast
            base_prob = 0.60 * hourly_any_rain + 0.40 * (precip_chance / 100.0)
        else:
            base_prob = precip_chance / 100.0

        # Storm persistence bonus: organized weather systems are more reliable
        storm_keywords = ["storm", "system", "front", "low pressure", "squall", "line"]
        has_storm_system = any(kw in short_fc.lower() for kw in storm_keywords)
        if has_storm_system:
            base_prob = min(0.95, base_prob + 0.05)

        # Active rain/flood alerts
        if has_rain_alert:
            base_prob = min(1.0, base_prob + 0.10)

        prob = min(1.0, max(0.01, base_prob))

        reasoning = (
            f"NWS precipitation probability: {precip_chance}%. "
            f"Clustered hourly model (window={max_hours}h): {prob:.0%}. "
            f"Moon phase: {moon_phase}. "
            f"{'Storm system detected. ' if has_storm_system else ''}"
            f"{'Active rain/flood alerts. ' if has_rain_alert else ''}"
            f"Forecast: {short_fc}."
        )
        return prob, reasoning

    @staticmethod
    def _estimate_wet_bulb(temp_f: float, dewpoint_f: Optional[float]) -> Optional[float]:
        """
        Estimates wet-bulb temperature (°F) using Stull formula approximation.
        Returns None if dewpoint is unavailable.
        Only meaningful in the 30–45°F range for mixed-precip detection.
        """
        if dewpoint_f is None:
            return None
        # Convert to Celsius for Stull formula
        t_c = (temp_f - 32) * 5 / 9
        rh = 100 * math.exp(
            (17.625 * (dewpoint_f - 32) * 5 / 9) / (243.04 + (dewpoint_f - 32) * 5 / 9)
            - (17.625 * t_c) / (243.04 + t_c)
        )
        rh = max(1.0, min(100.0, rh))
        # Stull wet-bulb approx (°C)
        wb_c = (
            t_c * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
            + math.atan(t_c + rh)
            - math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
            - 4.686035
        )
        return wb_c * 9 / 5 + 32

    def _score_snow(
        self,
        precip_chance: float,
        hourly: list,
        detailed_fc: str,
        alerts: list,
        moon: dict,
        low_temp: Optional[float],
    ) -> tuple[float, str]:
        """
        Conservative snow probability using:
          - sub-freezing + wet-bulb checks per hourly slot
          - mixed-precip penalty for marginal temperatures
          - multi-condition alignment gate (>40% requires ≥2 strong conditions)
          - lower fallback prior (3%)
        """
        snow_keywords = ["snow", "flurr", "blizzard", "winter storm", "sleet", "freezing"]
        accumulation_keywords = ["accumulation", "inches", "accumulating", "heavy snow"]
        has_snow_fc = any(kw in detailed_fc.lower() for kw in snow_keywords)
        has_accumulation_fc = any(kw in detailed_fc.lower() for kw in accumulation_keywords)
        snow_alert = any(
            "snow" in a["event"].lower() or "winter" in a["event"].lower()
            for a in alerts
        )

        # Count qualifying snow hours and detect mixed-precip risk
        snow_hours = 0
        mixed_precip_hours = 0
        very_cold_hours = 0  # temp < 30°F
        total_hours_checked = 0

        for h in hourly[:12]:
            temp = h.get("temp_f")
            precip = h.get("precip_chance", 0) or 0
            dewpoint = h.get("dewpoint_f")
            fc_text = (h.get("short_forecast") or "").lower()
            total_hours_checked += 1

            if temp is not None:
                if temp < 30 and precip >= 20:
                    snow_hours += 1
                    very_cold_hours += 1
                elif temp <= 34 and precip >= 20:
                    # Check wet-bulb for mixed-precip risk
                    wb = self._estimate_wet_bulb(temp, dewpoint)
                    if wb is not None and wb > 34.0:
                        # Wet-bulb above 34°F → likely rain or sleet, not accumulating snow
                        mixed_precip_hours += 1
                    else:
                        snow_hours += 1
                elif any(kw in fc_text for kw in snow_keywords) and precip >= 10:
                    snow_hours += 1

        # Base probability
        if snow_hours > 0 and total_hours_checked > 0:
            snow_hour_fraction = snow_hours / total_hours_checked
            base_prob = (precip_chance / 100.0) * snow_hour_fraction
        elif has_snow_fc:
            base_prob = precip_chance / 100.0 * 0.5
        else:
            base_prob = 0.03  # conservative prior — snow is rare

        # Mixed-precip penalty: marginal temps reduce snow confidence
        if mixed_precip_hours > 0 and snow_hours > 0:
            mixed_frac = mixed_precip_hours / (mixed_precip_hours + snow_hours)
            base_prob *= (1.0 - 0.40 * mixed_frac)  # up to -40% for all-mixed

        # Snow alert bonus
        if snow_alert:
            base_prob = min(1.0, base_prob + 0.15)

        # Multi-condition alignment gate: cap at 0.40 without ≥2 strong conditions
        strong_conditions = sum([
            snow_hours >= 3,
            snow_alert,
            has_accumulation_fc,
            very_cold_hours >= 3,
        ])
        if strong_conditions < 2:
            base_prob = min(0.40, base_prob)

        prob = min(1.0, max(0.01, base_prob))

        reasoning = (
            f"Snow keywords in forecast: {'yes' if has_snow_fc else 'no'}. "
            f"Accumulation language: {'yes' if has_accumulation_fc else 'no'}. "
            f"Precip chance: {precip_chance}%. "
            f"Snow-qualifying hours: {snow_hours}/{total_hours_checked} "
            f"(very cold: {very_cold_hours}, mixed-precip risk: {mixed_precip_hours}). "
            f"Strong conditions met: {strong_conditions}/4. "
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

