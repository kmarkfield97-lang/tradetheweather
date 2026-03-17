"""
Daily P&L tracker with weather-aware exit framework.

Exit logic philosophy:
  - Compare expected value of holding vs exiting at current market price
  - Classify each position as locked / near_locked / live / broken
  - Use staged profit-taking, progressive trailing stops, thesis invalidation
  - Staged portfolio brake on daily drawdown (-3% / -4% / -5%)
  - Never rely on a blanket time-based no-sell rule near settlement
"""

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from src.analysis.engine import SERIES_CITY_MAP

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
DAILY_FILE = os.path.join(DATA_DIR, "daily_state.json")

# ─── Daily account-level thresholds ──────────────────────────────────────────
PROFIT_TARGET_PCT = 0.05        # halt new trades at +5% day gain
DAILY_BRAKE_SOFT_PCT = 0.03     # -3%: disable new entries, tighten exits
DAILY_BRAKE_MEDIUM_PCT = 0.04   # -4%: reduce weaker / non-locked positions
DAILY_BRAKE_HARD_PCT = 0.05     # -5%: flatten all non-locked positions immediately
MAX_TRADES = 999999             # unlimited — cash reserve enforces real limit
MAX_POSITION_PCT = 0.20
MIN_CASH_RESERVE_PCT = 0.20

# ─── Staged profit-taking (fractional exits at price bands) ──────────────────
# Each entry: (min_cents, max_cents, trim_fraction)
# trim_fraction = portion of contracts to exit (0 = none, 1.0 = all)
PROFIT_TAKE_BANDS = [
    (70, 79, 0.25),   # 70–79¢: trim 25% if not locked
    (80, 89, 0.40),   # 80–89¢: trim 40% if not locked
    (90, 95, 0.65),   # 90–95¢: trim 65% unless locked
    (96, 99, 0.85),   # 96–99¢: exit 85% unless source mechanics secure outcome
]

# ─── Progressive trailing stop bands ─────────────────────────────────────────
# Arms after peak > entry + 25%. Tightens as peak rises.
# Each entry: (peak_min_cents, peak_max_cents, giveback_fraction)
# giveback_fraction = exit if mark < peak * (1 - giveback_fraction)
TRAILING_STOP_ARM_PCT = 0.25        # arm after +25% gain from entry
TRAILING_STOP_BANDS = [
    (40, 59, 0.22),   # peak 40–59¢: allow 22% giveback
    (60, 79, 0.16),   # peak 60–79¢: allow 16% giveback
    (80, 89, 0.10),   # peak 80–89¢: allow 10% giveback
    (90, 99, 0.05),   # peak 90¢+:    allow 5% giveback
]

# ─── Salvage stop (fail-safe only) ───────────────────────────────────────────
SALVAGE_STOP_PCT = 0.35         # exit if mark < 35% of entry (unrecoverable)

# ─── Settlement / fair-value parameters ──────────────────────────────────────
SETTLEMENT_RISK_BUFFER = 0.05   # subtract 5% from modeled prob for settlement uncertainty
SLIPPAGE_CENTS = 2              # estimated slippage cost in cents per contract
FEE_CENTS = 0                   # Kalshi fees (currently 0 for makers)
FINAL_HOUR_CONSERVATISM = 1.20  # multiply exit attractiveness by this in last hour

# ─── Thesis invalidation sensitivity ─────────────────────────────────────────
THESIS_TEMP_DIVERGENCE_F = 4.0  # °F: if current obs diverges this much, invalidate
THESIS_TREND_REVERSAL_F_PER_HR = -1.5  # °F/hr: sustained cooling triggers invalidation

# ─── Position state classification ───────────────────────────────────────────
STATE_LOCKED = "locked"
STATE_NEAR_LOCKED = "near_locked"
STATE_LIVE = "live"
STATE_BROKEN = "broken"

# ─── Exit reason constants ────────────────────────────────────────────────────
# Use these constants everywhere an exit reason is recorded so the classifier
# and history module can match against stable strings without substring hacks.
EXIT_THESIS_INVALIDATION = "thesis_invalidation"
EXIT_FAIR_VALUE          = "fair_value"
EXIT_TRAILING_STOP       = "trailing_stop"
EXIT_STAGED_PROFIT       = "staged_profit"
EXIT_SALVAGE             = "salvage"
EXIT_DAILY_HALT          = "daily_halt"
EXIT_EXPIRED             = "expired"

# ─── Fragile-trade price threshold ───────────────────────────────────────────
FRAGILE_LOW_PRICE_CENTS   = 20   # entries below this are flagged "low_price_entry"
FRAGILE_SAME_DAY_HOURS    = 6    # entries with <6h left flagged "same_day_entry"
FRAGILE_FINAL_HOURS_HOURS = 3    # entries with <3h left flagged "final_hours_entry"

# ─── Correlated exposure caps ─────────────────────────────────────────────────
# Fractions of starting_balance; enforced per city
CITY_TEMP_EXPOSURE_PCT = 0.15       # max 15% in temp markets for one city
CITY_PRECIP_EXPOSURE_PCT = 0.15     # max 15% in precip markets for one city
CITY_TOTAL_EXPOSURE_PCT = 0.20      # max 20% in any single city across all types
THRESHOLD_STACK_GAP_F = 3.0         # block new YES if existing YES within ±3°F same city/type


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_ticker(ticker: str) -> dict:
    """Extract market_type, threshold, is_bucket, and city prefix from ticker."""
    result = {"market_type": None, "threshold": None, "is_bucket": False, "city": None}
    t = ticker.upper()

    if "HIGHT" in t or "HIGH" in t:
        result["market_type"] = "temp_high"
    elif "LOWT" in t or "LOW" in t:
        result["market_type"] = "temp_low"
    elif "RAIN" in t:
        result["market_type"] = "rain"
    elif "SNOW" in t:
        result["market_type"] = "snow"

    m = re.search(r"-([TB])([\d.]+)(?:-|$)", t)
    if m:
        result["is_bucket"] = (m.group(1) == "B")
        result["threshold"] = float(m.group(2))

    return result


def _hours_to_settlement(close_time_str: Optional[str]) -> Optional[float]:
    """Returns hours remaining to settlement, or None if unparseable."""
    if not close_time_str:
        return None
    try:
        ct = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        return (ct - datetime.now(timezone.utc)).total_seconds() / 3600
    except Exception:
        return None


def _get_close_time(ticker: str, kalshi_client) -> Optional[str]:
    try:
        data = kalshi_client.get_market(ticker)
        m = data.get("market", data)
        return m.get("close_time") or m.get("expiration_time")
    except Exception:
        return None


def _is_near_settlement(ticker: str, kalshi_client, min_hours: float) -> bool:
    ct = _get_close_time(ticker, kalshi_client)
    hrs = _hours_to_settlement(ct)
    return hrs is not None and hrs < min_hours


def _trailing_stop_floor(peak: int) -> Optional[int]:
    """
    Returns the trailing stop floor price (cents) for a given peak price,
    or None if the trailing stop has not yet armed.
    """
    for band_min, band_max, giveback in TRAILING_STOP_BANDS:
        if band_min <= peak <= band_max:
            return round(peak * (1 - giveback))
    return None


def _staged_trim_fraction(mark: int, is_locked: bool) -> float:
    """
    Returns the fraction of contracts to trim at the current mark price.
    0 means hold, 1.0 means exit all.
    """
    for band_min, band_max, fraction in PROFIT_TAKE_BANDS:
        if band_min <= mark <= band_max:
            return 0.0 if is_locked else fraction
    return 0.0


def _classify_position(
    pos,
    mark: int,
    hours_left: Optional[float],
    weather_report: Optional[dict],
) -> str:
    """
    Classify a position as locked / near_locked / live / broken.

    locked:      outcome effectively secured (already hit threshold or clearly lost)
    near_locked: high market confidence with corroborating time/obs context
    live:        still depends on future weather path
    broken:      thesis invalidated by observations, trend, or timing
    """
    parsed = _parse_ticker(pos.ticker)
    threshold = parsed.get("threshold")
    market_type = parsed.get("market_type")

    if weather_report and market_type in ("temp_high", "temp_low") and threshold is not None:
        obs = weather_report.get("recent_observations", [])
        trend = weather_report.get("temp_trend")  # °F/hr

        if obs:
            current_temp = obs[0].get("temp_f")
            if current_temp is not None:
                gap = (
                    current_temp - threshold
                    if market_type == "temp_high"
                    else threshold - current_temp
                )

                # locked: threshold crossed AND either:
                #   (a) less than 2h left, or
                #   (b) crossed consistently across 3+ observations (early lock-in)
                if gap >= 0:
                    if hours_left is not None and hours_left < 2.0:
                        return STATE_LOCKED
                    # Check for persistent threshold crossing in multiple observations
                    if len(obs) >= 3:
                        temps = [o.get("temp_f") for o in obs[:3]]
                        if all(t is not None for t in temps):
                            all_crossed = all(
                                (t >= threshold if market_type == "temp_high" else t <= threshold)
                                for t in temps
                            )
                            if all_crossed:
                                return STATE_LOCKED

                # broken: temp is far from threshold with limited time
                if gap < -THESIS_TEMP_DIVERGENCE_F and hours_left is not None and hours_left < 3.0:
                    return STATE_BROKEN

                # broken: sustained trend moving strongly against us
                if (trend is not None and
                        trend <= THESIS_TREND_REVERSAL_F_PER_HR and
                        gap < 0 and
                        hours_left is not None and hours_left < 4.0):
                    return STATE_BROKEN

                # broken: required warming rate is physically unrealistic (>5°F/hr needed)
                if gap < 0 and hours_left is not None and hours_left > 0:
                    required_rate = (-gap) / hours_left
                    if required_rate > 5.0:
                        return STATE_BROKEN

    # near_locked: final window with very high/low market price
    if hours_left is not None and hours_left < 1.0:
        if mark >= 90 or mark <= 10:
            return STATE_NEAR_LOCKED

    # near_locked: high market confidence + time context
    # Require hours_left < 3 or a large gap past threshold to avoid premature classification
    if mark >= 88:
        if hours_left is None or hours_left < 3.0:
            return STATE_NEAR_LOCKED
        # With > 3h left, require obs confirmation for near_locked
        if weather_report and market_type in ("temp_high", "temp_low") and threshold is not None:
            obs = weather_report.get("recent_observations", [])
            if obs:
                cur = obs[0].get("temp_f")
                if cur is not None:
                    gap = (
                        cur - threshold
                        if market_type == "temp_high"
                        else threshold - cur
                    )
                    if gap >= 5.0:
                        return STATE_NEAR_LOCKED
        # No obs confirmation with > 3h left — stay as LIVE
        return STATE_LIVE

    if mark <= 12:
        if hours_left is None or hours_left < 3.0:
            return STATE_NEAR_LOCKED
        return STATE_LIVE

    # ── Rain/snow: thesis invalidation via precipitation timing slip ──────────
    # If we're long YES on rain/snow but hourly precip has shifted outside the
    # settlement window, the thesis may be broken.
    if weather_report and market_type == "rain" and pos.side == "yes":
        hourly = weather_report.get("hourly", [])
        if hourly and hours_left is not None and hours_left < 4.0:
            # Check if meaningful precip (>= 20%) exists within the window
            window_hours = max(1, int(hours_left))
            window_probs = [
                h.get("precip_chance", 0) or 0
                for h in hourly[:window_hours]
            ]
            max_in_window = max(window_probs) if window_probs else 0
            # Check if precip that was expected has slipped beyond window
            extended_probs = [
                h.get("precip_chance", 0) or 0
                for h in hourly[window_hours:window_hours + 4]
            ]
            max_extended = max(extended_probs) if extended_probs else 0
            if max_in_window < 20 and max_extended >= 40:
                # Precipitation slipped outside settlement window — thesis broken
                logger.debug(
                    f"RAIN_THESIS_SLIP {pos.ticker}: max precip in window {max_in_window}% "
                    f"but {max_extended}% after window — broken"
                )
                return STATE_BROKEN

    return STATE_LIVE


def _model_hold_value(
    pos,
    mark: int,
    state: str,
    hours_left: Optional[float],
    weather_report: Optional[dict],
) -> float:
    """
    Estimates the expected value (cents) of holding to settlement.

    Penalty terms (applied to implied_prob before converting to cents):
      1. Settlement risk buffer (dynamic by market type)
      2. Model uncertainty penalty (from entry-time signal aggregation)
      3. Final-window volatility penalty (for live positions near close)
      4. Required warming rate penalty (temp markets)

    Returns EV in cents.
    """
    if state == STATE_LOCKED:
        # Functionally resolved — hold value ≈ 100 minus minimal settlement risk
        return 100 - (0.02 * 100)  # 2% residual settlement uncertainty

    if state == STATE_BROKEN:
        # Thesis broken — hold value ≈ 0
        return 2.0  # 2¢ residual salvage value

    # Base: use current market price as unbiased estimate of settlement prob
    implied_prob = mark / 100.0

    parsed = _parse_ticker(pos.ticker)
    market_type = parsed.get("market_type")
    threshold = parsed.get("threshold")

    # ── 1. Settlement risk buffer (market-type aware) ────────────────────────
    # Rain/snow settlement has higher uncertainty (NWS gauge vs Kalshi station)
    if market_type in ("rain", "snow"):
        base_risk_buffer = 0.08
    else:
        base_risk_buffer = 0.04

    # Final window adds additional uncertainty from last-mile station reporting
    if hours_left is not None and hours_left < 2.0:
        risk_buffer = base_risk_buffer * FINAL_HOUR_CONSERVATISM
    else:
        risk_buffer = base_risk_buffer

    # ── 2. Model uncertainty penalty (stored at entry time) ──────────────────
    model_unc = getattr(pos, "model_uncertainty", 0.3)
    uncertainty_penalty = 0.05 * model_unc  # up to 0.05 reduction at max uncertainty

    # ── 3. Final-window volatility penalty for live positions ─────────────────
    volatility_penalty = 0.0
    if state == STATE_LIVE and hours_left is not None and hours_left < 1.5:
        volatility_penalty = 0.02  # 2¢ — late live positions have higher noise

    # ── 4. Weather trend adjustment for temp markets ──────────────────────────
    if weather_report and market_type in ("temp_high", "temp_low") and threshold is not None:
        obs = weather_report.get("recent_observations", [])
        trend = weather_report.get("temp_trend")
        if obs and hours_left is not None:
            current_temp = obs[0].get("temp_f")
            if current_temp is not None:
                gap = (
                    current_temp - threshold
                    if market_type == "temp_high"
                    else threshold - current_temp
                )
                if gap >= 0:
                    # Threshold already crossed — apply trend penalty if cooling against us
                    if trend is not None:
                        if (market_type == "temp_high" and trend < -1.0) or \
                                (market_type == "temp_low" and trend > 1.0):
                            implied_prob = max(0.05, implied_prob - 0.08)
                else:
                    warming_needed = -gap
                    hrs_remaining = max(hours_left, 0.1)
                    required_rate = warming_needed / hrs_remaining
                    if required_rate > 3.0:
                        implied_prob = max(0.02, implied_prob - 0.15)
                    elif required_rate > 1.5:
                        implied_prob = max(0.05, implied_prob - 0.07)

    total_penalty_pct = risk_buffer + uncertainty_penalty + volatility_penalty
    adjusted_prob = max(0.01, min(0.99, implied_prob * (1.0 - total_penalty_pct)))

    logger.debug(
        f"hold_ev {pos.ticker}: mark={mark}¢ state={state} "
        f"implied={implied_prob:.3f} risk_buf={risk_buffer:.3f} "
        f"unc_pen={uncertainty_penalty:.3f} vol_pen={volatility_penalty:.3f} "
        f"→ adjusted={adjusted_prob:.3f} ev={adjusted_prob * 100:.1f}¢"
    )

    return adjusted_prob * 100


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    order_id: str
    side: str
    contracts: int
    entry_price: int            # cents
    cost_dollars: float
    status: str                 # "open" / "closed" / "expired"
    pnl_dollars: float = 0.0
    exit_price: Optional[int] = None
    placed_at: str = ""
    high_water_mark: Optional[int] = None   # peak mark-to-market price seen (MFE proxy, cents)
    low_water_mark: Optional[int] = None    # worst mark-to-market price seen (MAE proxy, cents)
    trimmed_contracts: int = 0              # contracts already exited via staged trim
    city: str = ""                          # city key (e.g. "NYC", "DENVER")
    market_type: str = ""                   # "temp_high" / "temp_low" / "rain" / "snow"
    model_uncertainty: float = 0.3          # from signal aggregation at entry time (0–1)
    exit_reason: str = ""                   # stable exit reason category (EXIT_* constant)
    # ── Entry-time decision snapshot ─────────────────────────────────────────
    # Populated once at trade execution from the TradeRecommendation.entry_context.
    # Never mutated after entry. Used by post-trade diagnostics.
    entry_our_prob: Optional[float] = None         # final adjusted probability
    entry_base_prob: Optional[float] = None        # base prob before signal adjustment
    entry_signal_adj: Optional[float] = None       # clamped signal adjustment applied
    entry_edge: Optional[float] = None             # edge at decision time (0–1)
    entry_sigma: Optional[float] = None            # dynamic sigma (°F) at entry
    entry_hours_left: Optional[float] = None       # hours to settlement at entry
    entry_spread: Optional[int] = None             # bid-ask spread in cents at entry
    entry_regime: str = ""                         # cap_regime from signal aggregator
    entry_signal_breakdown: list = field(default_factory=list)  # per-signal list
    entry_weights_version: str = ""                # signal_weights.json version tag
    entry_liquidity_dollars: Optional[float] = None  # executable liquidity at entry
    # ── Fragile-trade flags ───────────────────────────────────────────────────
    fragile_flags: list = field(default_factory=list)  # e.g. ["low_price_entry"]


@dataclass
class DailyState:
    date: str
    starting_balance: float
    current_balance: float
    trades_placed: int = 0
    positions: list[Position] = field(default_factory=list)
    trading_halted: bool = False
    halt_reason: str = ""
    goal_met: bool = False
    realized_pnl: float = 0.0
    daily_brake_level: int = 0   # 0=none, 1=soft(-3%), 2=medium(-4%), 3=hard(-5%)
    goal_exception_trades: int = 0  # trades allowed past +5% goal (max 2/day)


# ─── PnLTracker ──────────────────────────────────────────────────────────────

class PnLTracker:
    def __init__(self, kalshi_client=None, weather_pipeline=None):
        self.kalshi = kalshi_client
        self.weather = weather_pipeline
        os.makedirs(DATA_DIR, exist_ok=True)
        self.state = self._load_or_init()

    # ── State management ──────────────────────────────────────────────────────

    def _load_or_init(self) -> DailyState:
        today = date.today().isoformat()
        if os.path.exists(DAILY_FILE):
            with open(DAILY_FILE) as f:
                data = json.load(f)
            if data.get("date") == today:
                known_pos = {k for k in Position.__dataclass_fields__}
                known_state = {k for k in DailyState.__dataclass_fields__}
                positions = [
                    Position(**{k: v for k, v in p.items() if k in known_pos})
                    for p in data.get("positions", [])
                ]
                state_data = {k: v for k, v in data.items() if k in known_state and k != "positions"}
                return DailyState(**state_data, positions=positions)
        starting_balance = self._fetch_balance()
        state = DailyState(
            date=today,
            starting_balance=starting_balance,
            current_balance=starting_balance,
        )
        self._save(state)
        return state

    def _save(self, state: DailyState = None):
        state = state or self.state
        with open(DAILY_FILE, "w") as f:
            json.dump(asdict(state), f, indent=2)

    def _fetch_balance(self) -> float:
        if self.kalshi:
            try:
                return self.kalshi.get_balance()
            except Exception:
                pass
        return 50.0

    # ── Rule checks & daily brakes ────────────────────────────────────────────

    def refresh_balance(self):
        """Sync current balance from Kalshi and evaluate daily brakes."""
        self.state.current_balance = self._fetch_balance()
        self._check_rules()
        self._save()

    def _portfolio_value(self) -> float:
        """
        Returns total portfolio value: cash balance + cost basis of open positions.
        Open position dollars are tied up in contracts, not reflected in cash balance,
        so we add them back to get a true picture of account value.
        """
        open_cost = sum(
            p.cost_dollars for p in self.state.positions if p.status == "open"
        )
        return self.state.current_balance + open_cost

    def _check_rules(self):
        starting = self.state.starting_balance
        current = self._portfolio_value()
        if starting <= 0:
            return

        # Guard: if account is too small to trade meaningfully, disable entries
        # but do NOT treat percentage swings as meaningful profit targets.
        if starting < 5.0:
            if not self.state.trading_halted:
                self.state.trading_halted = True
                self.state.halt_reason = (
                    f"Account balance ${starting:.2f} too small to trade safely (minimum $5)"
                )
            return

        pnl_pct = (current - starting) / starting

        # Daily profit target — use realized_pnl only to avoid false triggers from
        # Kalshi balance timing (available balance can lag behind order placement,
        # causing _portfolio_value to double-count open position cost).
        realized_pnl_pct = self.state.realized_pnl / starting if starting > 0 else 0
        if realized_pnl_pct >= PROFIT_TARGET_PCT and not self.state.goal_met:
            self.state.goal_met = True
            self.state.trading_halted = True
            self.state.halt_reason = f"Daily profit target reached (+{realized_pnl_pct * 100:.1f}%)"
            return

        if self.state.trading_halted:
            return

        # Staged brake system
        if pnl_pct <= -DAILY_BRAKE_HARD_PCT and self.state.daily_brake_level < 3:
            self.state.daily_brake_level = 3
            self.state.trading_halted = True
            self.state.halt_reason = f"Daily stop loss triggered ({pnl_pct * 100:.1f}%)"
            self._save()
            self.trigger_stop_loss(locked_ok=True)

        elif pnl_pct <= -DAILY_BRAKE_MEDIUM_PCT and self.state.daily_brake_level < 2:
            self.state.daily_brake_level = 2
            self.state.trading_halted = True
            self.state.halt_reason = f"Daily brake medium: new entries disabled ({pnl_pct * 100:.1f}%)"
            logger.warning(f"Daily brake MEDIUM: {pnl_pct * 100:.1f}% drawdown — tightening exits")
            self._save()

        elif pnl_pct <= -DAILY_BRAKE_SOFT_PCT and self.state.daily_brake_level < 1:
            self.state.daily_brake_level = 1
            self.state.trading_halted = True
            self.state.halt_reason = f"Daily brake soft: new entries disabled ({pnl_pct * 100:.1f}%)"
            logger.warning(f"Daily brake SOFT: {pnl_pct * 100:.1f}% drawdown — new entries disabled")
            self._save()

    def can_trade(self) -> tuple[bool, str]:
        if self.state.trading_halted:
            return False, self.state.halt_reason
        # Reserve is anchored to starting_balance, not current_balance.
        # (Using current_balance * 0.20 would make the check always False.)
        reserve = self.state.starting_balance * MIN_CASH_RESERVE_PCT
        if self.state.current_balance <= reserve:
            return False, f"Balance ${self.state.current_balance:.2f} at or below 20% cash reserve (${reserve:.2f})"
        return True, ""

    def is_high_conviction_exception(self, rec) -> bool:
        """
        Returns True if a trade qualifies for a narrow bypass past the daily profit halt.
        Requirements (all must hold):
          - goal_met = True, daily_brake_level = 0 (positive day, no drawdown)
          - Under 2 exception trades already used today
          - rec.confidence == "high" and rec.edge >= 0.18
          - No existing open position in the same city + market_type (no correlation)
        """
        if not self.state.goal_met:
            return False
        if self.state.daily_brake_level > 0:
            return False
        if self.state.goal_exception_trades >= 2:
            return False

        confidence = getattr(rec, "confidence", "") or ""
        edge = getattr(rec, "edge", 0.0) or 0.0
        if confidence != "high" or edge < 0.18:
            return False

        city = getattr(rec, "city", "") or ""
        mtype = getattr(rec, "market_type", "") or ""
        for pos in self.state.positions:
            if pos.status == "open" and pos.city == city and pos.market_type == mtype:
                return False  # correlated position already open

        return True

    def record_goal_exception(self):
        """Increment exception trade counter and persist."""
        self.state.goal_exception_trades += 1
        self._save()

    def validate_position_size(self, cost_dollars: float) -> tuple[bool, str]:
        max_allowed = self.state.starting_balance * MAX_POSITION_PCT
        if cost_dollars > max_allowed:
            return False, f"Position size ${cost_dollars:.2f} exceeds 20% limit (${max_allowed:.2f})"
        return True, ""

    def get_exposure_summary(self) -> dict:
        """
        Returns per-city exposure breakdown across open positions.
        Used by check_correlation_limits() to enforce correlated exposure caps.
        """
        city_temp: dict[str, float] = {}
        city_precip: dict[str, float] = {}
        city_thresholds: dict[str, list] = {}  # city:market_type -> [threshold, ...]

        for pos in self.state.positions:
            if pos.status != "open":
                continue
            city = pos.city or ""
            mtype = pos.market_type or ""
            cost = pos.cost_dollars

            if mtype in ("temp_high", "temp_low"):
                city_temp[city] = city_temp.get(city, 0.0) + cost
            elif mtype in ("rain", "snow"):
                city_precip[city] = city_precip.get(city, 0.0) + cost

            # Track thresholds for stack-exposure check
            parsed = _parse_ticker(pos.ticker)
            thresh = parsed.get("threshold")
            if thresh is not None and city and mtype:
                key = f"{city}:{mtype}"
                city_thresholds.setdefault(key, []).append((thresh, pos.side))

        return {
            "city_temp_exposure": city_temp,
            "city_precip_exposure": city_precip,
            "city_thresholds": city_thresholds,
        }

    def check_correlation_limits(self, rec) -> tuple[bool, str]:
        """
        Enforces correlated exposure caps before entering a new position.
        rec must have: .city, .market_type, .cost_dollars, .side, and
        a .market_price attribute (used for threshold-stack check via .ticker).
        Returns (allowed: bool, reason: str).
        """
        city = getattr(rec, "city", "") or ""
        mtype = getattr(rec, "market_type", "") or ""
        cost = getattr(rec, "cost_dollars", 0.0)
        side = getattr(rec, "side", "yes")
        base = self.state.starting_balance
        if base <= 0:
            return True, ""

        exp = self.get_exposure_summary()

        # City-level temperature cap
        if mtype in ("temp_high", "temp_low"):
            current_temp = exp["city_temp_exposure"].get(city, 0.0)
            limit = base * CITY_TEMP_EXPOSURE_PCT
            if current_temp + cost > limit:
                return False, (
                    f"city temp exposure limit: {city} would be "
                    f"${current_temp + cost:.2f} > ${limit:.2f} cap"
                )

        # City-level precipitation cap
        if mtype in ("rain", "snow"):
            current_precip = exp["city_precip_exposure"].get(city, 0.0)
            limit = base * CITY_PRECIP_EXPOSURE_PCT
            if current_precip + cost > limit:
                return False, (
                    f"city precip exposure limit: {city} would be "
                    f"${current_precip + cost:.2f} > ${limit:.2f} cap"
                )

        # Total city exposure cap (temp + precip combined)
        total_temp = exp["city_temp_exposure"].get(city, 0.0)
        total_precip = exp["city_precip_exposure"].get(city, 0.0)
        total_city = total_temp + total_precip + cost
        total_limit = base * CITY_TOTAL_EXPOSURE_PCT
        if total_city > total_limit:
            return False, (
                f"total city exposure limit: {city} would be "
                f"${total_city:.2f} > ${total_limit:.2f} cap"
            )

        # Threshold-stack check: block new YES if already long a YES on same city/type
        # within ±THRESHOLD_STACK_GAP_F of this threshold
        if side == "yes" and mtype in ("temp_high", "temp_low"):
            rec_parsed = _parse_ticker(getattr(rec, "ticker", ""))
            new_thresh = rec_parsed.get("threshold")
            if new_thresh is not None:
                key = f"{city}:{mtype}"
                for existing_thresh, existing_side in exp["city_thresholds"].get(key, []):
                    if (existing_side == "yes" and
                            abs(existing_thresh - new_thresh) < THRESHOLD_STACK_GAP_F):
                        return False, (
                            f"threshold stack: {city} {mtype} already long YES "
                            f"@{existing_thresh}°F within {THRESHOLD_STACK_GAP_F}°F of "
                            f"new @{new_thresh}°F"
                        )

        return True, ""

    # ── Per-position exit evaluation ──────────────────────────────────────────

    def check_profit_takes(self) -> list[tuple["Position", int, int, str]]:
        """
        Evaluates all open positions for exit conditions.
        Returns list of (position, exit_price_cents, contracts_to_exit, exit_reason).

        Exit decision framework (in priority order):
        1. Thesis invalidation (broken state)
        2. Daily brake medium: reduce non-locked positions
        3. Staged profit-taking in price bands
        4. Progressive trailing stop
        5. Fair-value comparison (hold EV vs exit EV)
        6. Salvage stop (fail-safe)
        """
        if not self.kalshi:
            return []

        exits = []

        for pos in self.state.positions:
            if pos.status != "open":
                continue

            remaining = pos.contracts - pos.trimmed_contracts
            if remaining <= 0:
                continue

            # ── Get current market price ──────────────────────────────────────
            try:
                liquidity = self.kalshi.get_liquidity(pos.ticker)
            except Exception as e:
                logger.debug(f"Liquidity fetch failed for {pos.ticker}: {e}")
                continue

            mark = liquidity.get("best_yes_price" if pos.side == "yes" else "best_no_price")
            if not mark or mark <= 0:
                logger.debug(f"No liquid exit for {pos.ticker} {pos.side}")
                continue

            # ── Settlement timing ─────────────────────────────────────────────
            close_time_str = _get_close_time(pos.ticker, self.kalshi)
            hours_left = _hours_to_settlement(close_time_str)

            # ── Weather data (best-effort) ────────────────────────────────────
            weather_report = None
            if self.weather:
                try:
                    parsed = _parse_ticker(pos.ticker)
                    city = self._infer_city(pos.ticker)
                    if city:
                        weather_report = self.weather.get_full_report(city)
                except Exception:
                    pass

            # ── Update high_water_mark (MFE) and low_water_mark (MAE) ─────────
            changed = False
            if pos.high_water_mark is None or mark > pos.high_water_mark:
                pos.high_water_mark = mark
                changed = True
            if pos.low_water_mark is None or mark < pos.low_water_mark:
                pos.low_water_mark = mark
                changed = True
            if changed:
                self._save()

            hwm = pos.high_water_mark if pos.high_water_mark is not None else pos.entry_price

            # ── Classify position state ───────────────────────────────────────
            state = _classify_position(pos, mark, hours_left, weather_report)

            # ── Fair value comparison ─────────────────────────────────────────
            hold_ev = _model_hold_value(pos, mark, state, hours_left, weather_report)
            exit_ev = mark - SLIPPAGE_CENTS - FEE_CENTS

            # Apply final-window conservatism (make hold less attractive vs exiting).
            # Discount hold EV in final window for live positions to favor exiting.
            if hours_left is not None and hours_left < 0.5 and state == STATE_LIVE:
                hold_ev = hold_ev * 0.80   # very final window: strongly discount hold
            elif hours_left is not None and hours_left < 1.0 and state not in (STATE_LOCKED, STATE_NEAR_LOCKED):
                hold_ev = hold_ev / FINAL_HOUR_CONSERVATISM
            exit_ev_adj = exit_ev

            # ── Detailed decision log ─────────────────────────────────────────
            hrs_display = f"{hours_left:.1f}" if hours_left is not None else "N/A"
            logger.info(
                f"EXIT_EVAL {pos.ticker} {pos.side.upper()} | "
                f"entry={pos.entry_price}¢ mark={mark}¢ peak={hwm}¢ | "
                f"state={state} hrs_left={hrs_display} | "
                f"hold_ev={hold_ev:.1f}¢ exit_ev={exit_ev_adj:.1f}¢ | "
                f"pnl=${((mark - pos.entry_price) / 100 * remaining):+.2f} contracts={remaining}"
            )

            exit_contracts = 0
            exit_reason = ""

            # ── Priority 1: Thesis invalidation (broken) ──────────────────────
            if state == STATE_BROKEN:
                exit_contracts = remaining
                exit_reason = EXIT_THESIS_INVALIDATION

            # ── Priority 2: Daily brake medium → reduce non-locked ────────────
            elif self.state.daily_brake_level >= 2 and state not in (STATE_LOCKED, STATE_NEAR_LOCKED):
                exit_contracts = remaining
                exit_reason = EXIT_DAILY_HALT

            # ── Priority 3: Staged profit-taking ─────────────────────────────
            elif mark >= PROFIT_TAKE_BANDS[0][0]:  # at or above lowest band (70¢)
                is_locked = state in (STATE_LOCKED, STATE_NEAR_LOCKED)
                # near_locked: suppress trimming at 70–95¢ so position can run to
                # full settlement. Only the 96–99¢ band trim is allowed to reduce
                # last-mile slippage risk.
                if state == STATE_NEAR_LOCKED and mark <= 95:
                    trim_frac = 0.0
                else:
                    trim_frac = _staged_trim_fraction(mark, is_locked)
                if trim_frac > 0:
                    trim_count = max(1, math.floor(remaining * trim_frac))
                    # Only trim if we haven't already trimmed this band
                    already_trimmed_frac = pos.trimmed_contracts / max(pos.contracts, 1)
                    if trim_frac > already_trimmed_frac + 0.05:
                        exit_contracts = trim_count
                        exit_reason = EXIT_STAGED_PROFIT

            # ── Priority 4: Progressive trailing stop ─────────────────────────
            if exit_contracts == 0 and hwm >= pos.entry_price * (1 + TRAILING_STOP_ARM_PCT):
                floor = _trailing_stop_floor(hwm)
                if floor is not None and mark < floor:
                    exit_contracts = remaining
                    exit_reason = EXIT_TRAILING_STOP

            # ── Priority 5: Fair-value exit ───────────────────────────────────
            if exit_contracts == 0 and exit_ev_adj >= hold_ev and state not in (STATE_LOCKED,):
                exit_contracts = remaining
                exit_reason = EXIT_FAIR_VALUE

            # ── Priority 6: Salvage stop (fail-safe) ─────────────────────────
            if exit_contracts == 0:
                salvage_price = round(pos.entry_price * SALVAGE_STOP_PCT)
                if mark < salvage_price:
                    exit_contracts = remaining
                    exit_reason = EXIT_SALVAGE

            if exit_contracts > 0 and exit_reason:
                # Clamp to remaining
                exit_contracts = min(exit_contracts, remaining)
                logger.info(
                    f"EXIT DECISION: {pos.ticker} {pos.side.upper()} "
                    f"exit {exit_contracts}/{pos.contracts} @ {mark}¢ | reason={exit_reason}"
                )
                exits.append((pos, mark, exit_contracts, exit_reason))

        return exits

    def _infer_city(self, ticker: str) -> Optional[str]:
        """Infer city name from ticker prefix using engine's series map."""
        try:
            t = ticker.upper()
            for prefix, city in SERIES_CITY_MAP.items():
                if t.startswith(prefix):
                    return city
        except Exception:
            pass
        return None

    # ── Trade recording ───────────────────────────────────────────────────────

    def record_trade(self, ticker: str, order_id: str, side: str,
                     contracts: int, entry_price: int, cost_dollars: float,
                     city: str = "", market_type: str = "",
                     model_uncertainty: float = 0.3,
                     entry_context: Optional[dict] = None):
        """
        Records a new trade entry. entry_context is the dict attached to
        TradeRecommendation containing the full decision snapshot (probabilities,
        signal breakdown, sigma, spread, regime, etc.).
        """
        ctx = entry_context or {}

        # ── Fragile-trade flag computation ────────────────────────────────────
        fragile_flags: list[str] = []
        hours_left = ctx.get("hours_left")
        if entry_price > 0 and entry_price < FRAGILE_LOW_PRICE_CENTS:
            fragile_flags.append("low_price_entry")
        if hours_left is not None:
            if hours_left < FRAGILE_FINAL_HOURS_HOURS:
                fragile_flags.append("final_hours_entry")
            elif hours_left < FRAGILE_SAME_DAY_HOURS:
                fragile_flags.append("same_day_entry")
        if ctx.get("extreme_disagreement"):
            fragile_flags.append("model_market_disagreement")
        if ctx.get("model_uncertainty", model_uncertainty) > 0.6:
            fragile_flags.append("high_model_uncertainty")

        pos = Position(
            ticker=ticker,
            order_id=order_id,
            side=side,
            contracts=contracts,
            entry_price=entry_price,
            cost_dollars=cost_dollars,
            status="open",
            placed_at=datetime.now(timezone.utc).isoformat(),
            city=city,
            market_type=market_type,
            model_uncertainty=model_uncertainty,
            # ── Entry snapshot ────────────────────────────────────────────────
            entry_our_prob=ctx.get("our_prob"),
            entry_base_prob=ctx.get("base_prob"),
            entry_signal_adj=ctx.get("signal_adj"),
            entry_edge=ctx.get("edge"),
            entry_sigma=ctx.get("sigma_f"),
            entry_hours_left=hours_left,
            entry_spread=ctx.get("spread"),
            entry_regime=ctx.get("cap_regime", ""),
            entry_signal_breakdown=ctx.get("signal_breakdown", []),
            entry_weights_version=ctx.get("weights_version", ""),
            entry_liquidity_dollars=ctx.get("exec_liq"),
            fragile_flags=fragile_flags,
        )
        self.state.positions.append(pos)
        self.state.trades_placed += 1
        self._save()

    def record_exit(self, order_id: str, exit_price: int, pnl_dollars: float,
                    contracts_exited: Optional[int] = None,
                    exit_reason: str = ""):
        """
        Records an exit. If contracts_exited < pos.contracts, records a partial exit
        by updating trimmed_contracts. Full exit sets status to 'closed'.
        exit_reason should be one of the EXIT_* constants defined above.
        """
        for pos in self.state.positions:
            if pos.order_id == order_id and pos.status == "open":
                total = pos.contracts
                exited = contracts_exited if contracts_exited is not None else total
                pos.trimmed_contracts = min(total, pos.trimmed_contracts + exited)
                pos.pnl_dollars += pnl_dollars
                pos.exit_price = exit_price
                if exit_reason:
                    pos.exit_reason = exit_reason
                self.state.realized_pnl += pnl_dollars
                if pos.trimmed_contracts >= total:
                    pos.status = "closed"
                break
        self._save()

    # ── Stop loss ─────────────────────────────────────────────────────────────

    def trigger_stop_loss(self, locked_ok: bool = False) -> list[str]:
        """
        Exit all open positions. If locked_ok=True, allows locked positions to remain.
        Returns list of tickers exited.
        """
        if not self.kalshi:
            return []

        exited = []
        for pos in self.state.positions:
            if pos.status != "open":
                continue

            remaining = pos.contracts - pos.trimmed_contracts
            if remaining <= 0:
                continue

            # If locked_ok, check state and skip locked positions
            if locked_ok:
                try:
                    liq = self.kalshi.get_liquidity(pos.ticker)
                    mark = liq.get("best_yes_price" if pos.side == "yes" else "best_no_price") or 50
                    ct = _get_close_time(pos.ticker, self.kalshi)
                    hrs = _hours_to_settlement(ct)
                    state = _classify_position(pos, mark, hrs, None)
                    if state == STATE_LOCKED:
                        logger.info(f"Stop loss: skipping locked position {pos.ticker}")
                        continue
                except Exception:
                    pass  # can't determine state — exit anyway

            try:
                liquidity = self.kalshi.get_liquidity(pos.ticker)
                if pos.side == "yes":
                    exit_price = liquidity.get("best_yes_price")
                else:
                    exit_price = liquidity.get("best_no_price")

                if not exit_price:
                    logger.warning(f"Stop loss: no exit price for {pos.ticker}")
                    continue

                self.kalshi.exit_position(pos.ticker, pos.side, remaining, exit_price)

                if pos.side == "yes":
                    pnl = (exit_price / 100 - pos.entry_price / 100) * remaining
                else:
                    pnl = (pos.entry_price / 100 - exit_price / 100) * remaining

                self.record_exit(pos.order_id, exit_price, pnl, remaining,
                                 exit_reason=EXIT_DAILY_HALT)
                exited.append(pos.ticker)
                logger.info(f"Stop loss exit: {pos.ticker} {pos.side.upper()} {remaining}x @ {exit_price}¢ pnl=${pnl:+.2f}")

            except Exception as e:
                logger.error(f"Failed to exit {pos.ticker} on stop loss: {e}")

        self._save()
        return exited

    # ── Daily summary ─────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        starting = self.state.starting_balance
        current = self._portfolio_value()
        pnl = current - starting
        pnl_pct = pnl / starting * 100 if starting > 0 else 0

        open_positions = [p for p in self.state.positions if p.status == "open"]

        return {
            "date": self.state.date,
            "starting_balance": f"${starting:.2f}",
            "current_balance": f"${current:.2f}",
            "pnl": f"${pnl:+.2f}",
            "pnl_pct": f"{pnl_pct:+.1f}%",
            "trades_placed": self.state.trades_placed,
            "trades_remaining": "unlimited",
            "open_positions": len(open_positions),
            "realized_pnl": f"${self.state.realized_pnl:+.2f}",
            "goal_met": self.state.goal_met,
            "trading_halted": self.state.trading_halted,
            "halt_reason": self.state.halt_reason,
            "daily_brake_level": self.state.daily_brake_level,
            "profit_target": f"${starting * (1 + PROFIT_TARGET_PCT):.2f} (+5%)",
            "stop_loss_level": f"${starting * (1 - DAILY_BRAKE_HARD_PCT):.2f} (-5%)",
        }

    def format_summary(self) -> str:
        s = self.get_summary()
        if s["goal_met"]:
            status = "GOAL MET - Trading paused"
        elif s["trading_halted"]:
            status = f"HALTED: {s['halt_reason']}"
        elif s["daily_brake_level"] > 0:
            status = f"BRAKE LVL {s['daily_brake_level']} — {s['halt_reason']}"
        else:
            status = "Active"

        return (
            f"Daily P&L Summary — {s['date']}\n"
            f"Balance: {s['current_balance']} ({s['pnl']} / {s['pnl_pct']})\n"
            f"Target: {s['profit_target']} | Floor: {s['stop_loss_level']}\n"
            f"Trades: {s['trades_placed']} | Open positions: {s['open_positions']}\n"
            f"Realized P&L: {s['realized_pnl']}\n"
            f"Status: {status}"
        )
