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
from src.analysis import operating_profile as op_profile

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

# ─── Entry-relative trim gates ───────────────────────────────────────────────
# A trim band fires only if the position has gained at least this much from entry.
# Prevents aggressive de-risking on tiny gains (e.g. entered at 74¢, mark 76¢).
#
# Structure: (band_min_cents, min_gain_pct, min_gain_cents)
#   band_min_cents  – lowest cent of the trim band (matches PROFIT_TAKE_BANDS)
#   min_gain_pct    – unrealised gain % (from entry) required to allow full trim
#   min_gain_cents  – unrealised gain in cents required to allow full trim
#
# Both thresholds must be met for a full trim; if only one is met the trim
# fraction is halved.  If neither is met the trim is suppressed entirely.
TRIM_ENTRY_GATES = [
    (70, 0.10, 5),   # 70–79¢ band: need ≥10% gain AND ≥5¢ above entry
    (80, 0.15, 8),   # 80–89¢ band: need ≥15% gain AND ≥8¢ above entry
    (90, 0.20, 10),  # 90–95¢ band: need ≥20% gain AND ≥10¢ above entry
    (96, 0.25, 12),  # 96–99¢ band: need ≥25% gain AND ≥12¢ above entry
]

# Minimum net exit value (after slippage) that must exceed entry price for a
# trim to lock in any real profit at all.  Trims that would merely return
# capital at cost are suppressed.
TRIM_MIN_NET_PROFIT_CENTS = 1       # exit_net must beat entry by at least 1¢

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

# ─── Adverse-excursion stop (NO positions) ───────────────────────────────────
# For NO positions, the mark (YES price) can rise against us indefinitely while
# the fair-value model continues to hold because hold_ev still slightly exceeds
# exit_ev.  This stop adds a hard floor: if the YES price has risen more than
# ADVERSE_STOP_PCT × entry_price above entry, exit regardless of hold_ev.
# Example: entry_NO=26¢, ADVERSE_STOP_PCT=0.50 → exit when YES > 39¢.
# Does NOT require a prior gain; arms immediately after entry.
# Not gated by fair-value grace period (it is an emergency exit).
ADVERSE_STOP_PCT = 0.50         # exit NO if mark > entry + 50% of entry (adverse move)

# ─── Salvage stop (fail-safe only) ───────────────────────────────────────────
SALVAGE_STOP_PCT = 0.35         # exit if mark < 35% of entry (unrecoverable)

# ─── Settlement / fair-value parameters ──────────────────────────────────────
SETTLEMENT_RISK_BUFFER = 0.05   # subtract 5% from modeled prob for settlement uncertainty
SLIPPAGE_CENTS = 2              # estimated slippage cost in cents per contract
FEE_CENTS = 0                   # Kalshi fees (currently 0 for makers)
FINAL_HOUR_CONSERVATISM = 1.20  # multiply exit attractiveness by this in last hour

# ─── Fair-value exit grace period ────────────────────────────────────────────
# After entering a position, suppress the fair-value exit for this many minutes.
# Rationale: hold_ev is anchored to the current market price and discounts it
# by a risk buffer.  On a freshly entered position the market price IS our
# entry price, so hold_ev < entry_price by construction, which makes exit_ev
# appear attractive even though we just took the trade because our estimated
# probability is higher than the market's.  The grace period gives the market
# time to reprice toward our estimate before allowing a fair-value-only exit.
# Emergency exits (thesis invalidation, trailing stop, salvage) are NOT gated.
FAIR_VALUE_GRACE_MINUTES = 30

# ─── Thesis invalidation sensitivity ─────────────────────────────────────────
THESIS_TEMP_DIVERGENCE_F = 4.0  # °F: if current obs diverges this much, invalidate
THESIS_TREND_REVERSAL_F_PER_HR = -1.5  # °F/hr: sustained cooling triggers invalidation

# ─── Position state classification ───────────────────────────────────────────
STATE_LOCKED = "locked"
STATE_NEAR_LOCKED = "near_locked"
STATE_LIVE = "live"
STATE_BROKEN = "broken"
STATE_STALLED = "stalled"          # capital trap: poor EV, poor liquidity, no catalyst

# ─── Capital trap / stalled position thresholds ───────────────────────────────
STALL_MIN_AGE_MINUTES = 45          # position must be at least this old to be flagged
STALL_HOLD_EV_CEILING = 45.0        # hold EV below this (¢) raises concern when <2h left
STALL_EXIT_EV_CEILING = 35.0        # exit EV below this (¢) raises concern
STALL_SPREAD_WIDE_CENTS = 10        # spread at or above this is "wide"
STALL_POOR_LIQUIDITY_DOLLARS = 2.0  # total book depth below this is "poor"
STALL_MFE_REQUIRED_CENTS = 3        # if HWM never exceeded entry+this, no favorable excursion
STALL_SCORE_THRESHOLD = 3           # flags needed (out of 7) to be classified stalled
STALL_EXIT_MIN_MARK_CENTS = 10      # forced EXIT_STALLED requires mark ≥ this (avoids nuking near-worthless positions)
# Time-aware hold EV ceiling: with >2h left the market is still live; only apply
# the EV ceiling when time is short enough that the mark is likely anchored.
STALL_HOLD_EV_HOURS_THRESHOLD = 2.0  # only penalise hold EV when <2h left
# Minimum consecutive stall cycles before escalating from alert to forced action
STALL_ESCALATION_CYCLES = 3          # flag 3× in a row before treating as urgent
# Minimum stall cycles to re-alert on Telegram (suppress duplicate noise)
STALL_ALERT_EVERY_N_CYCLES = 2       # alert on first detection, then every 2 cycles

# ─── Exit reason constants ────────────────────────────────────────────────────
# Use these constants everywhere an exit reason is recorded so the classifier
# and history module can match against stable strings without substring hacks.
EXIT_THESIS_INVALIDATION = "thesis_invalidation"
EXIT_FAIR_VALUE          = "fair_value"
EXIT_TRAILING_STOP       = "trailing_stop"
EXIT_STAGED_PROFIT       = "staged_profit"
EXIT_SALVAGE             = "salvage"
EXIT_ADVERSE_STOP        = "adverse_excursion_stop"  # NO position moved too far against entry
EXIT_DAILY_HALT          = "daily_halt"
EXIT_EXPIRED             = "expired"
EXIT_STALLED             = "stalled_capital_trap"  # position is a capital trap with no catalyst

# ─── Fragile-trade price threshold ───────────────────────────────────────────
FRAGILE_LOW_PRICE_CENTS   = 20   # entries below this are flagged "low_price_entry"
FRAGILE_SAME_DAY_HOURS    = 6    # entries with <6h left flagged "same_day_entry"
FRAGILE_FINAL_HOURS_HOURS = 3    # entries with <3h left flagged "final_hours_entry"

# ─── Second-session trading window ────────────────────────────────────────────
# When the overnight temp_low session triggers a halt (soft/medium brake), the
# bot can still execute a limited number of trades in a separate later session
# (e.g., afternoon temp_high markets) provided the bars below are cleared.
#
# These constants control when and how much the second window is allowed.
# The hard -5% global brake is NEVER bypassed by second-session logic.
SECOND_SESSION_MAX_TRADES   = 1      # max trades allowed in the second session
SECOND_SESSION_MIN_HOUR_PT  = 10     # earliest PT hour for second session (10am = temp_high window)
SECOND_SESSION_MIN_EDGE     = 0.18   # minimum 18¢ edge required for second session
SECOND_SESSION_BUDGET_PCT   = 0.05   # max 5% of starting_balance per second-session trade

# ─── Correlated exposure caps ─────────────────────────────────────────────────
# Fractions of starting_balance; enforced per city
CITY_TEMP_EXPOSURE_PCT = 0.15       # max 15% in temp markets for one city
CITY_PRECIP_EXPOSURE_PCT = 0.15     # max 15% in precip markets for one city
CITY_TOTAL_EXPOSURE_PCT = 0.20      # max 20% in any single city across all types
THRESHOLD_STACK_GAP_F = 3.0         # block new YES if existing YES within ±3°F same city/type

# ─── Weather thesis / conflict-detection policy ───────────────────────────────
# Conservative default: for the same event key (city + market_type + settlement_date),
# only allow a new position if it is directionally consistent with all existing
# open positions AND all already-selected candidates in the current scan cycle.
#
# Directional opinion derived from (side, threshold, is_bucket):
#   YES on T>=X   → bullish_above(X):  believes outcome >= X
#   NO  on T>=X   → bearish_below(X):  believes outcome < X
#   YES on B[X-Y] → inside_bucket(X,Y): believes outcome in [X,Y)
#   NO  on B[X-Y] → outside_bucket(X,Y): believes outcome NOT in [X,Y)
#
# Conflict rules:
#   bullish_above(A) vs bearish_below(B) where A and B are within CONFLICT_THRESH_F → CONFLICT
#   bullish_above(A) vs bullish_above(B) — same direction, stacking — allowed unless within STACK_GAP
#   bearish_below(A) vs bearish_below(B) — same direction, stacking — allowed
#   inside_bucket vs bullish_above(X) where X falls inside bucket → partial conflict
#
# Two threshold positions conflict if their implied opinions are incompatible:
#   YES@T68 (believes high >=68) + NO@T72 (believes high <72) → contradictory
#   when the thresholds are close (within CONFLICT_THRESH_F) they directly contradict.
#   When far apart they may ladder reinforcing views, but we apply conservative blocking.
CONFLICT_THRESH_F = 8.0     # threshold distance within which opposing directions conflict
SAME_EVENT_POLICY = "allow_reinforcing_only"   # conservative default


def _make_event_key(city: str, market_type: str, settlement_date: str) -> str:
    """Canonical event key for grouping positions on the same underlying weather outcome."""
    return f"{city}|{market_type}|{settlement_date}"


def _weather_thesis(
    side: str,
    threshold: Optional[float],
    is_bucket: bool,
) -> dict:
    """
    Derive a normalized directional opinion from a trade's parameters.

    Returns a dict:
      {
        "direction": "bullish" | "bearish" | "inside_bucket" | "outside_bucket" | "unknown",
        "threshold": float | None,
        "bucket_low": float | None,   # only for bucket markets
        "bucket_high": float | None,  # only for bucket markets
        "side": "yes" | "no",
      }

    Direction semantics for threshold markets:
      YES on T>=X  → bullish:  we believe the outcome will be >= X
      NO  on T>=X  → bearish:  we believe the outcome will be  < X

    For bucket markets:
      YES on B[X-Y] → inside_bucket: we believe outcome ∈ [X, Y)
      NO  on B[X-Y] → outside_bucket: we believe outcome ∉ [X, Y)
    """
    s = (side or "").lower()
    if threshold is None:
        return {"direction": "unknown", "threshold": None,
                "bucket_low": None, "bucket_high": None, "side": s}

    if is_bucket:
        direction = "inside_bucket" if s == "yes" else "outside_bucket"
        return {"direction": direction, "threshold": threshold,
                "bucket_low": threshold, "bucket_high": None, "side": s}
    else:
        direction = "bullish" if s == "yes" else "bearish"
        return {"direction": direction, "threshold": threshold,
                "bucket_low": None, "bucket_high": None, "side": s}


def _theses_conflict(a: dict, b: dict) -> tuple[bool, str]:
    """
    Determine whether two weather theses on the same event key conflict.
    Returns (conflicts: bool, reason: str).

    Conflict logic:
    1. bullish(A) vs bearish(B):
       - A == B → direct contradiction (YES@T70 + NO@T70 is theoretically impossible
                 but treat as conflict if same ticker is re-evaluated)
       - |A - B| < CONFLICT_THRESH_F → near-threshold contradiction
         Example: YES@T70 (believes >=70) + NO@T72 (believes <72) → hidden contradiction
         because if high ends up 70 or 71, both are at risk of loss simultaneously
       - A > B → structural contradiction: YES@T72 (>=72) + NO@T70 (<70) → impossible
         to win both (requires 70 <= high < 70 or high >= 72 — no overlap)
         Actually: YES@T72 wins if high>=72; NO@T70 wins if high<70. These never both win.
       - A < B by >= CONFLICT_THRESH_F → reinforcing ladder: YES@T68 + NO@T75
         We think high is somewhere 68-75; this is a strangle-ish position, allowed.
    2. bearish(A) vs bearish(B): same direction, allowed (both believe high < some threshold)
    3. bullish(A) vs bullish(B): same direction, allowed (both believe high >= some threshold)
       [Note: THRESHOLD_STACK_GAP_F check in check_correlation_limits still applies]
    4. inside_bucket vs bullish: conflict if bullish threshold falls inside bucket range
    5. inside_bucket vs bearish: conflict if bearish threshold falls inside bucket range
    6. Two unknown theses: no conflict (we don't have enough info to judge)
    """
    dir_a = a.get("direction", "unknown")
    dir_b = b.get("direction", "unknown")
    thresh_a = a.get("threshold")
    thresh_b = b.get("threshold")

    if dir_a == "unknown" or dir_b == "unknown":
        return False, ""

    # Same direction — not a conflict
    if dir_a == dir_b:
        return False, ""

    # bullish vs bearish (or vice versa)
    if set([dir_a, dir_b]) == {"bullish", "bearish"}:
        bull_thresh = thresh_a if dir_a == "bullish" else thresh_b
        bear_thresh = thresh_a if dir_a == "bearish" else thresh_b
        if bull_thresh is None or bear_thresh is None:
            return False, ""

        # Direct: YES@T70 + NO@T70 → same threshold, opposite sides
        if abs(bull_thresh - bear_thresh) < 0.01:
            return True, (
                f"direct contradiction: YES@T{bull_thresh:.0f} vs NO@T{bear_thresh:.0f} "
                f"(same threshold, opposite sides)"
            )

        # Structural: bullish threshold > bearish threshold
        # YES@T72 + NO@T70 → can never both win (high must be both >=72 AND <70 — impossible)
        if bull_thresh > bear_thresh:
            return True, (
                f"structural contradiction: YES@T{bull_thresh:.0f} (>=) vs NO@T{bear_thresh:.0f} (<) "
                f"— gap={bull_thresh - bear_thresh:.1f}°F, can never both win"
            )

        # Near-threshold: |bull - bear| < CONFLICT_THRESH_F
        # YES@T70 + NO@T72: believes high >=70 AND <72 — narrow band, hidden contradiction
        gap = bear_thresh - bull_thresh
        if gap < CONFLICT_THRESH_F:
            return True, (
                f"near-threshold contradiction: YES@T{bull_thresh:.0f} vs NO@T{bear_thresh:.0f} "
                f"gap={gap:.1f}°F < {CONFLICT_THRESH_F}°F conflict threshold"
            )

        # Reinforcing ladder: bull_thresh < bear_thresh by >= CONFLICT_THRESH_F
        # YES@T68 + NO@T76: we believe high is between 68 and 76 — coherent strangle
        return False, ""

    # inside_bucket vs bullish/bearish
    if "inside_bucket" in (dir_a, dir_b) or "outside_bucket" in (dir_a, dir_b):
        # Conservative: treat any bucket vs threshold combination as potential conflict
        # unless we have clear evidence they're reinforcing. Block by default.
        return True, (
            f"bucket vs threshold conflict: {dir_a}@T{thresh_a} vs {dir_b}@T{thresh_b} "
            f"— mixed market types on same event; conservative block"
        )

    return False, ""


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
    Returns the base trim fraction for the current mark price band.
    0 means hold, 1.0 means exit all.
    Does NOT apply entry-relative gates — callers use _entry_relative_trim_fraction.
    """
    for band_min, band_max, fraction in PROFIT_TAKE_BANDS:
        if band_min <= mark <= band_max:
            return 0.0 if is_locked else fraction
    return 0.0


def _entry_gate_for_band(mark: int) -> Optional[tuple]:
    """Returns (min_gain_pct, min_gain_cents) for the band containing mark, or None."""
    # Resolve which PROFIT_TAKE_BAND mark falls in, then look up its gate.
    for pb_min, pb_max, _ in PROFIT_TAKE_BANDS:
        if pb_min <= mark <= pb_max:
            for gate_band_min, min_pct, min_cents in TRIM_ENTRY_GATES:
                if gate_band_min == pb_min:
                    return (min_pct, min_cents)
            return None  # band exists but no gate defined
    return None  # mark not in any band


def _entry_relative_trim_fraction(
    mark: int,
    entry_price: int,
    is_locked: bool,
) -> tuple[float, str]:
    """
    Returns (adjusted_trim_fraction, gate_reason) using both the current mark
    band and entry-relative profitability gates.

    Gate logic:
      - Both min_gain_pct AND min_gain_cents must be met → full base fraction
      - Only one met → half the base fraction (partial trim)
      - Neither met → suppress trim (0.0)
      - Net exit value (mark - slippage - fees) must beat entry by TRIM_MIN_NET_PROFIT_CENTS

    Returns (trim_fraction, reason_string) where reason_string explains the decision.
    """
    base_frac = _staged_trim_fraction(mark, is_locked)
    if base_frac == 0.0:
        return 0.0, "no_band_or_locked"

    # Net profit check: exit must lock in real profit after slippage
    exit_net = mark - SLIPPAGE_CENTS - FEE_CENTS
    if exit_net <= entry_price + TRIM_MIN_NET_PROFIT_CENTS - 1:
        return 0.0, f"no_net_profit(exit_net={exit_net}¢ entry={entry_price}¢)"

    gate = _entry_gate_for_band(mark)
    if gate is None:
        # No gate defined for this band — use base fraction as-is
        return base_frac, "no_gate_defined"

    min_gain_pct, min_gain_cents = gate
    gain_cents = mark - entry_price
    gain_pct = gain_cents / entry_price if entry_price > 0 else 0.0

    pct_ok = gain_pct >= min_gain_pct
    cents_ok = gain_cents >= min_gain_cents

    if pct_ok and cents_ok:
        return base_frac, f"gate_ok(gain={gain_cents}¢/{gain_pct:.1%})"
    elif pct_ok or cents_ok:
        half = base_frac / 2.0
        which = f"pct={'ok' if pct_ok else 'fail'}({gain_pct:.1%}>={min_gain_pct:.1%}) cents={'ok' if cents_ok else 'fail'}({gain_cents}>={min_gain_cents})"
        return half, f"gate_partial({which}) → half_trim={half:.2f}"
    else:
        return 0.0, f"gate_fail(gain={gain_cents}¢/{gain_pct:.1%} need>={min_gain_cents}¢/{min_gain_pct:.1%})"


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
    # ── Quality tier (from engine ranking) ───────────────────────────────────
    trade_tier: str = "standard"   # "top_tier" / "standard" / "marginal"
    # ── Cached market metadata ────────────────────────────────────────────────
    close_time: Optional[str] = None  # cached from Kalshi API; avoids repeated calls every 5 min
    # ── Settlement / conflict-detection fields ────────────────────────────────
    settlement_date: str = ""          # ISO date (YYYY-MM-DD) from close_time at entry
    threshold: Optional[float] = None  # numeric threshold (°F) parsed from ticker
    is_bucket: bool = False            # True if this is a bucket (B) market


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
    pending_buy_dollars: float = 0.0  # capital reserved by resting buy orders (updated by orchestrator)
    stall_alert_counts: dict = field(default_factory=dict)  # ticker -> consecutive stall cycles seen
    # ── Session-aware halt tracking ───────────────────────────────────────────
    # Records which market types were open when a soft/medium brake fired.
    # Used to determine whether a "second session" trade is eligible.
    halt_market_types: list = field(default_factory=list)  # e.g. ["temp_low"]
    second_session_trades: int = 0  # trades taken in second-session window today


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
        starting_balance = self._fetch_portfolio_value()
        # Kalshi's portfolio_value field can lag at midnight while settlement
        # proceeds are posting.  If the value looks implausibly low, wait and retry.
        if starting_balance < 5.0:
            import time as _time
            logger.warning(
                f"Starting balance ${starting_balance:.2f} looks implausibly low — "
                f"waiting 30s for settlement to post, then retrying"
            )
            _time.sleep(30)
            starting_balance = self._fetch_portfolio_value()
            logger.info(f"Retry starting balance: ${starting_balance:.2f}")
        state = DailyState(
            date=today,
            starting_balance=starting_balance,
            current_balance=self._fetch_balance(),
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

    def _fetch_portfolio_value(self) -> float:
        """Returns total portfolio value: cash + mark value of open positions."""
        if self.kalshi:
            try:
                return self.kalshi.get_portfolio_value()
            except Exception:
                pass
        return self._fetch_balance()

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
            # Hard brake clears any second-session allowance — global halt, no exceptions
            self.state.halt_market_types = ["temp_low", "temp_high", "rain", "snow"]
            self._save()
            self.trigger_stop_loss(locked_ok=True)

        elif pnl_pct <= -DAILY_BRAKE_MEDIUM_PCT and self.state.daily_brake_level < 2:
            self.state.daily_brake_level = 2
            self.state.trading_halted = True
            self.state.halt_reason = f"Daily brake medium: new entries disabled ({pnl_pct * 100:.1f}%)"
            # Record which market types caused the halt so second-session logic can filter
            if not self.state.halt_market_types:
                self.state.halt_market_types = list({
                    p.market_type for p in self.state.positions
                    if p.status == "open" and p.market_type
                })
            logger.warning(
                f"Daily brake MEDIUM: {pnl_pct * 100:.1f}% drawdown — tightening exits | "
                f"halt_market_types={self.state.halt_market_types}"
            )
            self._save()

        elif pnl_pct <= -DAILY_BRAKE_SOFT_PCT and self.state.daily_brake_level < 1:
            self.state.daily_brake_level = 1
            self.state.trading_halted = True
            self.state.halt_reason = f"Daily brake soft: new entries disabled ({pnl_pct * 100:.1f}%)"
            # Record which market types caused the halt
            if not self.state.halt_market_types:
                self.state.halt_market_types = list({
                    p.market_type for p in self.state.positions
                    if p.status == "open" and p.market_type
                })
            logger.warning(
                f"Daily brake SOFT: {pnl_pct * 100:.1f}% drawdown — new entries disabled | "
                f"halt_market_types={self.state.halt_market_types}"
            )
            self._save()

    def get_effective_deployable_capital(self) -> dict:
        """
        Returns a conservative accounting of capital that is genuinely free to deploy
        into new positions.

        effective_available =
            confirmed_cash_balance
            - reserve_floor  (anchored to starting_balance, never shrinks)
            - pending_buy_dollars  (resting buy orders not yet filled)

        The open-position cost basis is NOT subtracted here because it was already
        spent from cash when the orders executed (Kalshi deducts immediately).
        pending_buy_dollars covers orders placed but not yet filled.

        Returns a dict with all components for structured logging.
        """
        cash = self.state.current_balance
        reserve = self.state.starting_balance * MIN_CASH_RESERVE_PCT
        pending_buys = max(0.0, self.state.pending_buy_dollars)
        open_cost = sum(
            p.cost_dollars for p in self.state.positions if p.status == "open"
        )

        # Conservative: subtract pending buy commitments from usable cash.
        # (open_cost is already reflected in cash deductions from Kalshi.)
        effective = cash - reserve - pending_buys

        return {
            "cash_balance": cash,
            "reserve_floor": reserve,
            "pending_buy_dollars": pending_buys,
            "open_position_cost": open_cost,
            "effective_available": effective,
        }

    def can_trade(self) -> tuple[bool, str]:
        if self.state.trading_halted:
            return False, self.state.halt_reason

        cap = self.get_effective_deployable_capital()
        cash = cap["cash_balance"]
        reserve = cap["reserve_floor"]
        pending = cap["pending_buy_dollars"]
        effective = cap["effective_available"]
        open_cost = cap["open_position_cost"]

        logger.info(
            f"CAN_TRADE_CHECK | cash=${cash:.2f} reserve=${reserve:.2f} "
            f"pending_buys=${pending:.2f} open_pos_cost=${open_cost:.2f} "
            f"effective_available=${effective:.2f} | halted={self.state.trading_halted}"
        )

        # Primary gate: raw cash must clear the reserve floor even before pending
        if cash <= reserve:
            reason = (
                f"Cash ${cash:.2f} at or below 20% reserve floor ${reserve:.2f}"
            )
            logger.info(f"CAN_TRADE=NO | {reason}")
            return False, reason

        # Secondary gate: after subtracting pending buy commitments, must still be positive
        if effective <= 0:
            reason = (
                f"No deployable capital after reserve (${reserve:.2f}) "
                f"and pending buys (${pending:.2f}); cash=${cash:.2f}"
            )
            logger.info(f"CAN_TRADE=NO | {reason}")
            return False, reason

        logger.info(f"CAN_TRADE=YES | effective_available=${effective:.2f}")
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

    def can_trade_second_session(self, rec) -> tuple[bool, str]:
        """
        Returns (allowed, reason) for a second-session trade exception.

        The second session allows a limited number of trades in a different market
        family (temp_high, rain) after the overnight temp_low session has triggered
        a soft or medium daily brake.

        Hard rules that always apply:
          - Hard brake (-5%, daily_brake_level == 3): NEVER allowed
          - Global halt from profit target: NOT allowed (goal_met uses its own gate)
          - Max SECOND_SESSION_MAX_TRADES per day
          - Only temp_high and rain market types eligible
          - Market type must NOT be in halt_market_types (wasn't the session that halted)
          - Time gate: must be past SECOND_SESSION_MIN_HOUR_PT in PT
          - Edge gate: >= SECOND_SESSION_MIN_EDGE
          - Confidence must be "high"
          - Budget gate: cost <= SECOND_SESSION_BUDGET_PCT of starting_balance
        """
        from zoneinfo import ZoneInfo

        if not self.state.trading_halted:
            return False, "not halted — use normal can_trade()"

        # Hard brake blocks everything — no exceptions
        if self.state.daily_brake_level >= 3:
            return False, "hard stop-loss active — no second session allowed"

        # Profit target halt uses its own exception mechanism
        if self.state.goal_met:
            return False, "goal_met — use is_high_conviction_exception() instead"

        # Second session trade cap
        if self.state.second_session_trades >= SECOND_SESSION_MAX_TRADES:
            return False, f"second session limit reached ({SECOND_SESSION_MAX_TRADES} trade)"

        market_type = getattr(rec, "market_type", "") or ""
        if not market_type:
            return False, "unknown market_type"

        # Only temp_high and rain qualify for second session
        if market_type not in ("temp_high", "rain"):
            return False, f"market_type={market_type} not eligible for second session"

        # Block if this market type was part of the session that caused the halt
        if market_type in self.state.halt_market_types:
            return False, (
                f"market_type={market_type} was in halted session "
                f"(halt_market_types={self.state.halt_market_types})"
            )

        # Time gate: temp_high/rain second session only after morning hours
        try:
            pt_now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/Los_Angeles"))
            if pt_now.hour < SECOND_SESSION_MIN_HOUR_PT:
                return False, (
                    f"before second session window "
                    f"(PT hour={pt_now.hour}, need >={SECOND_SESSION_MIN_HOUR_PT})"
                )
        except Exception:
            return False, "could not determine PT time for second session gate"

        # Edge and confidence gate (higher bar than normal)
        edge = getattr(rec, "edge", 0.0) or 0.0
        confidence = getattr(rec, "confidence", "") or ""
        if edge < SECOND_SESSION_MIN_EDGE:
            return False, (
                f"edge={edge:.3f} < {SECOND_SESSION_MIN_EDGE:.3f} second session minimum"
            )
        if confidence != "high":
            return False, f"confidence={confidence} must be 'high' for second session"

        # Budget gate: strict cap on second session exposure
        cost = getattr(rec, "cost_dollars", 0.0) or 0.0
        budget = self.state.starting_balance * SECOND_SESSION_BUDGET_PCT
        if cost > budget:
            return False, (
                f"cost ${cost:.2f} > second session budget ${budget:.2f} "
                f"({SECOND_SESSION_BUDGET_PCT:.0%} of starting_balance)"
            )

        return True, (
            f"second session approved: market_type={market_type} "
            f"edge={edge:.3f} confidence={confidence} cost=${cost:.2f} "
            f"PT_hour={pt_now.hour}"
        )

    def record_second_session_trade(self):
        """Increment second session trade counter and persist."""
        self.state.second_session_trades += 1
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
        city_thresholds: dict[str, list] = {}  # city:market_type -> [(threshold, side), ...]
        # event_theses: event_key -> [(thesis_dict, ticker), ...]
        # Used by the directional conflict check.
        event_theses: dict[str, list] = {}

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

            # Track thresholds for stack-exposure check (legacy path, kept for YES-YES gap check)
            parsed = _parse_ticker(pos.ticker)
            thresh = parsed.get("threshold")
            if thresh is not None and city and mtype:
                key = f"{city}:{mtype}"
                city_thresholds.setdefault(key, []).append((thresh, pos.side))

            # Build weather thesis for directional conflict detection
            # Prefer the threshold stored on the position; fall back to ticker parse.
            pos_thresh = pos.threshold if pos.threshold is not None else thresh
            pos_is_bucket = pos.is_bucket if hasattr(pos, "is_bucket") else parsed.get("is_bucket", False)
            if city and mtype:
                ekey = _make_event_key(city, mtype, pos.settlement_date or "")
                thesis = _weather_thesis(pos.side, pos_thresh, pos_is_bucket)
                event_theses.setdefault(ekey, []).append((thesis, pos.ticker))

        return {
            "city_temp_exposure": city_temp,
            "city_precip_exposure": city_precip,
            "city_thresholds": city_thresholds,
            "event_theses": event_theses,
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

        # ── Directional conflict check vs open positions ───────────────────────
        # Check if the new trade's weather thesis conflicts with any existing open
        # position on the same event key (city + market_type + settlement_date).
        rec_settlement = getattr(rec, "settlement_date", "") or ""
        rec_threshold = getattr(rec, "threshold", None)
        rec_is_bucket = getattr(rec, "is_bucket", False)

        # Fall back to ticker parse if the recommendation doesn't carry these fields
        if rec_threshold is None:
            _rec_parsed = _parse_ticker(getattr(rec, "ticker", ""))
            rec_threshold = _rec_parsed.get("threshold")
            rec_is_bucket = _rec_parsed.get("is_bucket", False)

        new_thesis = _weather_thesis(side, rec_threshold, rec_is_bucket)
        if new_thesis["direction"] != "unknown" and city and mtype:
            ekey = _make_event_key(city, mtype, rec_settlement)
            for existing_thesis, existing_ticker in exp.get("event_theses", {}).get(ekey, []):
                conflicts, conflict_reason = _theses_conflict(new_thesis, existing_thesis)
                if conflicts:
                    logger.info(
                        f"CONFLICT_BLOCK {getattr(rec, 'ticker', '?')} "
                        f"city={city} type={mtype} date={rec_settlement} "
                        f"new_side={side} new_thresh={rec_threshold} "
                        f"existing={existing_ticker} — {conflict_reason}"
                    )
                    return False, (
                        f"directional conflict with open position {existing_ticker}: "
                        f"{conflict_reason}"
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
            if not pos.close_time:
                pos.close_time = _get_close_time(pos.ticker, self.kalshi)
            hours_left = _hours_to_settlement(pos.close_time)

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

            # ── Entry-relative metrics (used in trim gate + logging) ──────────
            gain_cents = mark - pos.entry_price
            gain_pct = gain_cents / pos.entry_price if pos.entry_price > 0 else 0.0
            exit_net = mark - SLIPPAGE_CENTS - FEE_CENTS

            # Determine which trim band the current mark falls in (for logging)
            _trim_band_label = "none"
            for _bmin, _bmax, _bfrac in PROFIT_TAKE_BANDS:
                if _bmin <= mark <= _bmax:
                    _trim_band_label = f"{_bmin}-{_bmax}¢({_bfrac:.0%}base)"
                    break

            # ── Detailed decision log ─────────────────────────────────────────
            hrs_display = f"{hours_left:.1f}" if hours_left is not None else "N/A"
            logger.info(
                f"EXIT_EVAL {pos.ticker} {pos.side.upper()} | "
                f"entry={pos.entry_price}¢ mark={mark}¢ peak={hwm}¢ | "
                f"gain={gain_cents:+d}¢ ({gain_pct:+.1%}) exit_net={exit_net}¢ | "
                f"trim_band={_trim_band_label} state={state} hrs_left={hrs_display} | "
                f"hold_ev={hold_ev:.1f}¢ exit_ev={exit_ev_adj:.1f}¢ | "
                f"pnl=${(gain_cents / 100 * remaining):+.2f} contracts={remaining}"
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
                    gate_reason = "near_locked_suppressed"
                else:
                    trim_frac, gate_reason = _entry_relative_trim_fraction(
                        mark, pos.entry_price, is_locked
                    )

                logger.info(
                    f"TRIM_GATE {pos.ticker} | "
                    f"entry={pos.entry_price}¢ mark={mark}¢ band={_trim_band_label} | "
                    f"gain={gain_cents:+d}¢ ({gain_pct:+.1%}) exit_net={exit_net}¢ | "
                    f"hold_ev={hold_ev:.1f}¢ exit_ev={exit_ev_adj:.1f}¢ | "
                    f"trim_frac={trim_frac:.2f} gate={gate_reason}"
                )

                if trim_frac > 0:
                    trim_count = max(1, math.floor(remaining * trim_frac))
                    # Only trim if we haven't already trimmed this band
                    already_trimmed_frac = pos.trimmed_contracts / max(pos.contracts, 1)
                    if trim_frac > already_trimmed_frac + 0.05:
                        exit_contracts = trim_count
                        exit_reason = EXIT_STAGED_PROFIT
                    else:
                        logger.info(
                            f"TRIM_SKIP {pos.ticker}: already trimmed "
                            f"{pos.trimmed_contracts}/{pos.contracts} "
                            f"({already_trimmed_frac:.1%}) — "
                            f"trim_frac={trim_frac:.2f} not > already+5%"
                        )

            # ── Priority 4: Progressive trailing stop ─────────────────────────
            if exit_contracts == 0 and hwm >= pos.entry_price * (1 + TRAILING_STOP_ARM_PCT):
                floor = _trailing_stop_floor(hwm)
                if floor is not None and mark < floor:
                    exit_contracts = remaining
                    exit_reason = EXIT_TRAILING_STOP

            # ── Priority 4.5: Adverse-excursion stop (NO positions only) ──────
            # The trailing stop only arms after a gain; it never fires on a
            # losing NO position.  This fills the gap: if YES has risen more
            # than ADVERSE_STOP_PCT × entry_NO above our entry, the market is
            # moving against the thesis and we cut the loss before it compounds.
            # Not gated by the fair-value grace period (emergency exit).
            if exit_contracts == 0 and pos.side == "no":
                adverse_threshold = pos.entry_price + round(pos.entry_price * ADVERSE_STOP_PCT)
                if mark >= adverse_threshold:
                    exit_contracts = remaining
                    exit_reason = EXIT_ADVERSE_STOP
                    logger.warning(
                        f"ADVERSE_STOP {pos.ticker} NO | "
                        f"entry={pos.entry_price}¢ mark={mark}¢ "
                        f"threshold={adverse_threshold}¢ "
                        f"(entry + {ADVERSE_STOP_PCT:.0%}) — cutting adverse position"
                    )

            # ── Priority 5: Fair-value exit ───────────────────────────────────
            # Suppress for FAIR_VALUE_GRACE_MINUTES after entry: hold_ev is
            # anchored to the current market price and discounts it by a risk
            # buffer, so on a fresh position exit_ev trivially exceeds hold_ev
            # even though we entered because our estimated prob is higher than
            # the market's.  Emergency exits (1–4 and 6) are NOT gated.
            #
            # Quality-based tolerance: top-tier positions receive additional
            # grace time (profile-driven) before a fair-value exit can fire.
            # This avoids exiting strong, well-supported positions that are
            # simply waiting for the market to reprice toward our estimate.
            _fv_grace_ok = True
            if exit_contracts == 0 and exit_ev_adj >= hold_ev and state not in (STATE_LOCKED,):
                # Determine effective grace period
                _fv_hq, _fv_hq_reason = self._is_high_quality_hold(pos, state, hours_left)
                _fv_quality_bonus = (
                    op_profile.get_param("quality_fv_grace_minutes_bonus")
                    if _fv_hq else 0
                )
                _effective_grace = FAIR_VALUE_GRACE_MINUTES + _fv_quality_bonus

                if pos.placed_at:
                    try:
                        placed_dt = datetime.fromisoformat(pos.placed_at.replace("Z", "+00:00"))
                        age_minutes_fv = (datetime.now(timezone.utc) - placed_dt).total_seconds() / 60
                        if age_minutes_fv < _effective_grace:
                            _fv_grace_ok = False
                            if _fv_quality_bonus > 0:
                                logger.info(
                                    f"FV_QUALITY_GRACE {pos.ticker}: suppressing fair-value exit — "
                                    f"high_quality_hold=True grace={_effective_grace}min "
                                    f"(base={FAIR_VALUE_GRACE_MINUTES} bonus=+{_fv_quality_bonus}) "
                                    f"age={age_minutes_fv:.0f}min "
                                    f"exit_ev={exit_ev_adj:.1f}¢ hold_ev={hold_ev:.1f}¢ | "
                                    f"reason={_fv_hq_reason}"
                                )
                            else:
                                logger.info(
                                    f"FV_GRACE {pos.ticker}: suppressing fair-value exit — "
                                    f"position is only {age_minutes_fv:.0f}min old "
                                    f"(grace={_effective_grace}min) "
                                    f"exit_ev={exit_ev_adj:.1f}¢ hold_ev={hold_ev:.1f}¢"
                                )
                    except Exception:
                        pass  # parse failure → allow exit (safe default)
                if _fv_grace_ok:
                    exit_contracts = remaining
                    exit_reason = EXIT_FAIR_VALUE

            # ── Priority 6: Capital-trap / stalled exit ────────────────────────
            # Triggered when a position has been flagged as urgently stalled for
            # >= effective_escalation_threshold consecutive balance-refresh cycles.
            # Bypasses the fair-value grace period intentionally: a stalled position
            # that has been stuck for many cycles with poor EV should be exited even
            # if it is still young (entry price may already be above fair value).
            # Only fires if there is a real exit price available (mark > 0 is already
            # confirmed above) and the position has an exit EV above a minimal floor.
            #
            # Quality-based holding tolerance: top-tier positions with a strong thesis
            # receive extra stall cycles (profile-driven) before forced exit, so the
            # bot doesn't prematurely close strong positions that are just quiet.
            if exit_contracts == 0:
                stall_cycles = self.state.stall_alert_counts.get(pos.ticker, 0)
                # Compute effective escalation threshold (quality bonus applied here too)
                _hq_for_stall, _hq_stall_reason = self._is_high_quality_hold(
                    pos, state, hours_left
                )
                _quality_bonus = (
                    op_profile.get_param("quality_stall_cycle_bonus")
                    if _hq_for_stall else 0
                )
                _eff_esc_threshold = STALL_ESCALATION_CYCLES + _quality_bonus
                if stall_cycles >= _eff_esc_threshold:
                    _stall_exit_ev = mark - SLIPPAGE_CENTS - FEE_CENTS
                    # Require a meaningful mark price (≥ STALL_EXIT_MIN_MARK_CENTS) so we
                    # don't force-exit a position that is still trading at a non-trivial
                    # level just because exit EV clears the 1¢ floor (P1-2 fix).
                    if _stall_exit_ev > 1 and mark >= STALL_EXIT_MIN_MARK_CENTS:
                        exit_contracts = remaining
                        exit_reason = EXIT_STALLED
                        logger.warning(
                            f"STALL_EXIT_TRIGGER {pos.ticker} {pos.side.upper()} | "
                            f"stall_cycles={stall_cycles} >= eff_threshold={_eff_esc_threshold} "
                            f"(base={STALL_ESCALATION_CYCLES} quality_bonus=+{_quality_bonus} "
                            f"hq={_hq_for_stall} reason={_hq_stall_reason}) | "
                            f"mark={mark}¢ exit_ev={_stall_exit_ev:.1f}¢ — forcing exit"
                        )
                elif _quality_bonus > 0 and stall_cycles > 0:
                    logger.info(
                        f"STALL_QUALITY_HOLD_ACTIVE {pos.ticker} {pos.side.upper()} | "
                        f"stall_cycles={stall_cycles} < eff_threshold={_eff_esc_threshold} "
                        f"(quality_bonus=+{_quality_bonus}) — holding strong position | "
                        f"reason={_hq_stall_reason}"
                    )

            # ── Priority 7: Salvage stop (fail-safe) ─────────────────────────
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

    # ── Exitability scoring ───────────────────────────────────────────────────

    def _score_exitability(self, pos, liquidity: dict) -> dict:
        """
        Estimate how realistically a position can be exited at a reasonable fill.

        Score is 0–100 (higher = more exitable).

        Components (in order of penalty severity):
          book_depth:      can the book absorb our remaining size?
          spread_width:    is the spread narrow relative to price?
          slippage_ratio:  does slippage consume an unreasonable fraction of exit value?

        Slippage-ratio penalty replaces the old flat low-price penalty.
        A 15¢ price with 2¢ slippage = 13% exit cost — very costly.
        A 50¢ price with 2¢ slippage = 4% — manageable.
        This is more meaningful than penalising low prices in absolute terms,
        because high-price positions can also be illiquid.

        Returns a dict with score and component breakdown.
        """
        side = pos.side
        best_price = liquidity.get("best_yes_price" if side == "yes" else "best_no_price")
        spread = liquidity.get("spread")
        total_volume = liquidity.get("total_volume", 0.0)
        remaining = pos.contracts - pos.trimmed_contracts

        score = 100
        flags = []

        # No price at all — cannot exit regardless of other factors
        if not best_price or best_price <= 0:
            return {
                "score": 0,
                "best_price": None,
                "spread": spread,
                "total_volume": total_volume,
                "remaining_contracts": remaining,
                "flags": ["no_best_price"],
            }

        # Spread penalty: wide spread relative to best price is costly
        # Use spread/best_price ratio so the penalty scales with how much of the
        # exit value the spread consumes.
        if spread is not None and spread > 0 and best_price > 0:
            spread_ratio = spread / best_price
            if spread_ratio >= 0.30:        # spread ≥30% of price — very wide
                score -= 35
                flags.append(f"very_wide_spread({spread}¢/{spread_ratio:.0%}_of_price)")
            elif spread_ratio >= 0.15:      # spread 15–30% — wide
                score -= 20
                flags.append(f"wide_spread({spread}¢/{spread_ratio:.0%}_of_price)")
            elif spread >= STALL_SPREAD_WIDE_CENTS:  # absolute floor
                score -= 10
                flags.append(f"wide_spread_abs({spread}¢)")

        # Depth penalty: thin book relative to our position size
        # Scale penalty by how many contracts we need to unwind
        position_dollars = (best_price / 100.0) * remaining
        if total_volume <= 0:
            score -= 40
            flags.append("empty_book")
        elif total_volume < STALL_POOR_LIQUIDITY_DOLLARS:
            score -= 35
            flags.append(f"poor_depth(${total_volume:.2f})")
        elif total_volume < position_dollars * 0.5:
            # Book depth less than half our position size — can't exit cleanly
            score -= 20
            flags.append(f"thin_depth_vs_size(${total_volume:.2f}<50%_of_${position_dollars:.2f})")
        elif total_volume < STALL_POOR_LIQUIDITY_DOLLARS * 3:
            score -= 10
            flags.append(f"thin_depth(${total_volume:.2f})")

        # Slippage ratio penalty: how costly is the fixed slippage relative to the exit price?
        # SLIPPAGE_CENTS=2¢; at 10¢ that's 20% cost, at 50¢ that's 4%, at 80¢ that's 2.5%
        slippage_ratio = SLIPPAGE_CENTS / best_price if best_price > 0 else 1.0
        if slippage_ratio >= 0.20:      # slippage ≥20% of exit value — effectively zero value
            score -= 20
            flags.append(f"high_slippage_ratio({slippage_ratio:.0%})")
        elif slippage_ratio >= 0.10:    # slippage 10–20% — costly
            score -= 10
            flags.append(f"elevated_slippage_ratio({slippage_ratio:.0%})")

        score = max(0, min(100, score))
        return {
            "score": score,
            "best_price": best_price,
            "spread": spread,
            "total_volume": total_volume,
            "remaining_contracts": remaining,
            "slippage_ratio": round(slippage_ratio, 3),
            "flags": flags,
        }

    # ── High-quality hold assessment ──────────────────────────────────────────

    @staticmethod
    def _is_high_quality_hold(pos, state: str, hours_left: Optional[float]) -> tuple[bool, str]:
        """
        Returns (is_high_quality, reason_string).

        A position is considered "high quality" for holding purposes when it
        meets ALL of:
          1. Classified top_tier at entry
          2. Not in STATE_BROKEN or STATE_STALLED
          3. Strong initial edge at entry (entry_edge >= 0.12, i.e. ≥12¢)
          4. Low model uncertainty at entry (< 0.50)
          5. Decent liquidity at entry ($30+)
          6. No fragile flags (low-price, dangerous disagree, etc.)

        Additionally, near_locked state is always treated as high-quality to
        avoid unnecessary reduction of positions that are working well.

        Used to apply profile-driven extra stall cycles and fair-value grace
        time before forcing out positions that still have genuine path potential.

        Hard vetoes that immediately return False:
          - STATE_BROKEN (thesis invalidated — don't extend hold)
          - Any fragile flag present (fragile trades are not high-quality holds)
        """
        # Hard veto: broken positions must exit regardless of quality
        if state == STATE_BROKEN:
            return False, "state=broken"

        # near_locked is always considered high quality — it's working
        if state == STATE_NEAR_LOCKED:
            return True, "state=near_locked"

        # Must be a top_tier trade
        if getattr(pos, "trade_tier", "standard") != "top_tier":
            return False, f"trade_tier={getattr(pos, 'trade_tier', 'standard')}"

        # Fragile-flag veto: low-price or disagreement entries are not held longer
        fragile = getattr(pos, "fragile_flags", []) or []
        if fragile:
            return False, f"fragile_flags={fragile}"

        # Strong entry edge
        entry_edge = getattr(pos, "entry_edge", None)
        if entry_edge is None or entry_edge < 0.12:
            return False, f"entry_edge={entry_edge} < 0.12"

        # Low model uncertainty
        model_unc = getattr(pos, "model_uncertainty", 0.5)
        if model_unc >= 0.50:
            return False, f"model_uncertainty={model_unc:.2f} >= 0.50"

        # Decent entry liquidity
        entry_liq = getattr(pos, "entry_liquidity_dollars", None)
        if entry_liq is not None and entry_liq < 30.0:
            return False, f"entry_liq=${entry_liq:.0f} < $30"

        return True, (
            f"top_tier entry_edge={entry_edge:.3f} "
            f"unc={model_unc:.2f} liq=${entry_liq or 0:.0f}"
        )

    # ── Stalled / capital-trap classification ─────────────────────────────────

    def classify_stalled_positions(self) -> list[dict]:
        """
        Scans all open positions and flags any that appear to be capital traps.

        Design notes:
          - Only LIVE positions are evaluated; BROKEN/LOCKED/NEAR_LOCKED are skipped
            (BROKEN is handled by check_profit_takes; LOCKED/NEAR_LOCKED are working)
          - Hold EV flag is only raised when <STALL_HOLD_EV_HOURS_THRESHOLD hours
            remain, because with plenty of time the market is still live and a mid-range
            hold EV is not a stall signal
          - Consecutive stall counts are tracked in DailyState.stall_alert_counts to
            allow deduplication, escalation, and repeat-count logging
          - Positions cleared from stall (e.g. turned LOCKED) have their count reset

        Returns list of report dicts. Does NOT execute any exits.
        """
        if not self.kalshi:
            return []

        reports = []
        now_utc = datetime.now(timezone.utc)
        tickers_stalled_this_cycle: set = set()

        for pos in self.state.positions:
            if pos.status != "open":
                continue

            remaining = pos.contracts - pos.trimmed_contracts
            if remaining <= 0:
                continue

            # Age check — only consider positions old enough to have developed
            age_minutes = 0.0
            if pos.placed_at:
                try:
                    placed_dt = datetime.fromisoformat(pos.placed_at.replace("Z", "+00:00"))
                    age_minutes = (now_utc - placed_dt).total_seconds() / 60
                except Exception:
                    pass
            if age_minutes < STALL_MIN_AGE_MINUTES:
                continue

            # Get current market data — skip position if liquidity unavailable
            try:
                liquidity = self.kalshi.get_liquidity(pos.ticker)
            except Exception as e:
                logger.debug(f"STALL_CHECK: liquidity fetch failed for {pos.ticker}: {e}")
                continue

            mark = liquidity.get("best_yes_price" if pos.side == "yes" else "best_no_price")
            # If no mark at all, we cannot classify state or compute EV — skip
            if not mark or mark <= 0:
                logger.debug(f"STALL_CHECK: no mark for {pos.ticker} {pos.side} — skipping")
                continue

            # Use cached close_time to avoid an API call every 5-min cycle (P2 fix).
            if not pos.close_time:
                pos.close_time = _get_close_time(pos.ticker, self.kalshi)
            hours_left = _hours_to_settlement(pos.close_time)

            weather_report = None
            if self.weather:
                try:
                    city = self._infer_city(pos.ticker)
                    if city:
                        weather_report = self.weather.get_full_report(city)
                except Exception:
                    pass

            state = _classify_position(pos, mark, hours_left, weather_report)
            hold_ev = _model_hold_value(pos, mark, state, hours_left, weather_report)
            exit_ev = mark - SLIPPAGE_CENTS - FEE_CENTS
            exitability = self._score_exitability(pos, liquidity)

            # ── Skip non-live states ──────────────────────────────────────────
            if state == STATE_BROKEN:
                # check_profit_takes() handles broken positions
                continue
            if state in (STATE_LOCKED, STATE_NEAR_LOCKED):
                # Capital is working toward a profitable outcome — not a trap
                self.state.stall_alert_counts.pop(pos.ticker, None)
                continue

            # ── Quality-based stall assessment ────────────────────────────────
            # Pre-check quality before scoring stall flags: high-quality positions
            # get a slightly lower effective hold-EV ceiling (harder to trigger the
            # weak_hold_ev flag) in addition to the extra escalation cycle bonus.
            _stall_hq, _stall_hq_reason = self._is_high_quality_hold(pos, state, hours_left)
            _ev_bonus = op_profile.get_param("quality_stall_hold_ev_bonus_cents") if _stall_hq else 0.0
            _effective_hold_ev_ceiling = STALL_HOLD_EV_CEILING - _ev_bonus  # lower = harder to flag

            # ── Stall flag scoring ────────────────────────────────────────────
            stall_flags = []

            # Hold EV is only meaningful as a stall signal when time is short.
            # With >STALL_HOLD_EV_HOURS_THRESHOLD hours left, a mid-range hold EV
            # (e.g. 44¢) simply means the market hasn't moved yet — not a trap.
            if (hours_left is None or hours_left < STALL_HOLD_EV_HOURS_THRESHOLD) and \
                    hold_ev < _effective_hold_ev_ceiling:
                _hrs_str = f"{hours_left:.1f}" if hours_left is not None else "N/A"
                stall_flags.append(f"weak_hold_ev({hold_ev:.1f}¢,{_hrs_str}h_left)")

            if exit_ev < STALL_EXIT_EV_CEILING:
                stall_flags.append(f"weak_exit_ev({exit_ev:.1f}¢)")

            spread = liquidity.get("spread")
            if spread is not None and spread >= STALL_SPREAD_WIDE_CENTS:
                stall_flags.append(f"wide_spread({spread}¢)")

            if liquidity.get("total_volume", 0.0) < STALL_POOR_LIQUIDITY_DOLLARS:
                stall_flags.append(f"poor_liquidity(${liquidity.get('total_volume', 0):.2f})")

            # Only check MFE if high_water_mark has been set — it's populated by
            # check_profit_takes(), so a brand-new position that hasn't been through
            # that loop yet would always trigger this flag spuriously (P2 fix).
            if pos.high_water_mark is not None:
                if pos.high_water_mark <= pos.entry_price + STALL_MFE_REQUIRED_CENTS:
                    stall_flags.append(
                        f"no_favorable_excursion(hwm={pos.high_water_mark}¢ entry={pos.entry_price}¢)"
                    )

            if hours_left is not None and hours_left < 1.0 and mark <= 30:
                stall_flags.append(f"late_and_losing(hrs={hours_left:.1f} mark={mark}¢)")

            is_stalled = len(stall_flags) >= STALL_SCORE_THRESHOLD

            if not is_stalled:
                # Position cleared stall criteria — reset its counter
                self.state.stall_alert_counts.pop(pos.ticker, None)
                continue

            tickers_stalled_this_cycle.add(pos.ticker)

            # ── Consecutive stall cycle tracking ─────────────────────────────
            prev_count = self.state.stall_alert_counts.get(pos.ticker, 0)
            stall_cycle = prev_count + 1
            self.state.stall_alert_counts[pos.ticker] = stall_cycle

            # ── Quality-based holding tolerance ───────────────────────────────
            # High-quality positions get extra stall cycles before being treated
            # as urgent.  This prevents the bot from forcing out strong positions
            # that happen to be quiet but still have real path potential.
            # The bonus is profile-driven (0 for protection_first, 1+ for others).
            is_high_quality_hold, hq_reason = self._is_high_quality_hold(
                pos, state, hours_left
            )
            quality_cycle_bonus = 0
            if is_high_quality_hold:
                quality_cycle_bonus = op_profile.get_param("quality_stall_cycle_bonus")
                if quality_cycle_bonus > 0:
                    logger.info(
                        f"STALL_QUALITY_HOLD {pos.ticker} {pos.side.upper()} | "
                        f"high_quality_hold=True cycle_bonus=+{quality_cycle_bonus} | "
                        f"reason={hq_reason} profile={op_profile.ACTIVE_PROFILE}"
                    )

            effective_escalation_threshold = STALL_ESCALATION_CYCLES + quality_cycle_bonus

            # ── Determine action — escalate after repeated cycles ─────────────
            is_urgent = stall_cycle >= effective_escalation_threshold
            if exitability["score"] >= 50 and exit_ev > 5:
                action = "escalate_fair_value_exit"
            elif exitability["score"] >= 20 and exit_ev > 2:
                action = "attempt_partial_reduction"
            else:
                action = "monitor_and_log"
            # Upgrade action if this has been stalled repeatedly and is still exitable
            if is_urgent and action == "monitor_and_log" and exitability["score"] >= 15 and exit_ev > 1:
                action = "attempt_partial_reduction"

            # ── Decide whether to surface an alert this cycle ─────────────────
            # Alert on first detection, then only every STALL_ALERT_EVERY_N_CYCLES.
            # This prevents a 5-minute refresh loop from flooding Telegram.
            should_alert = (stall_cycle == 1) or (stall_cycle % STALL_ALERT_EVERY_N_CYCLES == 0)

            report = {
                "ticker": pos.ticker,
                "side": pos.side,
                "age_minutes": round(age_minutes, 1),
                "mark_cents": mark,
                "entry_price_cents": pos.entry_price,
                "hold_ev": round(hold_ev, 2),
                "exit_ev": round(exit_ev, 2),
                "exitability_score": exitability["score"],
                "exitability_flags": exitability["flags"],
                "spread_cents": spread,
                "total_volume_dollars": liquidity.get("total_volume", 0.0),
                "hours_left": hours_left,
                "state": state,
                "stall_flags": stall_flags,
                "stall_flag_count": len(stall_flags),
                "action": action,
                "remaining_contracts": remaining,
                "cost_dollars": pos.cost_dollars,
                "stall_cycle": stall_cycle,
                "is_urgent": is_urgent,
                "should_alert": should_alert,
                "is_high_quality_hold": is_high_quality_hold,
                "quality_cycle_bonus": quality_cycle_bonus,
                "effective_escalation_threshold": effective_escalation_threshold,
            }

            logger.warning(
                f"STALLED_POSITION {pos.ticker} {pos.side.upper()} "
                f"[cycle={stall_cycle}{'★URGENT' if is_urgent else ''}] "
                f"[hq={is_high_quality_hold} bonus=+{quality_cycle_bonus} esc_at={effective_escalation_threshold}] | "
                f"age={age_minutes:.0f}min mark={mark}¢ state={state} "
                f"hrs_left={f'{hours_left:.1f}' if hours_left is not None else 'N/A'} | "
                f"hold_ev={hold_ev:.1f}¢ exit_ev={exit_ev:.1f}¢ "
                f"exitability={exitability['score']}/100 | "
                f"flags={stall_flags} | action={action}"
            )

            reports.append(report)

        # Reset counters for any ticker that was previously stalled but not seen this cycle
        for ticker in list(self.state.stall_alert_counts.keys()):
            if ticker not in tickers_stalled_this_cycle:
                self.state.stall_alert_counts.pop(ticker, None)

        # NOTE: _save() is intentionally NOT called here.
        # Stall counts are persisted by the orchestrator after check_profit_takes()
        # completes, so a crash in check_profit_takes can't overcount cycles.

        return reports

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
                     entry_context: Optional[dict] = None,
                     settlement_date: str = "",
                     threshold: Optional[float] = None,
                     is_bucket: bool = False):
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
            trade_tier=ctx.get("trade_tier", "standard"),
            settlement_date=settlement_date,
            threshold=threshold,
            is_bucket=is_bucket,
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
