"""
Validation framework for proposed strategy changes.

Purpose
-------
Before any parameter change is applied to the live bot, it must be evaluated
against historical data. This module provides three complementary tools:

  1. historical_replay  — score a trade_history slice under a candidate config
                          and compare its metrics to the baseline config that
                          was actually live at the time.

  2. shadow_mode_eval   — evaluate a candidate config against live signals
                          without placing orders, logging the results for
                          later comparison.

  3. ab_compare         — compare two sets of ValidationMetrics (baseline vs
                          candidate) and return a structured verdict dict that
                          the advisor uses to gate recommendations.

Design principles
-----------------
  - No live orders are placed by any function here.
  - All functions are synchronous and safe to call from the scheduler.
  - Evaluation is multi-metric: win rate, calibration quality, realized edge,
    drawdown, false positive rate, slippage sensitivity, and PnL stability.
  - The framework outputs a ValidationResult with a `pass_validation` bool.
    Only recommendations with pass_validation=True may advance from
    "test_in_shadow_mode" or "validate_in_backtest" to "safe_config_update".
  - Minimum sample gates prevent spurious conclusions on thin data.

IMPORTANT: historical replay cannot perfectly reproduce the live bot's behavior
(market prices, fills, METAR data at the time are not fully stored). The replay
is best treated as a *direction check*, not a precise P&L simulation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

DATA_DIR           = os.path.join(os.path.dirname(__file__), "../../data")
HISTORY_FILE       = os.path.join(DATA_DIR, "trade_history.json")
SHADOW_LOG_FILE    = os.path.join(DATA_DIR, "shadow_mode_log.json")
VALIDATION_LOG_FILE = os.path.join(DATA_DIR, "validation_log.json")

# ── Minimum sample requirements before a ValidationResult can be trusted ──────
MIN_TRADES_VALIDATE  = 15   # need at least this many comparable trades
MIN_TRADES_CONCLUDE  = 30   # need this many for a "pass" verdict to be binding

# ── Metric thresholds for automated pass/fail ─────────────────────────────────
# A candidate must beat (or not significantly underperform) the baseline on a
# *majority* of these checks to pass. The bar is intentionally conservative.
WIN_RATE_REGRESSION_LIMIT   = -0.05   # candidate win_rate must not be >5pp below baseline
PNL_REGRESSION_LIMIT_PCT    = -0.10   # candidate avg_pnl must not be >10% below baseline
DRAWDOWN_INCREASE_LIMIT_PCT =  0.15   # max acceptable increase in drawdown vs baseline
CALIBRATION_TOLERANCE       =  0.08   # max acceptable calibration error (fraction)


@dataclass
class ValidationMetrics:
    """
    Computed metrics for a single config evaluated on a set of trades.
    Both baseline and candidate produce one of these; ab_compare diffs them.
    """
    label: str                          # "baseline" or "candidate"
    trade_count: int = 0
    win_count: int = 0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0           # peak-to-trough in dollars (positive = loss)
    avg_edge_cents: float = 0.0         # average edge (our_prob - market_price) in cents
    realized_edge_capture: float = 0.0  # fraction of theoretical edge actually captured
    false_positive_rate: float = 0.0    # fraction of trades that were predicted wins but lost
    calibration_error: float = 0.0      # mean |predicted_prob - binary_outcome|
    pnl_std: float = 0.0                # PnL standard deviation (consistency proxy)
    by_city: dict = field(default_factory=dict)
    by_market_type: dict = field(default_factory=dict)
    computed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "label":                  self.label,
            "trade_count":            self.trade_count,
            "win_count":              self.win_count,
            "win_rate":               round(self.win_rate, 4),
            "avg_pnl_per_trade":      round(self.avg_pnl_per_trade, 4),
            "total_pnl":              round(self.total_pnl, 4),
            "max_drawdown":           round(self.max_drawdown, 4),
            "avg_edge_cents":         round(self.avg_edge_cents, 4),
            "realized_edge_capture":  round(self.realized_edge_capture, 4),
            "false_positive_rate":    round(self.false_positive_rate, 4),
            "calibration_error":      round(self.calibration_error, 4),
            "pnl_std":                round(self.pnl_std, 4),
            "by_city":                self.by_city,
            "by_market_type":         self.by_market_type,
            "computed_at":            self.computed_at,
        }


@dataclass
class ValidationResult:
    """
    Output of ab_compare — the structured verdict used by the advisor to gate
    whether a recommendation can advance to safe_config_update.
    """
    recommendation_id: str
    baseline: ValidationMetrics
    candidate: ValidationMetrics
    pass_validation: bool
    verdict_reason: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    insufficient_data: bool = False
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "recommendation_id":  self.recommendation_id,
            "baseline":           self.baseline.to_dict(),
            "candidate":          self.candidate.to_dict(),
            "pass_validation":    self.pass_validation,
            "verdict_reason":     self.verdict_reason,
            "checks_passed":      self.checks_passed,
            "checks_failed":      self.checks_failed,
            "insufficient_data":  self.insufficient_data,
            "generated_at":       self.generated_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    trades: list[dict],
    label: str,
    edge_cents_field: str = "edge_cents",
) -> ValidationMetrics:
    """
    Computes ValidationMetrics from a list of closed position dicts.

    Expected fields per trade dict:
        pnl_dollars: float
        our_prob: float          (predicted probability, 0–1)
        outcome: "win"|"loss"|"scratch"   OR  pnl_dollars > 0
        edge_cents: float        (our_prob*100 - market_price_cents)  [optional]
        city: str                [optional]
        market_type: str         [optional]
    """
    if not trades:
        return ValidationMetrics(label=label)

    pnls   = [t.get("pnl_dollars", 0.0) for t in trades]
    wins   = sum(1 for p in pnls if p > 0)
    losses = len(pnls) - wins

    win_rate = wins / len(pnls)
    total_pnl = sum(pnls)
    avg_pnl   = total_pnl / len(pnls)

    # Max drawdown (sequential peak-to-trough on cumulative P&L curve)
    cum_pnl   = 0.0
    peak      = 0.0
    max_dd    = 0.0
    for p in pnls:
        cum_pnl += p
        peak     = max(peak, cum_pnl)
        dd       = peak - cum_pnl
        max_dd   = max(max_dd, dd)

    # Edge metrics
    edges = [t.get(edge_cents_field, 0.0) for t in trades
             if t.get(edge_cents_field) is not None]
    avg_edge = statistics.mean(edges) if edges else 0.0

    # Realized edge capture: how much of the theoretical edge did we actually capture?
    # Proxy: avg_pnl_per_trade / (avg_edge_cents / 100).
    # Meaningful only when average edge is positive; clamp outliers to [-2, 2].
    if avg_edge > 0:
        realized_capture = avg_pnl / (avg_edge / 100.0)
    else:
        realized_capture = 0.0
    realized_capture = max(-2.0, min(2.0, realized_capture))

    # Calibration error: mean |predicted_prob - binary_outcome|
    # binary_outcome = 1.0 for win, 0.0 for loss/scratch
    calibration_errors = []
    for t in trades:
        prob = t.get("our_prob")
        if prob is None:
            continue
        outcome_val = 1.0 if t.get("pnl_dollars", 0.0) > 0 else 0.0
        calibration_errors.append(abs(prob - outcome_val))
    cal_err = statistics.mean(calibration_errors) if calibration_errors else 0.0

    # False positive rate: predicted win (our_prob > 0.5) but lost
    predicted_wins = [t for t in trades if t.get("our_prob", 0.5) > 0.5]
    fp_rate = (
        sum(1 for t in predicted_wins if t.get("pnl_dollars", 0.0) <= 0) / len(predicted_wins)
        if predicted_wins else 0.0
    )

    # PnL std
    pnl_std = statistics.stdev(pnls) if len(pnls) >= 2 else 0.0

    # Per-city breakdown
    city_pnls: dict[str, list[float]] = {}
    for t in trades:
        city = t.get("city", "UNKNOWN")
        city_pnls.setdefault(city, []).append(t.get("pnl_dollars", 0.0))
    by_city = {
        c: {
            "total_pnl": round(sum(ps), 2),
            "win_rate":  round(sum(1 for p in ps if p > 0) / len(ps), 3),
            "trades":    len(ps),
        }
        for c, ps in city_pnls.items()
    }

    # Per-market_type breakdown
    type_pnls: dict[str, list[float]] = {}
    for t in trades:
        mt = t.get("market_type", "UNKNOWN")
        type_pnls.setdefault(mt, []).append(t.get("pnl_dollars", 0.0))
    by_type = {
        mt: {
            "total_pnl": round(sum(ps), 2),
            "win_rate":  round(sum(1 for p in ps if p > 0) / len(ps), 3),
            "trades":    len(ps),
        }
        for mt, ps in type_pnls.items()
    }

    return ValidationMetrics(
        label=label,
        trade_count=len(pnls),
        win_count=wins,
        win_rate=round(win_rate, 4),
        avg_pnl_per_trade=round(avg_pnl, 4),
        total_pnl=round(total_pnl, 4),
        max_drawdown=round(max_dd, 4),
        avg_edge_cents=round(avg_edge, 4),
        realized_edge_capture=round(realized_capture, 4),
        false_positive_rate=round(fp_rate, 4),
        calibration_error=round(cal_err, 4),
        pnl_std=round(pnl_std, 4),
        by_city=by_city,
        by_market_type=by_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Historical replay
# ─────────────────────────────────────────────────────────────────────────────

def historical_replay(
    recommendation_id: str,
    lookback_days: int = 30,
    affected_cities: Optional[list[str]] = None,
    affected_market_types: Optional[list[str]] = None,
) -> ValidationResult:
    """
    Evaluates the historical trade record as the "baseline".
    The "candidate" cannot be replayed without re-running the full engine, so
    we compare the historical slice against the full history as a sanity check.

    More precisely, this function:
      - Loads trade_history.json
      - Computes ValidationMetrics for:
          a) All history (full baseline)
          b) The recent slice (lookback_days) filtered to affected_cities / market_types
      - Returns a ValidationResult comparing the two
      - If recent slice has fewer trades than MIN_TRADES_VALIDATE, marks as
        insufficient_data

    This is useful for answering: "Is the recent period representative of the
    overall baseline, or is something anomalous?"  If recent metrics are
    significantly worse, a candidate change targeting that period has a lower
    evidence bar to clear.
    """
    history = _load_json_safe(HISTORY_FILE, [])

    all_positions: list[dict] = []
    for record in history:
        for pos in record.get("positions", []):
            if pos.get("status") in ("closed", "expired"):
                pos_copy = dict(pos)
                pos_copy.setdefault("date", record.get("date", ""))
                all_positions.append(pos_copy)

    # Full history baseline
    baseline_metrics = _compute_metrics(all_positions, "baseline_full")

    # Recent + filtered slice
    cutoff_ts = datetime.now(timezone.utc).timestamp() - lookback_days * 86400
    recent: list[dict] = []
    for pos in all_positions:
        date_str = pos.get("date", "")
        try:
            ts = datetime.fromisoformat(date_str + "T12:00:00+00:00").timestamp()
            if ts < cutoff_ts:
                continue
        except Exception:
            pass  # include if date parse fails (conservative)

        if affected_cities and pos.get("city") not in affected_cities:
            continue
        if affected_market_types and pos.get("market_type") not in affected_market_types:
            continue
        recent.append(pos)

    candidate_metrics = _compute_metrics(recent, "recent_slice")

    result = ab_compare(recommendation_id, baseline_metrics, candidate_metrics)
    _log_validation_result(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shadow mode logging
# ─────────────────────────────────────────────────────────────────────────────

def log_shadow_evaluation(
    ticker: str,
    city: str,
    market_type: str,
    our_prob: float,
    market_price_cents: int,
    edge_cents: float,
    signal_breakdown: list[dict],
    weights_version: str,
    context: Optional[dict] = None,
) -> None:
    """
    Records a shadow-mode evaluation for a trade candidate that was NOT executed
    (or was executed, but we also want the candidate-config score for comparison).

    Call this from the engine whenever a recommendation is scored — regardless of
    whether it is ultimately executed. The actual_outcome field is filled in by
    the settlement backfill job.

    This data is the foundation for future offline weight optimization.
    """
    record = {
        "ticker":            ticker,
        "city":              city,
        "market_type":       market_type,
        "our_prob":          round(our_prob, 4),
        "market_price_cents": market_price_cents,
        "edge_cents":        round(edge_cents, 2),
        "signal_breakdown":  signal_breakdown,   # list of {name, prob_adjustment, confidence, note}
        "weights_version":   weights_version,
        "context":           context or {},
        "actual_outcome":    None,   # filled in by backfill
        "would_have_won":    None,   # filled in by backfill
        "logged_at":         datetime.now(timezone.utc).isoformat(),
    }

    log = _load_json_safe(SHADOW_LOG_FILE, [])
    log.append(record)
    # Keep last 2000 shadow evaluations to avoid unbounded growth
    log = log[-2000:]
    try:
        with open(SHADOW_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        logger.warning(f"shadow_log write failed: {e}")


def backfill_shadow_outcomes(settled_markets: dict) -> int:
    """
    After settlement, fill in actual_outcome and would_have_won for any pending
    shadow log entries matching the settled tickers.

    settled_markets: dict of ticker -> "yes" | "no" (winning side)
    Returns count of records updated.
    """
    log = _load_json_safe(SHADOW_LOG_FILE, [])
    updated = 0
    for rec in log:
        if rec.get("would_have_won") is not None:
            continue
        ticker = rec.get("ticker", "")
        if ticker not in settled_markets:
            continue
        winning_side = settled_markets[ticker]
        our_side = "yes" if rec.get("our_prob", 0.5) > 0.5 else "no"
        rec["actual_outcome"] = winning_side
        rec["would_have_won"] = (our_side == winning_side)
        updated += 1

    if updated:
        try:
            with open(SHADOW_LOG_FILE, "w") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            logger.warning(f"shadow_log backfill write failed: {e}")

    return updated


def get_shadow_metrics(
    weights_version: Optional[str] = None,
    lookback_days: int = 14,
    min_trades: int = MIN_TRADES_VALIDATE,
) -> Optional[ValidationMetrics]:
    """
    Computes ValidationMetrics from the shadow log, filtered to a specific
    weights_version (or all versions if None) and to resolved records only.

    Returns None if fewer than min_trades resolved records exist.
    """
    log = _load_json_safe(SHADOW_LOG_FILE, [])
    cutoff_ts = datetime.now(timezone.utc).timestamp() - lookback_days * 86400

    resolved = []
    for rec in log:
        if rec.get("would_have_won") is None:
            continue
        try:
            ts = datetime.fromisoformat(rec["logged_at"].replace("Z", "+00:00")).timestamp()
            if ts < cutoff_ts:
                continue
        except Exception:
            pass
        if weights_version and rec.get("weights_version") != weights_version:
            continue
        resolved.append({
            "pnl_dollars":  1.0 if rec.get("would_have_won") else -1.0,   # normalised
            "our_prob":     rec.get("our_prob", 0.5),
            "edge_cents":   rec.get("edge_cents", 0.0),
            "city":         rec.get("city", "UNKNOWN"),
            "market_type":  rec.get("market_type", "UNKNOWN"),
        })

    if len(resolved) < min_trades:
        logger.info(
            f"shadow_metrics: only {len(resolved)} resolved records "
            f"(need {min_trades}) — returning None"
        )
        return None

    label = f"shadow_{weights_version or 'all'}_{lookback_days}d"
    return _compute_metrics(resolved, label)


# ─────────────────────────────────────────────────────────────────────────────
# 3. A/B comparison
# ─────────────────────────────────────────────────────────────────────────────

def ab_compare(
    recommendation_id: str,
    baseline: ValidationMetrics,
    candidate: ValidationMetrics,
) -> ValidationResult:
    """
    Compares baseline vs candidate ValidationMetrics and produces a structured
    ValidationResult with explicit pass/fail for each check.

    Checks:
      1. Sufficient data gate
      2. Win rate regression
      3. Average PnL regression
      4. Max drawdown increase
      5. Calibration error
      6. Realized edge capture (candidate should not be significantly worse)

    A recommendation passes if no check is failed AND data is sufficient.
    If data is insufficient, pass_validation=False and insufficient_data=True.
    """
    checks_passed: list[str] = []
    checks_failed: list[str] = []
    insufficient_data = False

    # 1. Sufficient data
    n_base = baseline.trade_count
    n_cand = candidate.trade_count
    if n_base < MIN_TRADES_VALIDATE or n_cand < MIN_TRADES_VALIDATE:
        insufficient_data = True
        reason = (
            f"Insufficient data: baseline={n_base} trades, "
            f"candidate={n_cand} trades (need {MIN_TRADES_VALIDATE})"
        )
        return ValidationResult(
            recommendation_id=recommendation_id,
            baseline=baseline,
            candidate=candidate,
            pass_validation=False,
            verdict_reason=reason,
            insufficient_data=True,
        )

    binding = (n_base >= MIN_TRADES_CONCLUDE and n_cand >= MIN_TRADES_CONCLUDE)

    # 2. Win rate regression
    win_delta = candidate.win_rate - baseline.win_rate
    if win_delta < WIN_RATE_REGRESSION_LIMIT:
        checks_failed.append(
            f"win_rate_regression: {baseline.win_rate:.1%} → {candidate.win_rate:.1%} "
            f"(delta={win_delta:+.1%}, limit={WIN_RATE_REGRESSION_LIMIT:+.1%})"
        )
    else:
        checks_passed.append(
            f"win_rate_ok: {baseline.win_rate:.1%} → {candidate.win_rate:.1%}"
        )

    # 3. Average PnL regression
    if baseline.avg_pnl_per_trade != 0:
        pnl_delta_pct = (candidate.avg_pnl_per_trade - baseline.avg_pnl_per_trade) / abs(
            baseline.avg_pnl_per_trade
        )
    else:
        pnl_delta_pct = 0.0

    if pnl_delta_pct < PNL_REGRESSION_LIMIT_PCT:
        checks_failed.append(
            f"avg_pnl_regression: {baseline.avg_pnl_per_trade:+.4f} → "
            f"{candidate.avg_pnl_per_trade:+.4f} ({pnl_delta_pct:+.1%})"
        )
    else:
        checks_passed.append(
            f"avg_pnl_ok: delta={pnl_delta_pct:+.1%}"
        )

    # 4. Max drawdown increase
    if baseline.max_drawdown > 0:
        dd_increase_pct = (candidate.max_drawdown - baseline.max_drawdown) / baseline.max_drawdown
    else:
        dd_increase_pct = 0.0

    if dd_increase_pct > DRAWDOWN_INCREASE_LIMIT_PCT:
        checks_failed.append(
            f"drawdown_increase: {baseline.max_drawdown:.2f} → {candidate.max_drawdown:.2f} "
            f"({dd_increase_pct:+.1%}, limit={DRAWDOWN_INCREASE_LIMIT_PCT:.0%})"
        )
    else:
        checks_passed.append(
            f"drawdown_ok: delta={dd_increase_pct:+.1%}"
        )

    # 5. Calibration error
    if candidate.calibration_error > CALIBRATION_TOLERANCE:
        checks_failed.append(
            f"calibration_error: {candidate.calibration_error:.3f} > {CALIBRATION_TOLERANCE:.3f}"
        )
    else:
        checks_passed.append(
            f"calibration_ok: {candidate.calibration_error:.3f}"
        )

    # 6. Realized edge capture (warn if candidate is meaningfully worse)
    if (baseline.realized_edge_capture > 0 and
            candidate.realized_edge_capture < baseline.realized_edge_capture * 0.70):
        checks_failed.append(
            f"edge_capture_degraded: {baseline.realized_edge_capture:.2f} → "
            f"{candidate.realized_edge_capture:.2f}"
        )
    else:
        checks_passed.append(
            f"edge_capture_ok: {candidate.realized_edge_capture:.2f}"
        )

    pass_validation = len(checks_failed) == 0 and binding

    if not binding and len(checks_failed) == 0:
        verdict = (
            f"All {len(checks_passed)} checks passed but data is marginal "
            f"(need {MIN_TRADES_CONCLUDE} trades for binding verdict). "
            "Treat as directional signal only."
        )
        pass_validation = False  # be conservative
    elif len(checks_failed) == 0:
        verdict = (
            f"All {len(checks_passed)} checks passed on {n_cand} candidate trades. "
            "Candidate config appears safe."
        )
    else:
        verdict = (
            f"{len(checks_failed)} of {len(checks_passed)+len(checks_failed)} checks failed. "
            f"Failed: {'; '.join(checks_failed[:2])}."
        )

    result = ValidationResult(
        recommendation_id=recommendation_id,
        baseline=baseline,
        candidate=candidate,
        pass_validation=pass_validation,
        verdict_reason=verdict,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        insufficient_data=insufficient_data,
    )
    _log_validation_result(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Validation log
# ─────────────────────────────────────────────────────────────────────────────

def _log_validation_result(result: ValidationResult) -> None:
    """Append a ValidationResult to validation_log.json (rolling last 200)."""
    log = _load_json_safe(VALIDATION_LOG_FILE, [])
    log.append(result.to_dict())
    log = log[-200:]
    try:
        with open(VALIDATION_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        logger.warning(f"validation_log write failed: {e}")


def get_validation_result_for(recommendation_id: str) -> Optional[dict]:
    """
    Retrieves the most recent ValidationResult for a given recommendation_id.
    Returns None if no result exists.
    """
    log = _load_json_safe(VALIDATION_LOG_FILE, [])
    matches = [r for r in log if r.get("recommendation_id") == recommendation_id]
    return matches[-1] if matches else None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json_safe(path: str, default: Any) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        logger.warning(f"validation._load_json_safe({path}): {e}")
        return default
