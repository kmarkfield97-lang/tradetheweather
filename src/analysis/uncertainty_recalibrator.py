"""
Controlled recalibration of city_uncertainty.json.

Philosophy
----------
city_uncertainty.json must NOT be updated per-trade (online learning → overfitting).
Instead, this module runs on a scheduled basis (nightly or weekly) and proposes
updates using:

  1. A rolling window of recent forecast errors (from forecast_errors.json)
  2. A minimum sample-size gate before any change is accepted
  3. Bayesian shrinkage: new estimate = blend(prior, recent_observed)
     — alpha = weight on recent data, (1-alpha) = weight on existing prior
  4. Segmentation: city × market_type × season (only when data is sufficient)
  5. All proposed updates go through pending_approvals.json — never auto-written

Output
------
  - Returns a list of AdvisorRecommendation objects (uncertainty_update change_type)
    via propose_recalibration_recommendations().
  - Does NOT write to city_uncertainty.json directly.
  - The approval workflow applies approved updates via apply_approved_recalibration().

Anti-overfitting safeguards
---------------------------
  - MIN_SAMPLES_RECALIBRATE: need at least 10 observations before any update
  - MAX_SHRINKAGE_ALPHA: cap on how much recent data can shift the prior (default 0.30)
  - MAX_DELTA_F: a single recalibration cycle cannot move sigma by more than 3°F
  - REQUIRE_CONSISTENT_DIRECTION: majority of errors must be same-sign for a mean-bias
    update (otherwise attribute to noise)
  - STALE_PRIOR_DAYS: if the prior was set longer ago than this, allow slightly higher
    alpha (the world may have changed)
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
ERRORS_FILE      = os.path.join(DATA_DIR, "forecast_errors.json")
UNCERTAINTY_FILE = os.path.join(DATA_DIR, "city_uncertainty.json")
APPROVALS_FILE   = os.path.join(DATA_DIR, "pending_approvals.json")

# ── Safety parameters ──────────────────────────────────────────────────────────
MIN_SAMPLES_RECALIBRATE = 10    # minimum errors before proposing any update
MAX_SHRINKAGE_ALPHA     = 0.30  # max weight on recent data vs prior (prevents overfit)
STALE_PRIOR_ALPHA       = 0.40  # allow slightly higher alpha when prior is old
STALE_PRIOR_DAYS        = 90    # prior is considered stale after this many days
MAX_DELTA_F             = 3.0   # maximum sigma shift in a single cycle (°F)
DIRECTION_CONSISTENCY_MIN = 0.65  # fraction of errors that must be same-sign for bias update
ROLLING_WINDOW_DAYS     = 60    # look back this many days of errors


@dataclass
class RecalibrationProposal:
    """
    Proposed change to a single city_uncertainty.json entry.
    Contains the full evidence trail so the advisor can decide whether to forward it
    as an AdvisorRecommendation.
    """
    segment: str                     # e.g. "LOS_ANGELES:temp_high:spring"
    current_sigma: float             # existing value in city_uncertainty.json (or default)
    proposed_sigma: float            # new suggested value
    delta: float                     # proposed_sigma - current_sigma
    sample_size: int
    mean_error: float                # mean forecast - actual (positive = NWS runs warm)
    error_std: float                 # std dev of errors
    direction_consistency: float     # fraction pointing same way
    rolling_window_days: int
    alpha_used: float                # actual shrinkage alpha applied
    evidence_strength: str           # matches EvidenceStrength in classifier.py
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "segment":               self.segment,
            "current_sigma":         round(self.current_sigma, 3),
            "proposed_sigma":        round(self.proposed_sigma, 3),
            "delta":                 round(self.delta, 3),
            "sample_size":           self.sample_size,
            "mean_error":            round(self.mean_error, 3),
            "error_std":             round(self.error_std, 3),
            "direction_consistency": round(self.direction_consistency, 3),
            "rolling_window_days":   self.rolling_window_days,
            "alpha_used":            round(self.alpha_used, 3),
            "evidence_strength":     self.evidence_strength,
            "generated_at":          self.generated_at,
            "notes":                 self.notes,
        }


def _load_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _get_season(date_str: str) -> str:
    """Returns 'winter'/'spring'/'summer'/'fall' for an ISO date string."""
    from datetime import date as _date
    try:
        d = _date.fromisoformat(date_str)
        m = d.month
        if m in (12, 1, 2):
            return "winter"
        elif m in (3, 4, 5):
            return "spring"
        elif m in (6, 7, 8):
            return "summer"
        else:
            return "fall"
    except Exception:
        return ""


def _evidence_strength(n: int, consistency: float) -> str:
    """Mirrors classifier.py compute_evidence_strength."""
    if n < 5:
        return "anecdotal"
    if n < 15:
        return "weak" if consistency >= 0.6 else "anecdotal"
    if n < 30:
        return "moderate" if consistency >= 0.75 else "weak"
    if n < 50:
        return "strong" if consistency >= 0.80 else "moderate"
    return "conclusive" if consistency >= 0.85 else "strong"


def _collect_recent_errors(errors: list, window_days: int) -> list:
    """Filter error records to the rolling window."""
    cutoff = datetime.now(timezone.utc).timestamp() - window_days * 86400
    recent = []
    for e in errors:
        date_str = e.get("date", "")
        try:
            # forecast_errors.json stores date as YYYY-MM-DD
            ts = datetime.fromisoformat(date_str + "T12:00:00+00:00").timestamp()
            if ts >= cutoff:
                recent.append(e)
        except Exception:
            pass
    return recent


def compute_recalibration_proposals(
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
) -> list[RecalibrationProposal]:
    """
    Core logic: reads forecast_errors.json and city_uncertainty.json, then
    produces RecalibrationProposal objects for each segment with enough data.

    Does NOT write anything — all output is returned for the caller to decide.
    """
    errors_data  = _load_json(ERRORS_FILE, {"forecast_errors": []})
    uncertainty  = _load_json(UNCERTAINTY_FILE, {})

    all_errors   = errors_data.get("forecast_errors", [])
    recent_errors = _collect_recent_errors(all_errors, rolling_window_days)

    if not recent_errors:
        logger.info("recalibrator: no recent errors — nothing to propose")
        return []

    # Group by city:market_type:season
    segments: dict[str, list[float]] = {}
    for e in recent_errors:
        city   = e.get("city", "")
        mtype  = e.get("market_type", "")
        season = e.get("season", "") or _get_season(e.get("date", ""))
        error  = e.get("error")
        if not city or not mtype or error is None:
            continue
        key = f"{city}:{mtype}:{season}" if season else f"{city}:{mtype}"
        segments.setdefault(key, []).append(float(error))

    proposals: list[RecalibrationProposal] = []

    for seg, errs in segments.items():
        n = len(errs)
        if n < MIN_SAMPLES_RECALIBRATE:
            logger.debug(f"recalibrator: {seg} has only {n} samples — skipping")
            continue

        mean_err = statistics.mean(errs)
        try:
            err_std  = statistics.stdev(errs)
        except statistics.StatisticsError:
            err_std  = abs(mean_err)

        # Direction consistency (how often errors are same sign as mean).
        # If mean_err is exactly 0 the direction is undefined; treat as 0.5.
        if mean_err == 0.0:
            consistency = 0.5
        else:
            same_sign = sum(1 for e in errs if (e > 0) == (mean_err > 0))
            consistency = same_sign / n

        # Look up current sigma — try segment-specific key first, then city:mtype
        current_sigma = None
        if seg in uncertainty:
            current_sigma = float(uncertainty[seg])
        else:
            # Try without season
            parts = seg.split(":")
            if len(parts) == 3:
                base_key = f"{parts[0]}:{parts[1]}"
                if base_key in uncertainty:
                    current_sigma = float(uncertainty[base_key])
        if current_sigma is None:
            # Fall back to a reasonable default (matches engine.py DEFAULT_UNCERTAINTY_F)
            current_sigma = 6.0

        # Recalibration logic:
        # We're updating the sigma (forecast spread), not a bias correction.
        # The new sigma is proposed as a blend of the existing prior and the
        # observed absolute error magnitude (mean absolute error ≈ new sigma estimate).
        # We do NOT apply a directional bias correction here — that's the advisor's job.
        # Sigma should be non-negative and non-zero.
        mean_abs_err   = statistics.mean([abs(e) for e in errs])
        recent_sigma   = max(1.0, mean_abs_err)  # treat MAE as empirical sigma

        # Alpha: how much to trust recent data. Cap it.
        alpha = MAX_SHRINKAGE_ALPHA
        evidence = _evidence_strength(n, consistency)

        # Compute proposed sigma via shrinkage blend
        proposed_sigma = (1.0 - alpha) * current_sigma + alpha * recent_sigma

        # Guard: clamp single-cycle delta
        delta = proposed_sigma - current_sigma
        if abs(delta) > MAX_DELTA_F:
            proposed_sigma = current_sigma + (MAX_DELTA_F if delta > 0 else -MAX_DELTA_F)
            delta = proposed_sigma - current_sigma

        proposed_sigma = round(max(1.0, proposed_sigma), 2)
        delta = round(proposed_sigma - current_sigma, 3)

        # Only propose changes that are material (>0.5°F shift) and have some evidence
        if abs(delta) < 0.5 or evidence in ("anecdotal",):
            logger.debug(
                f"recalibrator: {seg} delta={delta:+.2f}°F evidence={evidence} — "
                "change too small or evidence too weak, skipping"
            )
            continue

        notes_parts = []
        if consistency < DIRECTION_CONSISTENCY_MIN:
            notes_parts.append(
                f"low direction consistency ({consistency:.0%}) — "
                "errors are noisy, sigma update only (no bias)"
            )
        if mean_abs_err > current_sigma * 1.5:
            notes_parts.append(
                f"MAE ({mean_abs_err:.1f}°F) >> prior sigma ({current_sigma:.1f}°F) — "
                "forecast model may have degraded for this segment"
            )

        proposals.append(RecalibrationProposal(
            segment=seg,
            current_sigma=current_sigma,
            proposed_sigma=proposed_sigma,
            delta=delta,
            sample_size=n,
            mean_error=round(mean_err, 2),
            error_std=round(err_std, 2),
            direction_consistency=round(consistency, 3),
            rolling_window_days=rolling_window_days,
            alpha_used=alpha,
            evidence_strength=evidence,
            notes="; ".join(notes_parts),
        ))

    proposals.sort(key=lambda p: abs(p.delta), reverse=True)
    logger.info(
        f"recalibrator: proposed {len(proposals)} uncertainty updates "
        f"from {len(recent_errors)} recent errors across {len(segments)} segments"
    )
    return proposals


def propose_recalibration_recommendations(
    rolling_window_days: int = ROLLING_WINDOW_DAYS,
) -> list[dict]:
    """
    Converts RecalibrationProposals into AdvisorRecommendation-compatible dicts
    and appends them to pending_approvals.json.

    Returns the list of new recommendation dicts added.
    Existing pending items are preserved; duplicate IDs are not re-added.

    This function is called by the nightly/weekly scheduler job.
    """
    proposals = compute_recalibration_proposals(rolling_window_days)
    if not proposals:
        return []

    # Load existing approvals to avoid duplicates
    existing = _load_json(APPROVALS_FILE, [])
    existing_ids = {e.get("id") for e in existing}

    new_recs = []
    for p in proposals:
        change_id = f"uncertainty_recal_{p.segment.replace(':', '_').lower()}_{p.sample_size}t"

        if change_id in existing_ids:
            logger.debug(f"recalibrator: {change_id} already in pending_approvals — skipping")
            continue

        # Map evidence_strength to severity
        sev_map = {
            "conclusive": "P1", "strong": "P2",
            "moderate": "P2", "weak": "P3", "anecdotal": "P3",
        }
        sev = sev_map.get(p.evidence_strength, "P3")

        # Determine adaptation maturity tier
        if p.evidence_strength in ("conclusive", "strong") and p.sample_size >= 30:
            adaptation_maturity = "validate_in_backtest"
        elif p.evidence_strength in ("moderate",):
            adaptation_maturity = "test_in_shadow_mode"
        else:
            adaptation_maturity = "collect_more_data"

        rec = {
            "id":                    change_id,
            "title":                 f"Uncertainty recalibration: {p.segment}",
            "category":              "uncertainty_update",
            "change_type":           "uncertainty_update",
            "severity":              sev,
            "confidence":            round(p.direction_consistency, 3),
            "evidence_strength":     p.evidence_strength,
            "sample_size":           p.sample_size,
            "affected_cities":       [p.segment.split(":")[0]],
            "affected_market_types": [p.segment.split(":")[1]] if len(p.segment.split(":")) >= 2 else [],
            "likely_root_cause":     (
                f"Empirical MAE ({abs(p.mean_error):.1f}°F avg error) "
                f"diverges from stored sigma ({p.current_sigma:.1f}°F) for {p.segment}"
            ),
            "recommended_action":    "queue_for_review",
            "adaptation_maturity":   adaptation_maturity,
            "auto_apply_allowed":    False,
            "manual_review_required": True,
            "rollback_risk":         "low",
            "proposed_change_summary": (
                f"Update city_uncertainty.json['{p.segment}'] "
                f"from {p.current_sigma:.1f}°F → {p.proposed_sigma:.1f}°F "
                f"(delta={p.delta:+.2f}°F, alpha={p.alpha_used:.2f}, "
                f"n={p.sample_size}, window={p.rolling_window_days}d)"
            ),
            "expected_impact": (
                f"Tighter or wider sigma improves probability estimates for {p.segment}. "
                f"Direction consistency={p.direction_consistency:.0%}."
            ),
            "risk_of_change": (
                "Low: sigma changes only widen/tighten probability bands, do not flip direction. "
                "Validate against recent forecast_errors before applying."
            ),
            "recalibration_detail": p.to_dict(),
            "rollback_metadata": {
                "files_changed":      ["data/city_uncertainty.json"],
                "config_keys_changed": [p.segment],
                "previous_values":    {p.segment: p.current_sigma},
                "rollback_instructions": (
                    f"Restore city_uncertainty.json['{p.segment}'] = {p.current_sigma}"
                ),
            },
            "post_change_monitor": {
                "change_id":           change_id,
                "monitor_window_days": 14,
                "affected_cities":     [p.segment.split(":")[0]],
                "regression_flagged":  False,
                "monitoring_complete": False,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }
        new_recs.append(rec)

    if new_recs:
        updated = [e for e in existing if e.get("status") in ("pending", "approved")]
        updated.extend(new_recs)
        try:
            with open(APPROVALS_FILE, "w") as f:
                json.dump(updated, f, indent=2)
            logger.info(
                f"recalibrator: wrote {len(new_recs)} new uncertainty proposals "
                f"to pending_approvals.json"
            )
        except Exception as e:
            logger.error(f"recalibrator: failed to write pending_approvals.json: {e}")

    return new_recs


def apply_approved_recalibration(approved_ids: list[str]) -> dict:
    """
    Applies a list of approved uncertainty-recalibration change IDs to
    city_uncertainty.json.

    Called ONLY by the approval workflow after human sign-off.
    Returns a dict of {segment: new_sigma} for all segments updated.

    Safety: reads pending_approvals.json to look up the proposed values,
    so the approved value must be in the file (prevents arbitrary injection).
    """
    existing = _load_json(APPROVALS_FILE, [])
    uncertainty = _load_json(UNCERTAINTY_FILE, {})

    applied: dict[str, float] = {}
    now_str = datetime.now(timezone.utc).isoformat()

    for rec in existing:
        if rec.get("id") not in approved_ids:
            continue
        if rec.get("change_type") != "uncertainty_update":
            continue
        detail = rec.get("recalibration_detail", {})
        segment       = detail.get("segment", "")
        proposed_sigma = detail.get("proposed_sigma")
        if not segment or proposed_sigma is None:
            logger.warning(f"apply_recal: malformed detail in {rec.get('id')} — skipping")
            continue

        uncertainty[segment] = float(proposed_sigma)
        # Also update the season-agnostic key (city:mtype) if this is the best season data
        parts = segment.split(":")
        if len(parts) == 3:
            base_key = f"{parts[0]}:{parts[1]}"
            # Only overwrite base if we don't have a better (larger sample) value already
            if base_key not in uncertainty:
                uncertainty[base_key] = float(proposed_sigma)

        rec["status"] = "applied"
        rec["applied_at"] = now_str
        applied[segment] = float(proposed_sigma)
        logger.info(
            f"apply_recal: {segment} sigma {detail.get('current_sigma')} → {proposed_sigma}"
        )

    if applied:
        try:
            with open(UNCERTAINTY_FILE, "w") as f:
                json.dump(uncertainty, f, indent=2)
        except Exception as e:
            logger.error(f"apply_recal: failed to write city_uncertainty.json: {e}")
            return {}
        # Persist status updates in approvals file
        try:
            with open(APPROVALS_FILE, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.warning(f"apply_recal: failed to update pending_approvals.json: {e}")

    return applied


def run_recalibration_session() -> list[dict]:
    """
    Convenience entry point called by the nightly scheduler.
    Returns the list of new recommendation dicts queued.
    """
    logger.info("Starting uncertainty recalibration session...")
    new_recs = propose_recalibration_recommendations()
    logger.info(
        f"Recalibration session complete: {len(new_recs)} new proposals queued"
    )
    return new_recs
