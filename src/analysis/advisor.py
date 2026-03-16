"""
Structured advisor module.

Replaces the old GPT advisor's free-form additive/update/replace output with
a fully structured recommendation format that includes severity, confidence,
evidence strength, change-type guardrails, auto-apply eligibility, and
rollback metadata.

The advisor analyses the learning log, forecast errors, and trade history,
then produces AdvisorRecommendation objects that are stored in
data/pending_approvals.json for the Telegram approval flow.

Balance-first philosophy:
  - Recommendations that increase risk require stronger evidence
  - Auto-apply is only permitted for clearly safe change categories
  - The rollback_metadata field must be populated before any change is applied
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.analysis.classifier import (
    CHANGE_TYPE_POLICY,
    CityPerformancePenalty,
    EvidenceStrength,
    RecommendedAction,
    Severity,
    classify_city_penalty,
    classify_finding_severity,
    compute_evidence_strength,
    MIN_SAMPLES_MONITOR,
    MIN_SAMPLES_CONSIDER,
    MIN_SAMPLES_PENALIZE,
)

# ── Adaptation maturity tiers ─────────────────────────────────────────────────
# Every recommendation is classified into one of these tiers before it can be
# acted on. A recommendation may only advance to safe_config_update after it has
# been validated (see validation.py). The advisor assigns an initial tier; the
# validation framework may promote or demote it.
#
# Ordering from least to most ready:
#   observe_only → collect_more_data → test_in_shadow_mode →
#   validate_in_backtest → safe_config_update → manual_review_required
AdaptationMaturity = str   # one of the values below

MATURITY_OBSERVE_ONLY       = "observe_only"
MATURITY_COLLECT_MORE_DATA  = "collect_more_data"
MATURITY_SHADOW_MODE        = "test_in_shadow_mode"
MATURITY_BACKTEST           = "validate_in_backtest"
MATURITY_SAFE_CONFIG        = "safe_config_update"
MATURITY_MANUAL_REVIEW      = "manual_review_required"

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
APPROVALS_FILE    = os.path.join(DATA_DIR, "pending_approvals.json")
ADVICE_LOG_FILE   = os.path.join(DATA_DIR, "gpt_advice_log.json")
ERRORS_FILE       = os.path.join(DATA_DIR, "forecast_errors.json")
HISTORY_FILE      = os.path.join(DATA_DIR, "trade_history.json")
LEARNING_LOG_FILE = os.path.join(DATA_DIR, "learning_log.json")
CHANGELOG_FILE    = os.path.join(DATA_DIR, "CHANGELOG.md")


def _infer_adaptation_maturity(
    change_type: str,
    evidence: EvidenceStrength,
    sample_size: int,
    has_validation_result: bool = False,
    validation_passed: bool = False,
) -> AdaptationMaturity:
    """
    Assigns the initial adaptation maturity tier for a recommendation.

    Rules (conservative — err on the side of collecting more data):
    - Bug fixes / safe infra changes with log proof → safe_config_update
    - Never-apply change types → manual_review_required
    - conclusive evidence + sample_size >= 50 + passed validation → safe_config_update
    - strong + validated → validate_in_backtest or safe_config_update
    - moderate → test_in_shadow_mode
    - weak or small sample → collect_more_data
    - anecdotal / tiny sample → observe_only
    """
    policy = CHANGE_TYPE_POLICY.get(change_type, "review")

    # Never-auto-apply category
    if policy == "never":
        return MATURITY_MANUAL_REVIEW

    # Infrastructure / bug-fix category (safe policy)
    if policy == "safe":
        if evidence in ("conclusive", "strong", "moderate"):
            return MATURITY_SAFE_CONFIG
        return MATURITY_COLLECT_MORE_DATA

    # Strategy / parameter changes (review policy)
    evidence_rank = {
        "anecdotal": 0, "weak": 1, "moderate": 2, "strong": 3, "conclusive": 4
    }
    rank = evidence_rank.get(evidence, 0)

    if rank == 0 or sample_size < MIN_SAMPLES_MONITOR:
        return MATURITY_OBSERVE_ONLY
    if rank == 1 or sample_size < MIN_SAMPLES_CONSIDER:
        return MATURITY_COLLECT_MORE_DATA
    if rank == 2:
        return MATURITY_SHADOW_MODE
    if rank >= 3 and sample_size >= MIN_SAMPLES_PENALIZE:
        if has_validation_result and validation_passed:
            return MATURITY_SAFE_CONFIG
        return MATURITY_BACKTEST
    return MATURITY_COLLECT_MORE_DATA


@dataclass
class RollbackMetadata:
    """Records what was changed so it can be undone."""
    files_changed: list[str] = field(default_factory=list)
    config_keys_changed: list[str] = field(default_factory=list)
    previous_values: dict = field(default_factory=dict)
    rollback_instructions: str = ""

    def to_dict(self) -> dict:
        return {
            "files_changed": self.files_changed,
            "config_keys_changed": self.config_keys_changed,
            "previous_values": self.previous_values,
            "rollback_instructions": self.rollback_instructions,
        }


@dataclass
class PostChangeMonitor:
    """Tags for monitoring the effect of an applied change."""
    change_id: str = ""
    applied_at: str = ""
    monitor_window_days: int = 7
    baseline_win_rate: float = 0.0
    baseline_avg_pnl: float = 0.0
    affected_cities: list[str] = field(default_factory=list)
    affected_market_types: list[str] = field(default_factory=list)
    regression_flagged: bool = False
    monitoring_complete: bool = False

    def to_dict(self) -> dict:
        return {
            "change_id": self.change_id,
            "applied_at": self.applied_at,
            "monitor_window_days": self.monitor_window_days,
            "baseline_win_rate": round(self.baseline_win_rate, 3),
            "baseline_avg_pnl": round(self.baseline_avg_pnl, 2),
            "affected_cities": self.affected_cities,
            "affected_market_types": self.affected_market_types,
            "regression_flagged": self.regression_flagged,
            "monitoring_complete": self.monitoring_complete,
        }


@dataclass
class AdvisorRecommendation:
    """
    Structured recommendation from the advisor.
    Replaces the old free-form additive/update/replace dict.
    """
    id: str                                 # unique e.g. "2026-03-15_city_penalty_denver"
    title: str
    category: str                           # human label e.g. "city_performance"
    change_type: str                        # key in CHANGE_TYPE_POLICY
    severity: Severity
    confidence: float                       # 0–1
    evidence_strength: EvidenceStrength
    sample_size: int
    affected_markets: list[str] = field(default_factory=list)
    affected_cities: list[str] = field(default_factory=list)
    affected_market_types: list[str] = field(default_factory=list)
    likely_root_cause: str = ""
    recommended_action: RecommendedAction = "collect_more_data"
    auto_apply_allowed: bool = False        # True only for safe change types
    manual_review_required: bool = True
    rollback_risk: str = "low"              # "low" / "medium" / "high"
    proposed_change_summary: str = ""
    expected_impact: str = ""              # qualitative
    risk_of_change: str = ""               # qualitative
    rollback_metadata: RollbackMetadata = field(default_factory=RollbackMetadata)
    post_change_monitor: PostChangeMonitor = field(default_factory=PostChangeMonitor)
    # ── Adaptation maturity (new) ─────────────────────────────────────────────
    adaptation_maturity: AdaptationMaturity = MATURITY_COLLECT_MORE_DATA
    # Most recent validation result for this recommendation (None until validated)
    validation_result: Optional[dict] = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "pending"                # "pending" / "approved" / "rejected" / "applied"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "change_type": self.change_type,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "evidence_strength": self.evidence_strength,
            "sample_size": self.sample_size,
            "affected_markets": self.affected_markets,
            "affected_cities": self.affected_cities,
            "affected_market_types": self.affected_market_types,
            "likely_root_cause": self.likely_root_cause,
            "recommended_action": self.recommended_action,
            "auto_apply_allowed": self.auto_apply_allowed,
            "manual_review_required": self.manual_review_required,
            "rollback_risk": self.rollback_risk,
            "proposed_change_summary": self.proposed_change_summary,
            "expected_impact": self.expected_impact,
            "risk_of_change": self.risk_of_change,
            "rollback_metadata": self.rollback_metadata.to_dict(),
            "post_change_monitor": self.post_change_monitor.to_dict(),
            "adaptation_maturity": self.adaptation_maturity,
            "validation_result": self.validation_result,
            "generated_at": self.generated_at,
            "status": self.status,
        }


class StructuredAdvisor:
    """
    Analyses bot data and emits structured AdvisorRecommendation objects.
    Does not call any external LLM — pure data-driven analysis.
    The GPT/LLM layer (if re-enabled) should feed its output through this
    same dataclass so all recommendations are subject to the same guardrails.
    """

    def __init__(self):
        self._history = self._load_json(HISTORY_FILE, [])
        self._learning_log = self._load_json(LEARNING_LOG_FILE, [])
        self._errors_data = self._load_json(ERRORS_FILE, {"forecast_errors": [], "calibration": {}})
        self._applied_changes = self._load_changelog_ids()

    def _load_json(self, path: str, default):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default

    def _load_changelog_ids(self) -> set:
        """Extract change IDs from CHANGELOG.md to avoid re-suggesting the same thing."""
        ids: set = set()
        try:
            with open(CHANGELOG_FILE) as f:
                for line in f:
                    if line.startswith("### change_id:"):
                        ids.add(line.split(":", 1)[1].strip())
        except Exception:
            pass
        return ids

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def generate_recommendations(self) -> list[AdvisorRecommendation]:
        """
        Analyses all available data and returns a list of AdvisorRecommendations.
        Skips any change already recorded in the CHANGELOG.

        Each recommendation is assigned an adaptation_maturity tier and
        enriched with any existing validation result from validation_log.json.
        """
        from src.analysis.validation import get_validation_result_for

        recs: list[AdvisorRecommendation] = []

        recs.extend(self._analyze_city_performance())
        recs.extend(self._analyze_forecast_calibration())
        recs.extend(self._analyze_p0_findings())
        recs.extend(self._analyze_market_type_performance())
        recs.extend(self._analyze_exit_quality())

        # Filter already-applied changes
        recs = [r for r in recs if r.id not in self._applied_changes]

        # Enrich each recommendation with maturity tier + any cached validation result
        for rec in recs:
            val_result = get_validation_result_for(rec.id)
            validation_passed = (
                val_result.get("pass_validation", False) if val_result else False
            )
            rec.adaptation_maturity = _infer_adaptation_maturity(
                change_type=rec.change_type,
                evidence=rec.evidence_strength,
                sample_size=rec.sample_size,
                has_validation_result=(val_result is not None),
                validation_passed=validation_passed,
            )
            rec.validation_result = val_result

            # If maturity has been validated and passed, auto_apply_allowed may be
            # promoted for safe change types — but never for "review" or "never" policy.
            policy = CHANGE_TYPE_POLICY.get(rec.change_type, "review")
            if (rec.adaptation_maturity == MATURITY_SAFE_CONFIG
                    and policy == "safe"
                    and validation_passed):
                rec.auto_apply_allowed = True
                rec.manual_review_required = False

        # Sort by severity (P0 first), then confidence descending
        sev_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        recs.sort(key=lambda r: (sev_order.get(r.severity, 9), -r.confidence))

        return recs

    def save_recommendations(self, recs: list[AdvisorRecommendation]):
        """
        Writes recommendations to pending_approvals.json.
        Preserves existing pending items that have not been resolved.
        """
        existing = self._load_json(APPROVALS_FILE, [])
        # Keep existing items that are still pending or approved-not-yet-applied
        keep = [
            e for e in existing
            if e.get("status") in ("pending", "approved")
            and e.get("id") not in {r.id for r in recs}
        ]
        updated = keep + [r.to_dict() for r in recs]
        with open(APPROVALS_FILE, "w") as f:
            json.dump(updated, f, indent=2)
        logger.info(f"Saved {len(recs)} new recommendations to pending_approvals.json")

    def log_advice_session(self, recs: list[AdvisorRecommendation]):
        """Appends this advice session to gpt_advice_log.json."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendations_count": len(recs),
            "recommendations": [r.to_dict() for r in recs],
        }
        log = self._load_json(ADVICE_LOG_FILE, [])
        log.append(entry)
        log = log[-30:]  # keep last 30 sessions
        with open(ADVICE_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

    # -------------------------------------------------------------------------
    # Analysis sub-routines
    # -------------------------------------------------------------------------

    def _all_closed_positions(self) -> list[dict]:
        positions = []
        for record in self._history:
            for pos in record.get("positions", []):
                if pos.get("status") in ("closed", "expired"):
                    positions.append(pos)
        return positions

    def _analyze_city_performance(self) -> list[AdvisorRecommendation]:
        """Generates city-level edge penalty recommendations."""
        city_pnl: dict[str, list[float]] = {}
        city_exit_reasons: dict[str, list[str]] = {}
        for pos in self._all_closed_positions():
            city = pos.get("city", "")
            if not city:
                continue
            city_pnl.setdefault(city, []).append(pos.get("pnl_dollars", 0.0))
            city_exit_reasons.setdefault(city, []).append(pos.get("exit_reason", "") or "")

        recs = []
        for city, pnls in city_pnl.items():
            if len(pnls) < MIN_SAMPLES_MONITOR:
                continue
            penalty = classify_city_penalty(city, pnls, city_exit_reasons.get(city, []))
            if penalty.action == "monitor":
                continue

            change_id = f"city_penalty_{city.lower()}_{len(pnls)}t"
            sev, action, safe = classify_finding_severity(
                change_type="avoid_city" if penalty.action == "avoid" else "edge_threshold",
                evidence=compute_evidence_strength(
                    len(pnls),
                    (sum(1 for p in pnls if p < 0) / len(pnls)) if pnls else 0,
                ),
                sample_size=len(pnls),
                has_log_proof=False,
                increases_risk=False,  # raising edge reduces risk
            )

            policy = CHANGE_TYPE_POLICY.get("avoid_city", "review")
            rec = AdvisorRecommendation(
                id=change_id,
                title=f"City performance penalty: {city}",
                category="city_performance",
                change_type="avoid_city" if penalty.action == "avoid" else "edge_threshold",
                severity=sev,
                confidence=penalty.confidence,
                evidence_strength=compute_evidence_strength(
                    len(pnls),
                    sum(1 for p in pnls if p < 0) / len(pnls),
                ),
                sample_size=len(pnls),
                affected_cities=[city],
                likely_root_cause=penalty.classification,
                recommended_action=action,
                auto_apply_allowed=(policy == "safe" and safe),
                manual_review_required=(policy != "safe"),
                rollback_risk="low",
                proposed_change_summary=(
                    f"Apply {penalty.action} for {city}: "
                    f"edge penalty +{penalty.edge_penalty_cents:.0f}¢ "
                    f"(classification: {penalty.classification}, "
                    f"win_rate={penalty.win_rate:.0%}, total_pnl=${penalty.total_pnl:+.2f})"
                ),
                expected_impact=f"Reduce losses from {city} by requiring larger edge",
                risk_of_change=f"Misses real edges in {city} if classification is wrong",
                post_change_monitor=PostChangeMonitor(
                    change_id=change_id,
                    monitor_window_days=14,
                    affected_cities=[city],
                    baseline_win_rate=(sum(1 for p in pnls if p > 0) / len(pnls)),
                    baseline_avg_pnl=(sum(pnls) / len(pnls)),
                ),
            )
            recs.append(rec)
        return recs

    def _analyze_forecast_calibration(self) -> list[AdvisorRecommendation]:
        """Generates calibration recommendations by city+market_type+season."""
        errors = self._errors_data.get("forecast_errors", [])
        if not errors:
            return []

        # Group errors by city:market_type:season
        segments: dict[str, list[float]] = {}
        for err in errors:
            city = err.get("city", "")
            mtype = err.get("market_type", "")
            season = err.get("season", "")
            e = err.get("error")
            if city and mtype and e is not None:
                key = f"{city}:{mtype}:{season}" if season else f"{city}:{mtype}"
                segments.setdefault(key, []).append(float(e))

        recs = []
        for seg, errs in segments.items():
            n = len(errs)
            if n < MIN_SAMPLES_MONITOR:
                continue
            mean_err = statistics.mean(errs)
            consistency = sum(1 for e in errs if (e > 0) == (mean_err > 0)) / n
            evidence = compute_evidence_strength(n, consistency)

            # Only flag if mean error is materially large (>2°F)
            if abs(mean_err) < 2.0:
                continue

            change_id = f"calibration_{seg.replace(':', '_').lower()}_{n}t"
            sev, action, safe = classify_finding_severity(
                change_type="probability_formula",
                evidence=evidence,
                sample_size=n,
                has_log_proof=False,
                increases_risk=False,
            )

            rec = AdvisorRecommendation(
                id=change_id,
                title=f"Forecast calibration drift: {seg}",
                category="forecast_calibration",
                change_type="probability_formula",
                severity=sev,
                confidence=round(consistency, 3),
                evidence_strength=evidence,
                sample_size=n,
                likely_root_cause=(
                    f"NWS forecast runs {mean_err:+.1f}°F on average for {seg}"
                ),
                recommended_action=action,
                auto_apply_allowed=False,
                manual_review_required=True,
                rollback_risk="medium",
                proposed_change_summary=(
                    f"Station bias or city_uncertainty.json update for {seg}: "
                    f"mean_error={mean_err:+.1f}°F over {n} samples "
                    f"(consistency={consistency:.0%})"
                ),
                expected_impact="Reduce directional miscalibration in probability estimates",
                risk_of_change="Could over-correct if regime has shifted; needs pre-validation",
            )
            recs.append(rec)
        return recs

    def _analyze_p0_findings(self) -> list[AdvisorRecommendation]:
        """Surfaces P0 severity findings from structured lessons."""
        recs = []
        seen_types: set = set()

        for record in self._history[-14:]:  # last 14 days
            for lesson in record.get("structured_lessons", []):
                if lesson.get("severity") != "P0":
                    continue
                change_type = lesson.get("change_type") or "error_handling"
                if change_type in seen_types:
                    continue
                seen_types.add(change_type)

                change_id = f"p0_{change_type}_{record.get('date', '')}"
                policy = CHANGE_TYPE_POLICY.get(change_type, "review")
                rec = AdvisorRecommendation(
                    id=change_id,
                    title=f"P0 finding: {change_type}",
                    category="execution_bug",
                    change_type=change_type,
                    severity="P0",
                    confidence=lesson.get("confidence", 0.5),
                    evidence_strength=lesson.get("evidence_strength", "moderate"),
                    sample_size=1,
                    likely_root_cause=lesson.get("lesson_text", ""),
                    recommended_action="auto_apply_safe" if policy == "safe" else "fix_now",
                    auto_apply_allowed=(policy == "safe"),
                    manual_review_required=(policy != "safe"),
                    rollback_risk="low" if policy == "safe" else "medium",
                    proposed_change_summary=(
                        f"Fix {change_type} issue detected in {record.get('date', '')}: "
                        f"{lesson.get('lesson_text', '')[:200]}"
                    ),
                    expected_impact="Prevent recurrence of operational failure",
                    risk_of_change="Low if change is limited to the safe category",
                )
                recs.append(rec)
        return recs

    def _analyze_market_type_performance(self) -> list[AdvisorRecommendation]:
        """Flags market types with persistent underperformance."""
        type_pnl: dict[str, list[float]] = {}
        for pos in self._all_closed_positions():
            mtype = pos.get("market_type", "")
            if mtype:
                type_pnl.setdefault(mtype, []).append(pos.get("pnl_dollars", 0.0))

        recs = []
        for mtype, pnls in type_pnl.items():
            n = len(pnls)
            if n < MIN_SAMPLES_CONSIDER:
                continue
            win_rate = sum(1 for p in pnls if p > 0) / n
            total = sum(pnls)
            if win_rate >= 0.45 or total >= 0:
                continue  # not underperforming

            consistency = sum(1 for p in pnls if p < 0) / n
            evidence = compute_evidence_strength(n, consistency)
            change_id = f"market_type_{mtype}_{n}t"
            sev, action, safe = classify_finding_severity(
                "filter_change", evidence, n,
                has_log_proof=False, increases_risk=False,
            )

            recs.append(AdvisorRecommendation(
                id=change_id,
                title=f"Market type underperformance: {mtype}",
                category="market_type_performance",
                change_type="filter_change",
                severity=sev,
                confidence=round(consistency, 3),
                evidence_strength=evidence,
                sample_size=n,
                affected_market_types=[mtype],
                likely_root_cause=(
                    f"{mtype} win_rate={win_rate:.0%} total_pnl=${total:+.2f} over {n} trades"
                ),
                recommended_action=action,
                auto_apply_allowed=False,
                manual_review_required=True,
                rollback_risk="medium",
                proposed_change_summary=(
                    f"Review edge requirements or filters for {mtype}. "
                    f"Consider raising MIN_EDGE for this market type."
                ),
                expected_impact=f"Reduce loss rate on {mtype} contracts",
                risk_of_change="May miss valid setups if over-tightened",
            ))
        return recs

    def _analyze_exit_quality(self) -> list[AdvisorRecommendation]:
        """Flags exit quality patterns from structured lessons."""
        poor_exits: list[dict] = []
        for record in self._history[-30:]:
            for lesson in record.get("structured_lessons", []):
                rc = lesson.get("root_cause", {})
                if rc.get("exit_quality") == "poor":
                    poor_exits.append(lesson)

        n = len(poor_exits)
        if n < MIN_SAMPLES_MONITOR:
            return []

        salvage_count = sum(
            1 for l in poor_exits
            if "salvage_stop" in l.get("lesson_text", "")
        )
        late_day_count = sum(
            1 for l in poor_exits
            if l.get("root_cause", {}).get("late_day_path_failure")
        )

        recs = []
        if salvage_count >= 3:
            change_id = f"exit_salvage_pattern_{n}t"
            evidence = compute_evidence_strength(salvage_count, 0.8)
            sev, action, safe = classify_finding_severity(
                "sell_priority", evidence, salvage_count,
                has_log_proof=False, increases_risk=False,
            )
            recs.append(AdvisorRecommendation(
                id=change_id,
                title=f"Salvage stop pattern: {salvage_count} occurrences",
                category="exit_quality",
                change_type="sell_priority",
                severity=sev,
                confidence=0.55,
                evidence_strength=evidence,
                sample_size=salvage_count,
                likely_root_cause=(
                    "Positions reaching salvage stop suggest entries at wrong price or "
                    "thesis not invalidating early enough"
                ),
                recommended_action=action,
                auto_apply_allowed=False,
                manual_review_required=True,
                rollback_risk="medium",
                proposed_change_summary=(
                    "Review thesis_invalidation sensitivity or salvage stop threshold. "
                    "Consider tightening THESIS_TEMP_DIVERGENCE_F or SALVAGE_STOP_PCT."
                ),
                expected_impact="Reduce tail losses from positions reaching fail-safe exit",
                risk_of_change="May exit too early on recoverable positions",
            ))

        if late_day_count >= 3:
            change_id = f"exit_late_day_path_{late_day_count}t"
            evidence = compute_evidence_strength(late_day_count, 0.75)
            sev, action, safe = classify_finding_severity(
                "filter_change", evidence, late_day_count,
                has_log_proof=False, increases_risk=False,
            )
            recs.append(AdvisorRecommendation(
                id=change_id,
                title=f"Late-day path failure: {late_day_count} occurrences",
                category="exit_quality",
                change_type="filter_change",
                severity=sev,
                confidence=0.5,
                evidence_strength=evidence,
                sample_size=late_day_count,
                likely_root_cause=(
                    "Temp markets failing thesis late in the day — "
                    "possibly same-day cutoff too permissive or sigma too wide near close"
                ),
                recommended_action=action,
                auto_apply_allowed=False,
                manual_review_required=True,
                rollback_risk="medium",
                proposed_change_summary=(
                    "Review same-day cutoff hours or tighten MIN_EDGE for temp markets "
                    "in the 3–6h-to-close window."
                ),
                expected_impact="Reduce late-day path-dependent losses",
                risk_of_change="May miss valid late-day setups with genuine edge",
            ))
        return recs


def run_advisor_session() -> list[AdvisorRecommendation]:
    """
    Convenience function: run the advisor and save recommendations.
    Called by the orchestrator at EOD.
    """
    advisor = StructuredAdvisor()
    recs = advisor.generate_recommendations()
    advisor.save_recommendations(recs)
    advisor.log_advice_session(recs)
    logger.info(
        f"Advisor session complete: {len(recs)} recommendations "
        f"(P0={sum(1 for r in recs if r.severity=='P0')}, "
        f"P1={sum(1 for r in recs if r.severity=='P1')}, "
        f"P2={sum(1 for r in recs if r.severity=='P2')}, "
        f"P3={sum(1 for r in recs if r.severity=='P3')})"
    )
    return recs
