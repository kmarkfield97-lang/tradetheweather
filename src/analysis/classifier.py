"""
Trade finding classifier for the self-analysis and auto-iteration framework.

Severity classes:
  P0 — bug / execution failure / broken safety behavior (auto-fix if safe)
  P1 — clear logic defect / strong fix candidate (manual review required)
  P2 — strategy iteration candidate (needs evidence threshold met)
  P3 — observation / needs more data

Evidence thresholds and change-type guardrails are enforced here.
The balance-first philosophy is encoded as: recommendations that increase
risk face a higher required evidence burden than recommendations that reduce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# ── Minimum sample sizes before a pattern becomes actionable ─────────────────
MIN_SAMPLES_MONITOR = 5       # below this: P3 only
MIN_SAMPLES_CONSIDER = 15     # below this: raise edge, do not avoid
MIN_SAMPLES_PENALIZE = 30     # at or above this with bad stats: strong penalty
MIN_SAMPLES_AVOID = 50        # at or above this with persistent losses: may avoid

# ── Win-rate thresholds ───────────────────────────────────────────────────────
POOR_WIN_RATE = 0.40          # below this → underperforming
ACCEPTABLE_WIN_RATE = 0.50

# ── P&L thresholds for city penalties (dollars, cumulative) ──────────────────
CITY_PENALTY_SOFT_THRESHOLD = -3.0    # raise edge by 3¢
CITY_PENALTY_STRONG_THRESHOLD = -8.0  # raise edge by 7¢
CITY_AVOID_THRESHOLD = -15.0          # may temporarily avoid if large sample

# ── Evidence strength labels ─────────────────────────────────────────────────
EvidenceStrength = Literal["anecdotal", "weak", "moderate", "strong", "conclusive"]

# ── Severity levels ───────────────────────────────────────────────────────────
Severity = Literal["P0", "P1", "P2", "P3"]

# ── Recommended actions ───────────────────────────────────────────────────────
RecommendedAction = Literal[
    "fix_now",
    "auto_apply_safe",
    "queue_for_review",
    "monitor_only",
    "collect_more_data",
]

# ── Change categories and their auto-apply policy ────────────────────────────
# "safe"   → allowed to auto-apply after validation
# "review" → must go to manual review queue
# "never"  → never auto-apply without explicit user sign-off
CHANGE_TYPE_POLICY: dict[str, str] = {
    # Safe / fast-track
    "logging_improvement": "safe",
    "error_handling": "safe",
    "stale_data_guard": "safe",
    "duplicate_order_prevention": "safe",
    "config_tightening": "safe",          # only within pre-approved safe ranges
    "diagnostics_alert": "safe",
    "bug_fix_direct_evidence": "safe",    # must have log proof
    # Requires manual review
    "probability_formula": "review",
    "edge_threshold": "review",
    "sizing_adjustment": "review",
    "exposure_cap": "review",
    "filter_change": "review",
    "signal_weight": "review",
    "sell_priority": "review",
    "avoid_city": "review",
    "uncertainty_update": "review",   # city_uncertainty.json sigma changes
    # Never auto-apply
    "disable_brake": "never",
    "increase_max_position": "never",
    "remove_safety_control": "never",
    "execution_routing": "never",
    "api_auth": "never",
    "weaken_portfolio_protection": "never",
}

# Root-cause tag keys (all boolean yes/no)
ROOT_CAUSE_TAGS = [
    "entry_quality_poor",
    "exit_quality_poor",
    "model_error",
    "execution_issue",
    "sizing_issue",
    "liquidity_issue",
    "stale_data_issue",
    "weather_regime_issue",
    "forecast_error_driver",
    "station_bias_driver",
    "late_day_path_failure",
    "should_have_traded",
    "should_not_have_traded",
    "should_have_exited_earlier",
    "should_have_held_longer",
    "correlation_issue",
]


@dataclass
class RootCauseTags:
    entry_quality: Literal["good", "neutral", "poor"] = "neutral"
    exit_quality: Literal["good", "neutral", "poor"] = "neutral"
    model_error: bool = False
    execution_issue: bool = False
    sizing_issue: bool = False
    liquidity_issue: bool = False
    stale_data_issue: bool = False
    weather_regime_issue: bool = False
    forecast_error_driver: bool = False
    station_bias_driver: bool = False
    late_day_path_failure: bool = False
    should_have_traded: bool = False          # for missed opportunities
    should_not_have_traded: bool = False
    should_have_exited_earlier: bool = False
    should_have_held_longer: bool = False
    correlation_issue: bool = False

    def to_dict(self) -> dict:
        return {
            "entry_quality": self.entry_quality,
            "exit_quality": self.exit_quality,
            "model_error": self.model_error,
            "execution_issue": self.execution_issue,
            "sizing_issue": self.sizing_issue,
            "liquidity_issue": self.liquidity_issue,
            "stale_data_issue": self.stale_data_issue,
            "weather_regime_issue": self.weather_regime_issue,
            "forecast_error_driver": self.forecast_error_driver,
            "station_bias_driver": self.station_bias_driver,
            "late_day_path_failure": self.late_day_path_failure,
            "should_have_traded": self.should_have_traded,
            "should_not_have_traded": self.should_not_have_traded,
            "should_have_exited_earlier": self.should_have_exited_earlier,
            "should_have_held_longer": self.should_have_held_longer,
            "correlation_issue": self.correlation_issue,
        }


@dataclass
class OutcomeDecomposition:
    """
    Estimates the primary driver of a trade's outcome.
    Each field is a fraction in [0, 1]; they should sum to ~1.0.
    """
    bad_prediction: float = 0.0       # model was directionally wrong
    bad_execution: float = 0.0        # order routing / timing / price issue
    bad_sizing: float = 0.0           # position too large given uncertainty
    bad_exit_timing: float = 0.0      # held too long or exited too early
    unavoidable_variance: float = 0.0 # outcome within normal randomness of the market

    def to_dict(self) -> dict:
        return {
            "bad_prediction": round(self.bad_prediction, 3),
            "bad_execution": round(self.bad_execution, 3),
            "bad_sizing": round(self.bad_sizing, 3),
            "bad_exit_timing": round(self.bad_exit_timing, 3),
            "unavoidable_variance": round(self.unavoidable_variance, 3),
        }


@dataclass
class StructuredLesson:
    """Full structured analysis for a single closed trade."""
    ticker: str
    outcome: Literal["win", "loss", "scratch"]
    pnl_dollars: float
    # Human-readable
    lesson_text: str
    # Machine-readable
    severity: Severity
    confidence: float                       # 0–1
    evidence_strength: EvidenceStrength
    root_cause: RootCauseTags = field(default_factory=RootCauseTags)
    decomposition: OutcomeDecomposition = field(default_factory=OutcomeDecomposition)
    suggested_action: RecommendedAction = "collect_more_data"
    needs_more_data: bool = True
    safe_for_auto_fix: bool = False
    change_type: Optional[str] = None       # key in CHANGE_TYPE_POLICY
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "outcome": self.outcome,
            "pnl_dollars": round(self.pnl_dollars, 2),
            "lesson_text": self.lesson_text,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "evidence_strength": self.evidence_strength,
            "root_cause": self.root_cause.to_dict(),
            "decomposition": self.decomposition.to_dict(),
            "suggested_action": self.suggested_action,
            "needs_more_data": self.needs_more_data,
            "safe_for_auto_fix": self.safe_for_auto_fix,
            "change_type": self.change_type,
            "notes": self.notes,
        }


@dataclass
class CityPerformancePenalty:
    """
    Confidence-weighted city penalty, replacing the blunt avoid-list.
    Distinguishes bad model, bad luck, bad execution, and regime shift.
    """
    city: str
    sample_size: int
    total_pnl: float
    win_rate: float
    # Derived classification
    classification: Literal[
        "insufficient_data",
        "possible_bad_luck",
        "probable_model_issue",
        "probable_execution_issue",
        "probable_regime_shift",
        "persistent_underperformance",
    ] = "insufficient_data"
    # Action
    action: Literal["monitor", "raise_edge_soft", "raise_edge_strong", "avoid"] = "monitor"
    edge_penalty_cents: float = 0.0         # additional cents required above normal min edge
    confidence: float = 0.0                 # 0–1 in this classification

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "sample_size": self.sample_size,
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate, 3),
            "classification": self.classification,
            "action": self.action,
            "edge_penalty_cents": round(self.edge_penalty_cents, 1),
            "confidence": round(self.confidence, 3),
        }


def classify_city_penalty(
    city: str,
    pnls: list[float],
    exit_reasons: Optional[list[str]] = None,
) -> CityPerformancePenalty:
    """
    Replaces the blunt avoid-list with a confidence-weighted penalty.

    Heuristics:
    - Small sample → monitor only, no hard penalty
    - Medium sample + poor win rate + meaningful losses → raise edge
    - Large sample + persistent loss + not explained by execution → consider avoid
    - If losses are dominated by execution_issue exits (thesis_invalidated, stale),
      classify as execution vs model problem
    """
    n = len(pnls)
    total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / n if n > 0 else 0.0

    penalty = CityPerformancePenalty(
        city=city,
        sample_size=n,
        total_pnl=total,
        win_rate=win_rate,
    )

    if n < MIN_SAMPLES_MONITOR:
        penalty.classification = "insufficient_data"
        penalty.action = "monitor"
        penalty.confidence = 0.1
        return penalty

    if n < MIN_SAMPLES_CONSIDER:
        # Small-but-real sample
        if win_rate < POOR_WIN_RATE and total < CITY_PENALTY_SOFT_THRESHOLD:
            penalty.classification = "possible_bad_luck"
            penalty.action = "monitor"
            penalty.confidence = 0.3
        else:
            penalty.classification = "insufficient_data"
            penalty.action = "monitor"
            penalty.confidence = 0.2
        return penalty

    # Detect if losses are exit-reason-driven (execution) vs model (wrong direction)
    execution_biased = False
    if exit_reasons:
        exec_exits = sum(
            1 for r in exit_reasons
            if r and any(tag in r for tag in ["thesis_invalidated", "stale", "salvage"])
        )
        if exec_exits / len(exit_reasons) > 0.5:
            execution_biased = True

    if n < MIN_SAMPLES_PENALIZE:
        if win_rate < POOR_WIN_RATE and total < CITY_PENALTY_SOFT_THRESHOLD:
            penalty.classification = (
                "probable_execution_issue" if execution_biased else "probable_model_issue"
            )
            penalty.action = "raise_edge_soft"
            penalty.edge_penalty_cents = 3.0
            penalty.confidence = 0.45
        elif total < CITY_PENALTY_STRONG_THRESHOLD:
            penalty.classification = "probable_model_issue"
            penalty.action = "raise_edge_soft"
            penalty.edge_penalty_cents = 3.0
            penalty.confidence = 0.4
        else:
            penalty.classification = "possible_bad_luck"
            penalty.action = "monitor"
            penalty.confidence = 0.35
        return penalty

    # Large sample — stronger signal
    if win_rate < POOR_WIN_RATE and total < CITY_PENALTY_STRONG_THRESHOLD:
        if execution_biased:
            penalty.classification = "probable_execution_issue"
            penalty.action = "raise_edge_soft"
            penalty.edge_penalty_cents = 3.0
            penalty.confidence = 0.6
        else:
            penalty.classification = "persistent_underperformance"
            penalty.action = "raise_edge_strong"
            penalty.edge_penalty_cents = 7.0
            penalty.confidence = 0.65

    if n >= MIN_SAMPLES_AVOID and total < CITY_AVOID_THRESHOLD and win_rate < POOR_WIN_RATE:
        penalty.classification = "persistent_underperformance"
        penalty.action = "avoid"
        penalty.edge_penalty_cents = 10.0
        penalty.confidence = 0.75

    return penalty


def classify_finding_severity(
    change_type: str,
    evidence: EvidenceStrength,
    sample_size: int,
    has_log_proof: bool = False,
    increases_risk: bool = False,
) -> tuple[Severity, RecommendedAction, bool]:
    """
    Returns (severity, recommended_action, safe_for_auto_fix).

    balance-first: recommendations that increase risk require stronger evidence.
    """
    policy = CHANGE_TYPE_POLICY.get(change_type, "review")

    # P0: direct bug evidence in logs — only for safe change types
    if change_type in (
        "stale_data_guard", "duplicate_order_prevention",
        "error_handling", "logging_improvement", "bug_fix_direct_evidence",
    ) and has_log_proof:
        return "P0", "auto_apply_safe", (policy == "safe")

    # P0: never-apply changes cannot be auto-applied regardless
    if policy == "never":
        return "P0" if has_log_proof else "P1", "queue_for_review", False

    # Evidence strength → severity mapping
    evidence_rank = {
        "anecdotal": 0, "weak": 1, "moderate": 2, "strong": 3, "conclusive": 4
    }
    rank = evidence_rank.get(evidence, 0)

    # Risk-increasing changes need higher bar
    required_rank = 3 if increases_risk else 2

    if rank >= 4 and sample_size >= MIN_SAMPLES_PENALIZE:
        sev: Severity = "P1"
        action: RecommendedAction = "fix_now" if policy == "safe" else "queue_for_review"
        safe = (policy == "safe") and not increases_risk
    elif rank >= required_rank and sample_size >= MIN_SAMPLES_CONSIDER:
        sev = "P2"
        action = "queue_for_review"
        safe = False
    elif rank >= 1 and sample_size >= MIN_SAMPLES_MONITOR:
        sev = "P3"
        action = "monitor_only"
        safe = False
    else:
        sev = "P3"
        action = "collect_more_data"
        safe = False

    return sev, action, safe


def derive_structured_lesson(position: dict, won: bool) -> StructuredLesson:
    """
    Generates a StructuredLesson for a closed position.

    Uses available position fields to infer root cause and decomposition.
    All inferences are conservative — defaults to 'unavoidable_variance'
    when evidence is insufficient.
    """
    ticker = position.get("ticker", "")
    pnl = position.get("pnl_dollars", 0.0)
    side = position.get("side", "")
    entry = position.get("entry_price", 0)
    exit_p = position.get("exit_price")
    exit_reason = position.get("exit_reason", "")
    market_type = position.get("market_type", "")
    city = position.get("city", "")
    model_unc = position.get("model_uncertainty", 0.3)

    outcome: Literal["win", "loss", "scratch"]
    if pnl > 0.05:
        outcome = "win"
    elif pnl < -0.05:
        outcome = "loss"
    else:
        outcome = "scratch"

    tags = RootCauseTags()
    decomp = OutcomeDecomposition()

    # ── Entry quality ────────────────────────────────────────────────────────
    if model_unc > 0.6:
        tags.entry_quality = "poor"
        tags.sizing_issue = True
    elif model_unc < 0.25:
        tags.entry_quality = "good"
    else:
        tags.entry_quality = "neutral"

    # ── Exit quality ────────────────────────────────────────────────────────
    if exit_reason:
        if "staged_profit" in exit_reason or "trailing_stop" in exit_reason:
            tags.exit_quality = "good" if won else "neutral"
        elif "thesis_invalidated" in exit_reason:
            tags.exit_quality = "good"      # correct to cut quickly
            tags.forecast_error_driver = True
        elif "salvage_stop" in exit_reason:
            tags.exit_quality = "poor"      # position deteriorated to fail-safe
        elif "fair_value" in exit_reason and not won:
            tags.exit_quality = "neutral"
        elif "daily_brake" in exit_reason:
            tags.exit_quality = "neutral"

    # ── Late-day path failure ────────────────────────────────────────────────
    if market_type in ("temp_high", "temp_low") and not won:
        if exit_reason and "thesis_invalidated" in exit_reason:
            tags.late_day_path_failure = True

    # ── Outcome decomposition ────────────────────────────────────────────────
    if outcome == "loss":
        if tags.stale_data_issue or tags.execution_issue:
            decomp.bad_execution = 0.6
            decomp.unavoidable_variance = 0.4
        elif tags.forecast_error_driver or tags.model_error:
            decomp.bad_prediction = 0.6
            decomp.unavoidable_variance = 0.4
        elif tags.sizing_issue:
            decomp.bad_sizing = 0.4
            decomp.bad_prediction = 0.3
            decomp.unavoidable_variance = 0.3
        elif tags.exit_quality == "poor":
            decomp.bad_exit_timing = 0.5
            decomp.unavoidable_variance = 0.5
        else:
            # No strong signal — attribute to variance
            decomp.unavoidable_variance = 0.7
            decomp.bad_prediction = 0.3
    else:
        # Win — attribute primarily to correct prediction, some variance
        decomp.bad_prediction = 0.0
        decomp.unavoidable_variance = 0.4

    # ── Severity classification ──────────────────────────────────────────────
    if tags.execution_issue or tags.stale_data_issue:
        sev: Severity = "P0"
        action: RecommendedAction = "auto_apply_safe"
        safe = True
        change_type = "stale_data_guard" if tags.stale_data_issue else "error_handling"
        evidence: EvidenceStrength = "moderate"
    elif tags.model_error and not won:
        sev = "P2"
        action = "queue_for_review"
        safe = False
        change_type = "probability_formula"
        evidence = "weak"
    elif outcome == "loss" and decomp.bad_prediction >= 0.5:
        sev = "P3"
        action = "collect_more_data"
        safe = False
        change_type = None
        evidence = "anecdotal"
    else:
        sev = "P3"
        action = "collect_more_data"
        safe = False
        change_type = None
        evidence = "anecdotal"

    # ── Human-readable lesson ────────────────────────────────────────────────
    if won:
        lesson_text = (
            f"WIN [{sev}]: {ticker} {side.upper()} entry={entry}¢ "
            f"exit={exit_p}¢ pnl=${pnl:+.2f}. "
            f"Entry quality: {tags.entry_quality}. Signal was correct."
        )
    else:
        decomp_dict = decomp.to_dict()
        primary_driver = max(decomp_dict, key=lambda k: decomp_dict[k])
        lesson_text = (
            f"LOSS [{sev}]: {ticker} {side.upper()} entry={entry}¢ "
            f"exit={exit_p}¢ pnl=${pnl:+.2f}. "
            f"Primary driver: {primary_driver}. "
            f"Exit: {exit_reason or 'unknown'}. "
            f"Action: {action}."
        )

    confidence = 0.3 if outcome != "scratch" else 0.1
    if tags.execution_issue or tags.stale_data_issue:
        confidence = 0.7

    return StructuredLesson(
        ticker=ticker,
        outcome=outcome,
        pnl_dollars=pnl,
        lesson_text=lesson_text,
        severity=sev,
        confidence=confidence,
        evidence_strength=evidence,
        root_cause=tags,
        decomposition=decomp,
        suggested_action=action,
        needs_more_data=(action in ("collect_more_data", "monitor_only")),
        safe_for_auto_fix=safe,
        change_type=change_type,
    )


def compute_evidence_strength(sample_size: int, consistency: float) -> EvidenceStrength:
    """
    Converts sample size + consistency (0–1 fraction of observations pointing same direction)
    into an EvidenceStrength label.
    """
    if sample_size < MIN_SAMPLES_MONITOR:
        return "anecdotal"
    if sample_size < MIN_SAMPLES_CONSIDER:
        return "weak" if consistency >= 0.6 else "anecdotal"
    if sample_size < MIN_SAMPLES_PENALIZE:
        if consistency >= 0.75:
            return "moderate"
        return "weak"
    if sample_size < MIN_SAMPLES_AVOID:
        if consistency >= 0.80:
            return "strong"
        return "moderate"
    # Large sample
    if consistency >= 0.85:
        return "conclusive"
    return "strong"
