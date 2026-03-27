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

# ── P&L threshold separating wins/losses from scratches ──────────────────────
# Trades within ±SCRATCH_PNL_THRESHOLD are "scratch" (flat exit) — not wins or
# losses. Exported so history.py and advisor.py can use the same constant.
SCRATCH_PNL_THRESHOLD = 0.05

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
    # ── New tags added for richer diagnostics ────────────────────────────────
    low_edge_entry: bool = False       # edge was below tier threshold at entry
    poor_sell_timing: bool = False     # exit timing clearly suboptimal (evidence-based)
    halt_side_effects: bool = False    # daily brake forced an exit at bad price
    normal_variance: bool = False      # outcome within expected range for this sigma
    insufficient_telemetry: bool = False  # no entry snapshot stored — can't diagnose
    # ── Disagree-quality tags ────────────────────────────────────────────────
    low_price_dangerous_disagree: bool = False  # low-price + dangerous disagreement
    dangerous_disagree_market_right: bool = False  # disagreement classified dangerous & mkt won

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
            "low_edge_entry": self.low_edge_entry,
            "poor_sell_timing": self.poor_sell_timing,
            "halt_side_effects": self.halt_side_effects,
            "normal_variance": self.normal_variance,
            "insufficient_telemetry": self.insufficient_telemetry,
            "low_price_dangerous_disagree": self.low_price_dangerous_disagree,
            "dangerous_disagree_market_right": self.dangerous_disagree_market_right,
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
    # ── New diagnostic fields ─────────────────────────────────────────────────
    primary_root_cause: str = "insufficient_telemetry"  # stable string key for day-level rollup
    exit_reason: str = ""                   # normalised exit reason category
    mfe_cents: Optional[int] = None         # max favourable excursion (high_water_mark)
    mae_cents: Optional[int] = None         # max adverse excursion (low_water_mark)
    fragile_flags: list = field(default_factory=list)   # e.g. ["low_price_entry"]
    entry_snapshot_available: bool = False  # True when entry_our_prob etc. were stored

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
            "primary_root_cause": self.primary_root_cause,
            "exit_reason": self.exit_reason,
            "mfe_cents": self.mfe_cents,
            "mae_cents": self.mae_cents,
            "fragile_flags": self.fragile_flags,
            "entry_snapshot_available": self.entry_snapshot_available,
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
    # Exclude scratches from win-rate denominator so flat exits don't suppress win rate.
    wins = sum(1 for p in pnls if p > SCRATCH_PNL_THRESHOLD)
    scratch_count = sum(1 for p in pnls if abs(p) <= SCRATCH_PNL_THRESHOLD)
    decided = n - scratch_count
    win_rate = wins / decided if decided > 0 else 0.0

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


def derive_structured_lesson(position: dict) -> StructuredLesson:
    """
    Generates a StructuredLesson for a closed position.

    Uses available position fields to infer root cause and decomposition.
    All inferences are conservative — defaults to 'unavoidable_variance'
    when evidence is insufficient.

    The `won` parameter has been removed. Outcome is computed internally
    from pnl using the shared SCRATCH_PNL_THRESHOLD so scratches are never
    misclassified as losses.
    """
    ticker = position.get("ticker", "")
    pnl = position.get("pnl_dollars", 0.0)
    side = position.get("side", "")
    entry = position.get("entry_price", 0)
    exit_p = position.get("exit_price")
    exit_reason_raw = position.get("exit_reason", "") or ""
    market_type = position.get("market_type", "")
    city = position.get("city", "")
    model_unc = position.get("model_uncertainty", 0.3)

    # ── New fields from entry snapshot (optional — present only if logging active) ──
    entry_our_prob    = position.get("entry_our_prob")
    entry_edge        = position.get("entry_edge")
    entry_sigma       = position.get("entry_sigma")
    entry_hours_left  = position.get("entry_hours_left")
    mfe_cents         = position.get("high_water_mark")   # peak price seen
    mae_cents         = position.get("low_water_mark")    # worst price seen
    fragile_flags     = list(position.get("fragile_flags", []) or [])
    has_snapshot      = entry_our_prob is not None

    # ── Entry context (rich snapshot stored since recent logging improvements) ──
    entry_ctx         = position.get("entry_context") or {}
    ec_mip_verdict    = entry_ctx.get("mip_verdict", "NONE")
    ec_disagree_class = entry_ctx.get("disagree_classification", "actionable")
    ec_raw_disagree   = float(entry_ctx.get("raw_disagreement", 0.0))
    ec_is_low_price   = bool(entry_ctx.get("is_low_price_entry", False))
    ec_calib_level    = entry_ctx.get("calib_bias_level", "ok")
    ec_sigma          = float(entry_ctx.get("sigma_f", entry_sigma or 7.0))
    ec_signal_agree   = float(entry_ctx.get("signal_agreement", 0.5))

    # Normalised exit reason category (stable string)
    exit_reason_cat = _categorize_exit_reason_lesson(exit_reason_raw)

    # ── Three-class outcome ───────────────────────────────────────────────────
    outcome: Literal["win", "loss", "scratch"]
    if pnl > SCRATCH_PNL_THRESHOLD:
        outcome = "win"
    elif pnl < -SCRATCH_PNL_THRESHOLD:
        outcome = "loss"
    else:
        outcome = "scratch"

    won = (outcome == "win")

    tags = RootCauseTags()
    decomp = OutcomeDecomposition()

    # ── Entry quality ─────────────────────────────────────────────────────────
    if model_unc > 0.6:
        tags.entry_quality = "poor"
        tags.sizing_issue = True
    elif model_unc < 0.25:
        tags.entry_quality = "good"
    else:
        tags.entry_quality = "neutral"

    # Entry edge flag: edge < 7¢ is below the conservative tier threshold
    if entry_edge is not None and abs(entry_edge) < 0.07:
        tags.low_edge_entry = True

    # ── Exit quality ──────────────────────────────────────────────────────────
    if exit_reason_cat == "thesis_invalidation":
        tags.exit_quality = "good"      # correct to cut quickly
        tags.forecast_error_driver = True
        if market_type in ("temp_high", "temp_low") and outcome == "loss":
            tags.late_day_path_failure = True
    elif exit_reason_cat in ("staged_profit", "trailing_stop"):
        tags.exit_quality = "good" if won else "neutral"
    elif exit_reason_cat == "salvage":
        tags.exit_quality = "poor"      # position deteriorated to fail-safe
    elif exit_reason_cat == "daily_halt":
        tags.exit_quality = "neutral"
        tags.halt_side_effects = True
    elif exit_reason_cat == "fair_value":
        tags.exit_quality = "neutral"
    elif exit_reason_cat == "stalled_capital_trap":
        tags.exit_quality = "poor"              # position stagnated — capital was poorly deployed
        tags.should_have_exited_earlier = True  # stall exit = held too long with no catalyst

    # MAE signal: if position immediately went against us (low_water_mark < entry),
    # that's evidence of a model or timing issue.
    if mae_cents is not None and entry > 0 and mae_cents < entry * 0.70:
        # Dropped to <70% of entry quickly — adverse from the start
        if outcome == "loss":
            tags.model_error = True

    # ── Fragile-trade tags → root cause hints ────────────────────────────────
    if "low_price_entry" in fragile_flags and outcome == "loss":
        # Low-price contracts have high delta variance — default to normal variance
        # UNLESS the entry had dangerous disagreement, in which case it's a model error.
        if ec_is_low_price and ec_disagree_class == "dangerous" and ec_raw_disagree > 0.20:
            tags.model_error = True
            tags.low_price_dangerous_disagree = True
            tags.should_not_have_traded = True
            # The market was right: disagreement was classified as dangerous but we traded
            if outcome == "loss":
                tags.dangerous_disagree_market_right = True
        else:
            tags.normal_variance = True
    if "same_day_entry" in fragile_flags and outcome == "loss":
        tags.late_day_path_failure = True
    if "model_market_disagreement" in fragile_flags:
        tags.should_not_have_traded = True

    # ── Entry context disagree tags (from rich entry_context, independent of fragile_flags) ──
    if ec_disagree_class == "dangerous" and ec_raw_disagree > 0.20 and outcome == "loss":
        tags.model_error = True
        tags.low_price_dangerous_disagree = ec_is_low_price
        tags.dangerous_disagree_market_right = True
        tags.should_not_have_traded = True

    # ── Poor calibration tag ───────────────────────────────────────────────────
    if ec_calib_level in ("warn", "penalty") and market_type == "temp_low" and outcome == "loss":
        tags.forecast_error_driver = True

    # ── Telemetry gap flag ───────────────────────────────────────────────────
    if not has_snapshot:
        tags.insufficient_telemetry = True

    # ── Outcome decomposition ─────────────────────────────────────────────────
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
        elif tags.normal_variance:
            decomp.unavoidable_variance = 0.8
            decomp.bad_prediction = 0.2
        else:
            # No strong signal — attribute to variance
            decomp.unavoidable_variance = 0.7
            decomp.bad_prediction = 0.3
    elif outcome == "scratch":
        # Scratch: attribute entirely to market conditions / thin edge
        decomp.unavoidable_variance = 1.0
    else:
        # Win
        decomp.bad_prediction = 0.0
        decomp.unavoidable_variance = 0.4

    # ── Severity classification ───────────────────────────────────────────────
    if tags.execution_issue or tags.stale_data_issue:
        sev: Severity = "P0"
        action: RecommendedAction = "auto_apply_safe"
        safe = True
        change_type = "stale_data_guard" if tags.stale_data_issue else "error_handling"
        evidence: EvidenceStrength = "moderate"
    elif tags.model_error and outcome == "loss":
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

    # ── Primary root cause (stable string for day-level rollup) ──────────────
    primary_root_cause = _derive_primary_root_cause(tags, outcome)

    # ── Human-readable lesson ─────────────────────────────────────────────────
    lesson_text = _build_lesson_text(
        outcome=outcome,
        sev=sev,
        ticker=ticker,
        side=side,
        entry=entry,
        exit_p=exit_p,
        pnl=pnl,
        exit_reason_cat=exit_reason_cat,
        tags=tags,
        decomp=decomp,
        action=action,
        mfe_cents=mfe_cents,
        mae_cents=mae_cents,
        fragile_flags=fragile_flags,
        has_snapshot=has_snapshot,
        entry_edge=entry_edge,
        entry_hours_left=entry_hours_left,
        # Rich entry context
        ec_mip_verdict=ec_mip_verdict,
        ec_disagree_class=ec_disagree_class,
        ec_raw_disagree=ec_raw_disagree,
        ec_is_low_price=ec_is_low_price,
        ec_calib_level=ec_calib_level,
        ec_sigma=ec_sigma,
        ec_signal_agree=ec_signal_agree,
        city=city,
        market_type=market_type,
    )

    confidence = 0.1 if outcome == "scratch" else 0.3
    if tags.execution_issue or tags.stale_data_issue:
        confidence = 0.7
    if not has_snapshot:
        confidence = min(confidence, 0.2)   # cap confidence when telemetry is missing

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
        primary_root_cause=primary_root_cause,
        exit_reason=exit_reason_cat,
        mfe_cents=mfe_cents,
        mae_cents=mae_cents,
        fragile_flags=fragile_flags,
        entry_snapshot_available=has_snapshot,
    )


# ── Exit reason normalisation (local to classifier) ───────────────────────────
# Mirrors the one in history.py but kept here to avoid a circular import.
_LESSON_EXIT_PREFIXES = [
    ("thesis_invalidat",     "thesis_invalidation"),
    ("staged_profit",        "staged_profit"),
    ("trailing_stop",        "trailing_stop"),
    ("fair_value",           "fair_value"),
    ("salvage_stop",         "salvage"),
    ("adverse_excursion",    "adverse_excursion_stop"),  # EXIT_ADVERSE_STOP
    ("daily_brake",          "daily_halt"),
    ("daily_halt",           "daily_halt"),
    ("stalled_capital",      "stalled_capital_trap"),   # EXIT_STALLED = "stalled_capital_trap"
    ("expired",              "expired"),
]


def _categorize_exit_reason_lesson(exit_reason: str) -> str:
    if not exit_reason:
        return "unknown"
    er = exit_reason.lower()
    for prefix, category in _LESSON_EXIT_PREFIXES:
        if prefix in er:
            return category
    return "other"


def _derive_primary_root_cause(tags: RootCauseTags, outcome: str) -> str:
    """
    Returns a single stable string identifying the primary root cause.
    Used for day-level rollup in day_diagnosis.root_cause_summary.

    Priority waterfall (first matching tag wins):
      1. execution_issue / stale_data_issue  → "execution_issue"
      2. insufficient_telemetry              → "insufficient_telemetry"
      3. halt_side_effects                   → "halt_side_effects"
      4. model_error                         → "model_error"
      5. forecast_error_driver               → "forecast_error"
      6. sizing_issue                        → "sizing_issue"
      7. low_edge_entry                      → "low_edge"
      8. normal_variance                     → "normal_variance"
      9. scratch outcome                     → "scratch_no_edge"
      10. fallback                           → "normal_variance"
    """
    if tags.execution_issue or tags.stale_data_issue:
        return "execution_issue"
    if tags.halt_side_effects:
        return "halt_side_effects"
    # Specific disagree pattern takes priority over generic insufficient_telemetry
    if tags.dangerous_disagree_market_right:
        return "low_price_dangerous_disagree" if tags.low_price_dangerous_disagree else "dangerous_disagree"
    if tags.insufficient_telemetry:
        return "insufficient_telemetry"
    if tags.model_error:
        return "model_error"
    if tags.forecast_error_driver:
        return "forecast_error"
    if tags.sizing_issue:
        return "sizing_issue"
    if tags.low_edge_entry:
        return "low_edge"
    if tags.normal_variance or outcome == "scratch":
        return "normal_variance"
    return "normal_variance"


def _build_lesson_text(
    *,
    outcome: str,
    sev: str,
    ticker: str,
    side: str,
    entry: int,
    exit_p,
    pnl: float,
    exit_reason_cat: str,
    tags: RootCauseTags,
    decomp: OutcomeDecomposition,
    action: str,
    mfe_cents: Optional[int],
    mae_cents: Optional[int],
    fragile_flags: list,
    has_snapshot: bool,
    entry_edge: Optional[float],
    entry_hours_left: Optional[float],
    # Rich entry context (optional, defaults for backward compat)
    ec_mip_verdict: str = "NONE",
    ec_disagree_class: str = "actionable",
    ec_raw_disagree: float = 0.0,
    ec_is_low_price: bool = False,
    ec_calib_level: str = "ok",
    ec_sigma: float = 7.0,
    ec_signal_agree: float = 0.5,
    city: str = "",
    market_type: str = "",
) -> str:
    """Builds a human-readable lesson string that reflects the three-class outcome."""
    edge_str = f" edge={entry_edge*100:.1f}¢" if entry_edge is not None else ""
    hours_str = f" {entry_hours_left:.1f}h-to-close" if entry_hours_left is not None else ""
    excursion_str = ""
    if mfe_cents is not None or mae_cents is not None:
        mfe_part = f"MFE={mfe_cents}¢" if mfe_cents is not None else ""
        mae_part = f"MAE={mae_cents}¢" if mae_cents is not None else ""
        excursion_str = f" [{' '.join(x for x in [mfe_part, mae_part] if x)}]"
    fragile_str = f" [{','.join(fragile_flags)}]" if fragile_flags else ""
    telemetry_note = "" if has_snapshot else " [no entry snapshot]"

    if outcome == "win":
        return (
            f"WIN [{sev}]: {ticker} {side.upper()} entry={entry}¢ "
            f"exit={exit_p}¢ pnl=${pnl:+.2f}{edge_str}{hours_str}{excursion_str}"
            f"{fragile_str}. Entry quality: {tags.entry_quality}. "
            f"Exit: {exit_reason_cat}. Signal was correct."
        )

    if outcome == "scratch":
        exit_note = ""
        if exit_reason_cat == "daily_halt":
            exit_note = "Halt cleanup — not an edge failure."
        elif exit_reason_cat == "fair_value":
            exit_note = "Fair-value exit — market and model agreed on thin edge."
        elif exit_reason_cat == "unknown":
            exit_note = "Exit reason not recorded."
        else:
            exit_note = f"Exit: {exit_reason_cat}."
        return (
            f"SCRATCH [{sev}]: {ticker} {side.upper()} entry={entry}¢ "
            f"exit={exit_p}¢ pnl=${pnl:+.2f}{edge_str}{hours_str}{excursion_str}"
            f"{fragile_str}{telemetry_note}. {exit_note} "
            f"Action: {action}."
        )

    # ── Loss — attempt to produce the most specific pattern string available ──
    decomp_dict = decomp.to_dict()
    primary_driver = max(decomp_dict, key=lambda k: decomp_dict[k])

    # Pattern 1: low-price + dangerous disagreement — the most actionable pattern
    if tags.low_price_dangerous_disagree or tags.dangerous_disagree_market_right:
        disagree_pct = f"{ec_raw_disagree:.0%}" if ec_raw_disagree else "large"
        calib_note = (
            f" Calibration penalty active ({ec_calib_level})."
            if ec_calib_level in ("warn", "penalty") else ""
        )
        return (
            f"LOSS [{sev}]: {ticker} {side.upper()} entry={entry}¢ "
            f"exit={exit_p}¢ pnl=${pnl:+.2f}{edge_str}{hours_str}{excursion_str}"
            f"{fragile_str}{telemetry_note}. "
            f"PATTERN: low-price ({entry}¢) + dangerous model-vs-market disagreement "
            f"({disagree_pct} gap, mip={ec_mip_verdict}, class={ec_disagree_class}). "
            f"Market was right; model was too optimistic. "
            f"sigma={ec_sigma:.1f}°F signal_agreement={ec_signal_agree:.0%}.{calib_note} "
            f"Low-price large-disagreement entries need stricter gating. "
            f"Exit: {exit_reason_cat}. Action: {action}."
        )

    # Pattern 2: temp_low + poor calibration
    if market_type == "temp_low" and ec_calib_level in ("warn", "penalty"):
        return (
            f"LOSS [{sev}]: {ticker} {side.upper()} entry={entry}¢ "
            f"exit={exit_p}¢ pnl=${pnl:+.2f}{edge_str}{hours_str}{excursion_str}"
            f"{fragile_str}{telemetry_note}. "
            f"PATTERN: temp_low trade with degraded calibration (level={ec_calib_level}). "
            f"Forecast bias may not have been enforced strongly enough. "
            f"sigma={ec_sigma:.1f}°F. "
            f"Primary driver: {primary_driver}. Exit: {exit_reason_cat}. Action: {action}."
        )

    # Pattern 3: model error (general)
    if tags.model_error and not tags.low_price_dangerous_disagree:
        return (
            f"LOSS [{sev}]: {ticker} {side.upper()} entry={entry}¢ "
            f"exit={exit_p}¢ pnl=${pnl:+.2f}{edge_str}{hours_str}{excursion_str}"
            f"{fragile_str}{telemetry_note}. "
            f"PATTERN: model directional error — position moved immediately adverse "
            f"(MAE={mae_cents}¢ vs entry={entry}¢). "
            f"Primary driver: {primary_driver}. Exit: {exit_reason_cat}. Action: {action}."
        )

    # Default: generic loss
    return (
        f"LOSS [{sev}]: {ticker} {side.upper()} entry={entry}¢ "
        f"exit={exit_p}¢ pnl=${pnl:+.2f}{edge_str}{hours_str}{excursion_str}"
        f"{fragile_str}{telemetry_note}. "
        f"Primary driver: {primary_driver}. "
        f"Exit: {exit_reason_cat}. "
        f"Action: {action}."
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
