"""
Daily performance history tracker.

Records daily P&L summaries and trade outcomes for learning and analysis.
Produces both human-readable lessons and machine-readable structured lessons.
Implements confidence-weighted city penalties (replacing the blunt avoid-list).
Tracks missed opportunities (strong candidates that were rejected by the engine).
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from src.analysis.classifier import (
    CityPerformancePenalty,
    classify_city_penalty,
    compute_evidence_strength,
    derive_structured_lesson,
    MIN_SAMPLES_MONITOR,
    SCRATCH_PNL_THRESHOLD,
)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")
LEARNING_LOG_FILE = os.path.join(DATA_DIR, "learning_log.json")
MISSED_OPP_FILE = os.path.join(DATA_DIR, "missed_opportunities.json")


def _atomic_write(path: str, data) -> None:
    """Write JSON atomically: write to a temp file then rename, so a crash
    mid-write never leaves a partial/corrupt file at the target path."""
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
        json.dump(data, f, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, path)

# ── Outcome thresholds ────────────────────────────────────────────────────────
# Trades within ±SCRATCH_PNL_THRESHOLD of zero are classified as "scratch",
# not wins or losses. This prevents flat exits from poisoning the learning signal.
# SCRATCH_PNL_THRESHOLD is imported from classifier to keep them in sync.

# Large-loser threshold: a position whose loss exceeds this fraction of starting
# balance is flagged for targeted review in day_diagnosis.
LARGE_LOSER_BALANCE_FRACTION = 0.02   # 2% of starting balance

# A day is "scratch-heavy" when more than this fraction of all trades are scratches.
SCRATCH_HEAVY_THRESHOLD = 0.40


@dataclass
class HistoryInsights:
    days_recorded: int = 0
    win_rate_7d: float = 0.0
    avg_pnl_7d: float = 0.0
    best_city: str = ""
    worst_city: str = ""
    best_market_type: str = ""
    worst_market_type: str = ""
    # Replaced: avoid_cities is now derived from city_penalties
    avoid_cities: List[str] = field(default_factory=list)        # cities with action="avoid"
    raise_edge_cities: Dict[str, float] = field(default_factory=dict)  # city -> penalty cents
    city_penalties: Dict[str, dict] = field(default_factory=dict)      # city -> CityPerformancePenalty.to_dict()
    performance_by_city: Dict[str, dict] = field(default_factory=dict)
    performance_by_type: Dict[str, dict] = field(default_factory=dict)
    # Segmented calibration summary
    calibration_by_segment: Dict[str, dict] = field(default_factory=dict)  # "city:type:season" -> stats
    # P0 count from last session (any open bugs)
    open_p0_count: int = 0


class DailyHistoryTracker:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load / Save helpers
    # -------------------------------------------------------------------------

    def _load_history(self) -> list:
        try:
            with open(HISTORY_FILE) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_history(self, records: list):
        _atomic_write(HISTORY_FILE, records)

    def _load_learning_log(self) -> list:
        try:
            with open(LEARNING_LOG_FILE) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_learning_log(self, log: list):
        _atomic_write(LEARNING_LOG_FILE, log)

    def _load_missed_opportunities(self) -> list:
        try:
            with open(MISSED_OPP_FILE) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_missed_opportunities(self, records: list):
        _atomic_write(MISSED_OPP_FILE, records)

    # -------------------------------------------------------------------------
    # Record a day
    # -------------------------------------------------------------------------

    def record_day(self, daily_state) -> dict:
        """
        Appends a daily P&L summary to trade_history.json.
        daily_state: DailyState instance from PnLTracker.
        Returns the record dict that was saved.
        """
        from dataclasses import asdict

        starting = daily_state.starting_balance
        ending = daily_state.current_balance
        pnl_dollars = ending - starting
        pnl_pct = (pnl_dollars / starting * 100) if starting > 0 else 0.0

        positions_data = []
        for pos in daily_state.positions:
            pd = asdict(pos) if hasattr(pos, "__dataclass_fields__") else dict(pos)
            positions_data.append(pd)

        record = {
            "date": daily_state.date,
            "starting_balance": round(starting, 2),
            "ending_balance": round(ending, 2),
            "pnl_dollars": round(pnl_dollars, 2),
            "pnl_pct": round(pnl_pct, 2),
            "trades_placed": daily_state.trades_placed,
            "halt_reason": daily_state.halt_reason,
            "positions": positions_data,
            "trade_analyses": [],       # filled by analyze_day
            "structured_lessons": [],   # filled by analyze_day
            "day_takeaway": "",         # filled by analyze_day
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        history = self._load_history()
        history = [h for h in history if h.get("date") != record["date"]]
        history.append(record)
        self._save_history(history)
        return record

    # -------------------------------------------------------------------------
    # Analyze a day
    # -------------------------------------------------------------------------

    def analyze_day(self, daily_state) -> dict:
        """
        Generates trade analyses (both human-readable and structured) and a day takeaway.
        Appends analysis to learning_log.json.
        Returns a dict with trade_analyses, structured_lessons, day_takeaway, and day_diagnosis.

        Outcome classification (three-class):
          win     — pnl >  +SCRATCH_PNL_THRESHOLD
          scratch — pnl within ±SCRATCH_PNL_THRESHOLD
          loss    — pnl <  -SCRATCH_PNL_THRESHOLD

        Scratches are excluded from win_rate denominator so a flat exit
        does not dilute the signal about true edge realization.
        """
        from dataclasses import asdict

        trade_analyses = []
        structured_lessons = []
        wins = 0
        losses = 0
        scratches = 0
        total_pnl = 0.0
        p0_count = 0

        starting = daily_state.starting_balance
        ending = daily_state.current_balance
        pnl_pct = ((ending - starting) / starting * 100) if starting > 0 else 0.0
        large_loser_threshold = -max(
            abs(starting * LARGE_LOSER_BALANCE_FRACTION), 0.50
        )

        # ── Fragile trade categories ──────────────────────────────────────────
        # Tracks entries that fit high-risk patterns for separate review.
        fragile_trade_log: list[dict] = []
        # ── Exit reason distribution ──────────────────────────────────────────
        exit_reason_counts: dict[str, int] = {}
        # ── Large-loser log ───────────────────────────────────────────────────
        large_losers: list[dict] = []
        # ── Root cause accumulator ────────────────────────────────────────────
        root_cause_summary: dict[str, int] = {}

        for pos in daily_state.positions:
            pd = asdict(pos) if hasattr(pos, "__dataclass_fields__") else dict(pos)
            pnl = pd.get("pnl_dollars", 0.0)
            status = pd.get("status", "open")

            if status not in ("closed", "expired"):
                continue

            # ── Three-class outcome ───────────────────────────────────────────
            if pnl > SCRATCH_PNL_THRESHOLD:
                outcome_str = "win"
                wins += 1
            elif pnl < -SCRATCH_PNL_THRESHOLD:
                outcome_str = "loss"
                losses += 1
            else:
                outcome_str = "scratch"
                scratches += 1
            total_pnl += pnl

            # ── Exit reason tracking ──────────────────────────────────────────
            exit_reason = pd.get("exit_reason", "") or ""
            exit_reason_category = _categorize_exit_reason(exit_reason)
            exit_reason_counts[exit_reason_category] = (
                exit_reason_counts.get(exit_reason_category, 0) + 1
            )

            # ── Fragile trade detection ───────────────────────────────────────
            fragile_flags = list(pd.get("fragile_flags", []) or [])
            entry_price = pd.get("entry_price", 0) or 0
            entry_hours = pd.get("entry_hours_left")
            if entry_price > 0 and entry_price < 20:
                if "low_price_entry" not in fragile_flags:
                    fragile_flags.append("low_price_entry")
            if entry_hours is not None and entry_hours < 6:
                if "same_day_entry" not in fragile_flags:
                    fragile_flags.append("same_day_entry")
            if entry_hours is not None and entry_hours < 3:
                if "final_hours_entry" not in fragile_flags:
                    fragile_flags.append("final_hours_entry")

            if fragile_flags:
                fragile_trade_log.append({
                    "ticker": pd.get("ticker", ""),
                    "pnl_dollars": round(pnl, 2),
                    "outcome": outcome_str,
                    "fragile_flags": fragile_flags,
                    "exit_reason": exit_reason_category,
                })

            # ── Generate structured lesson ────────────────────────────────────
            lesson = derive_structured_lesson(pd)
            if lesson.severity == "P0":
                p0_count += 1

            # Accumulate root cause tags
            primary_rc = lesson.primary_root_cause
            root_cause_summary[primary_rc] = root_cause_summary.get(primary_rc, 0) + 1

            # ── Large-loser capture ───────────────────────────────────────────
            if pnl < large_loser_threshold:
                large_losers.append({
                    "ticker": pd.get("ticker", ""),
                    "side": pd.get("side", ""),
                    "entry_price": entry_price,
                    "exit_price": pd.get("exit_price"),
                    "contracts": pd.get("contracts", 0),
                    "pnl_dollars": round(pnl, 2),
                    "exit_reason": exit_reason_category,
                    "fragile_flags": fragile_flags,
                    "mfe_cents": pd.get("high_water_mark"),
                    "mae_cents": pd.get("low_water_mark"),
                    "primary_root_cause": primary_rc,
                    "entry_edge": pd.get("entry_edge"),
                    "entry_our_prob": pd.get("entry_our_prob"),
                    "entry_base_prob": pd.get("entry_base_prob"),
                    "entry_sigma": pd.get("entry_sigma"),
                    "entry_signal_breakdown": pd.get("entry_signal_breakdown", []),
                })

            # ── Legacy flat analysis (backward compat) ────────────────────────
            analysis = {
                "ticker": pd.get("ticker", ""),
                "side": pd.get("side", ""),
                "entry_price": entry_price,
                "exit_price": pd.get("exit_price"),
                "contracts": pd.get("contracts", 0),
                "pnl_dollars": round(pnl, 2),
                "outcome": outcome_str,
                "exit_reason": exit_reason_category,
                "fragile_flags": fragile_flags,
                "lesson": lesson.lesson_text,
                "severity": lesson.severity,
                "suggested_action": lesson.suggested_action,
            }
            trade_analyses.append(analysis)
            structured_lessons.append(lesson.to_dict())

        # ── Win rate: denominator excludes scratches ──────────────────────────
        decided_trades = wins + losses
        win_rate = wins / decided_trades if decided_trades > 0 else 0.0
        total_trades = wins + losses + scratches
        scratch_rate = scratches / total_trades if total_trades > 0 else 0.0
        scratch_heavy = scratch_rate > SCRATCH_HEAVY_THRESHOLD

        # ── Structured day diagnosis ──────────────────────────────────────────
        day_diagnosis = _build_day_diagnosis(
            date=daily_state.date,
            starting_balance=starting,
            pnl_dollars=round(ending - starting, 2),
            pnl_pct=round(pnl_pct, 2),
            wins=wins,
            losses=losses,
            scratches=scratches,
            win_rate=round(win_rate, 3),
            scratch_heavy=scratch_heavy,
            large_losers=large_losers,
            fragile_trade_log=fragile_trade_log,
            exit_reason_counts=exit_reason_counts,
            root_cause_summary=root_cause_summary,
            halt_reason=daily_state.halt_reason,
            daily_brake_level=daily_state.daily_brake_level,
            p0_count=p0_count,
        )

        # ── Structured day takeaway ───────────────────────────────────────────
        takeaway = _build_day_takeaway(day_diagnosis, p0_count)

        # Update history record with analyses
        history = self._load_history()
        for record in history:
            if record.get("date") == daily_state.date:
                record["trade_analyses"] = trade_analyses
                record["structured_lessons"] = structured_lessons
                record["day_takeaway"] = takeaway
                record["day_diagnosis"] = day_diagnosis
                record["p0_count"] = p0_count
                break
        self._save_history(history)

        # Append to learning log
        learning_entry = {
            "date": daily_state.date,
            "pnl_dollars": round(ending - starting, 2),
            "pnl_pct": round(pnl_pct, 2),
            "win_rate": round(win_rate, 3),
            "trades": total_trades,
            "wins": wins,
            "losses": losses,
            "scratches": scratches,
            "scratch_heavy": scratch_heavy,
            "p0_count": p0_count,
            "trade_analyses": trade_analyses,
            "structured_lessons": structured_lessons,
            "day_takeaway": takeaway,
            "day_diagnosis": day_diagnosis,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        log = self._load_learning_log()
        log = [l for l in log if l.get("date") != daily_state.date]
        log.append(learning_entry)
        self._save_learning_log(log)

        logger.info(
            f"Day analyzed: {daily_state.date} | "
            f"W/L/S={wins}/{losses}/{scratches} pnl=${total_pnl:+.2f} "
            f"scratch_heavy={scratch_heavy} P0={p0_count}"
        )

        return {
            "trade_analyses": trade_analyses,
            "structured_lessons": structured_lessons,
            "day_takeaway": takeaway,
            "day_diagnosis": day_diagnosis,
            "p0_count": p0_count,
        }

    # -------------------------------------------------------------------------
    # Missed opportunity recording
    # -------------------------------------------------------------------------

    def record_missed_opportunity(
        self,
        ticker: str,
        city: str,
        market_type: str,
        rejection_reason: str,
        our_prob: float,
        market_price_cents: int,
        edge_cents: float,
        actual_outcome: Optional[str] = None,   # "yes_won" / "no_won" / None
        scan_timestamp: Optional[str] = None,
    ):
        """
        Records a market that was evaluated but rejected, for post-hoc analysis.
        actual_outcome is filled in by the settlement backfill job if known.
        """
        record = {
            "ticker": ticker,
            "city": city,
            "market_type": market_type,
            "rejection_reason": rejection_reason,
            "our_prob": round(our_prob, 3),
            "market_price_cents": market_price_cents,
            "edge_cents": round(edge_cents, 1),
            "actual_outcome": actual_outcome,
            "scan_timestamp": scan_timestamp or datetime.now(timezone.utc).isoformat(),
            "would_have_won": None,   # filled in by settlement backfill
        }
        records = self._load_missed_opportunities()
        records.append(record)
        # Keep last 500 missed opportunity records
        records = records[-500:]
        self._save_missed_opportunities(records)

    def backfill_missed_opportunity_outcomes(self, settled_markets: dict):
        """
        After settlement, update missed opportunity records with actual outcomes.
        settled_markets: dict of ticker -> "yes" | "no" (winning side).
        """
        records = self._load_missed_opportunities()
        updated = 0
        for rec in records:
            if rec.get("would_have_won") is not None:
                continue
            ticker = rec.get("ticker", "")
            if ticker not in settled_markets:
                continue
            winning_side = settled_markets[ticker]
            # Determine what side we would have taken (from our_prob > 0.5 → yes)
            our_side = "yes" if rec.get("our_prob", 0) > 0.5 else "no"
            rec["actual_outcome"] = winning_side
            rec["would_have_won"] = (our_side == winning_side)
            updated += 1
        if updated:
            self._save_missed_opportunities(records)
            logger.info(f"Backfilled {updated} missed opportunity outcomes")

    def get_missed_opportunity_summary(self, lookback_days: int = 14) -> dict:
        """
        Returns a summary of missed opportunities over the last N days.
        Identifies filter-level patterns (too strict, timing, etc.).
        """
        records = self._load_missed_opportunities()
        cutoff = datetime.now(timezone.utc).timestamp() - lookback_days * 86400

        recent = []
        for rec in records:
            ts = rec.get("scan_timestamp", "")
            try:
                t = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                if t >= cutoff:
                    recent.append(rec)
            except Exception:
                pass

        total = len(recent)
        if total == 0:
            return {"total": 0, "would_have_won": 0, "would_have_loss": 0, "by_rejection_reason": {}}

        resolved = [r for r in recent if r.get("would_have_won") is not None]
        wins = sum(1 for r in resolved if r.get("would_have_won"))
        losses = len(resolved) - wins

        by_reason: dict[str, dict] = {}
        for rec in recent:
            reason = rec.get("rejection_reason", "unknown")
            by_reason.setdefault(reason, {"count": 0, "would_have_won": 0, "resolved": 0})
            by_reason[reason]["count"] += 1
            if rec.get("would_have_won") is not None:
                by_reason[reason]["resolved"] += 1
                if rec.get("would_have_won"):
                    by_reason[reason]["would_have_won"] += 1

        # Flag any rejection reason with >50% would-have-won rate and >=5 resolved samples
        filter_candidates = []
        for reason, stats in by_reason.items():
            n = stats["resolved"]
            if n >= 5:
                ww_rate = stats["would_have_won"] / n
                if ww_rate > 0.55:
                    filter_candidates.append({
                        "rejection_reason": reason,
                        "would_have_won_rate": round(ww_rate, 3),
                        "sample": n,
                    })

        return {
            "total": total,
            "resolved": len(resolved),
            "would_have_won": wins,
            "would_have_loss": losses,
            "by_rejection_reason": by_reason,
            "filter_candidates_for_review": filter_candidates,
        }

    # -------------------------------------------------------------------------
    # Insights
    # -------------------------------------------------------------------------

    def get_insights(self) -> HistoryInsights:
        """
        Returns a HistoryInsights dataclass summarizing recent performance.
        Uses confidence-weighted city penalties instead of a blunt avoid-list.
        """
        history = self._load_history()
        insights = HistoryInsights(days_recorded=len(history))

        if not history:
            return insights

        # Last 7 days
        recent = history[-7:]
        wins_7d = sum(1 for r in recent if r.get("pnl_dollars", 0) > 0)
        insights.win_rate_7d = round(wins_7d / len(recent), 3) if recent else 0.0
        insights.avg_pnl_7d = round(
            sum(r.get("pnl_dollars", 0) for r in recent) / len(recent), 2
        ) if recent else 0.0

        # Per-city and per-type performance (all history)
        city_pnl: Dict[str, List[float]] = {}
        city_exit_reasons: Dict[str, List[str]] = {}
        type_pnl: Dict[str, List[float]] = {}

        # Segmented calibration: city:market_type:season -> [errors]
        # (harvest from structured_lessons where forecast_error_driver=True)
        seg_errors: Dict[str, List[float]] = {}

        p0_total = 0

        for record in history:
            p0_total += record.get("p0_count", 0)
            for pos in record.get("positions", []):
                pnl = pos.get("pnl_dollars", 0.0)
                if pos.get("status") not in ("closed", "expired"):
                    continue
                city = pos.get("city", "UNKNOWN")
                mtype = pos.get("market_type", "UNKNOWN")
                exit_reason = pos.get("exit_reason", "") or ""

                city_pnl.setdefault(city, []).append(pnl)
                city_exit_reasons.setdefault(city, []).append(exit_reason)
                type_pnl.setdefault(mtype, []).append(pnl)

            # Segmented calibration from structured lessons
            for lesson in record.get("structured_lessons", []):
                rc = lesson.get("root_cause", {})
                if rc.get("forecast_error_driver"):
                    ticker = lesson.get("ticker", "")
                    # We don't have direct error magnitude here, but record the loss
                    pnl_l = lesson.get("pnl_dollars", 0.0)
                    season = _get_season_for_date(record.get("date", ""))
                    city_l = ""
                    mtype_l = ""
                    for pos in record.get("positions", []):
                        if pos.get("ticker") == ticker:
                            city_l = pos.get("city", "")
                            mtype_l = pos.get("market_type", "")
                            break
                    if city_l and mtype_l and season:
                        seg_key = f"{city_l}:{mtype_l}:{season}"
                        seg_errors.setdefault(seg_key, []).append(pnl_l)

        # Build performance dicts (scratch-aware win rate)
        perf_city = {}
        for city, pnls in city_pnl.items():
            total = sum(pnls)
            w = sum(1 for p in pnls if p > SCRATCH_PNL_THRESHOLD)
            s = sum(1 for p in pnls if abs(p) <= SCRATCH_PNL_THRESHOLD)
            decided = len(pnls) - s
            perf_city[city] = {
                "total_pnl": round(total, 2),
                "win_rate": round(w / decided, 3) if decided > 0 else 0.0,
                "trades": len(pnls),
                "scratches": s,
            }

        perf_type = {}
        for mtype, pnls in type_pnl.items():
            total = sum(pnls)
            w = sum(1 for p in pnls if p > SCRATCH_PNL_THRESHOLD)
            s = sum(1 for p in pnls if abs(p) <= SCRATCH_PNL_THRESHOLD)
            decided = len(pnls) - s
            perf_type[mtype] = {
                "total_pnl": round(total, 2),
                "win_rate": round(w / decided, 3) if decided > 0 else 0.0,
                "trades": len(pnls),
                "scratches": s,
            }

        insights.performance_by_city = perf_city
        insights.performance_by_type = perf_type
        insights.open_p0_count = p0_total

        known_cities = {c for c in perf_city if c and c != "UNKNOWN"}
        if known_cities:
            insights.best_city = max(known_cities, key=lambda c: perf_city[c]["total_pnl"])
            insights.worst_city = min(known_cities, key=lambda c: perf_city[c]["total_pnl"])
        elif perf_city:
            insights.best_city = max(perf_city, key=lambda c: perf_city[c]["total_pnl"])
            insights.worst_city = min(perf_city, key=lambda c: perf_city[c]["total_pnl"])

        if perf_type:
            insights.best_market_type = max(perf_type, key=lambda t: perf_type[t]["total_pnl"])
            insights.worst_market_type = min(perf_type, key=lambda t: perf_type[t]["total_pnl"])

        # ── Confidence-weighted city penalties (replaces blunt avoid-list) ───
        city_penalty_dicts = {}
        avoid_cities = []
        raise_edge_cities = {}

        for city, pnls in city_pnl.items():
            if len(pnls) < MIN_SAMPLES_MONITOR:
                continue
            exit_reasons = city_exit_reasons.get(city, [])
            penalty = classify_city_penalty(city, pnls, exit_reasons)
            city_penalty_dicts[city] = penalty.to_dict()

            if penalty.action == "avoid":
                avoid_cities.append(city)
            elif penalty.action in ("raise_edge_soft", "raise_edge_strong"):
                raise_edge_cities[city] = penalty.edge_penalty_cents

        insights.city_penalties = city_penalty_dicts
        insights.avoid_cities = avoid_cities
        insights.raise_edge_cities = raise_edge_cities

        # ── Segmented calibration summary ─────────────────────────────────────
        cal_summary = {}
        for seg_key, pnl_list in seg_errors.items():
            n = len(pnl_list)
            avg = sum(pnl_list) / n if n else 0
            cal_summary[seg_key] = {
                "samples": n,
                "avg_pnl_when_forecast_error": round(avg, 2),
                "evidence": compute_evidence_strength(
                    n, sum(1 for p in pnl_list if p < 0) / n if n else 0
                ),
            }
        insights.calibration_by_segment = cal_summary

        return insights

    def format_insights(self) -> str:
        """Returns a Telegram-ready summary of recent performance."""
        insights = self.get_insights()

        if insights.days_recorded == 0:
            return "No trading history yet."

        avoid_str = ", ".join(insights.avoid_cities) if insights.avoid_cities else "none"
        raise_str = (
            ", ".join(f"{c}(+{v:.0f}¢)" for c, v in insights.raise_edge_cities.items())
            if insights.raise_edge_cities else "none"
        )

        lines = [
            f"Performance Summary ({insights.days_recorded} days recorded)",
            f"7-day win rate: {insights.win_rate_7d:.0%} | Avg P&L: ${insights.avg_pnl_7d:+.2f}",
            f"Best city: {insights.best_city or 'N/A'} | Worst: {insights.worst_city or 'N/A'}",
            f"Best market type: {insights.best_market_type or 'N/A'} | Worst: {insights.worst_market_type or 'N/A'}",
            f"Avoid list: {avoid_str}",
            f"Edge raised: {raise_str}",
        ]

        if insights.open_p0_count > 0:
            lines.append(f"[{insights.open_p0_count} P0 findings in history — review logs]")

        if insights.performance_by_city:
            lines.append("\nCity breakdown:")
            for city, perf in sorted(
                insights.performance_by_city.items(),
                key=lambda x: x[1]["total_pnl"],
                reverse=True,
            )[:5]:
                penalty_info = insights.city_penalties.get(city, {})
                action = penalty_info.get("action", "")
                action_tag = f" [{action}]" if action and action != "monitor" else ""
                lines.append(
                    f"  {city}: {perf['total_pnl']:+.2f} ({perf['win_rate']:.0%} WR, "
                    f"{perf['trades']}t){action_tag}"
                )

        miss = self.get_missed_opportunity_summary(lookback_days=7)
        if miss.get("filter_candidates_for_review"):
            lines.append("\nFilter review candidates (possible over-restriction):")
            for cand in miss["filter_candidates_for_review"][:3]:
                lines.append(
                    f"  {cand['rejection_reason']}: "
                    f"{cand['would_have_won_rate']:.0%} would-have-won "
                    f"({cand['sample']} samples)"
                )

        return "\n".join(lines)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

# ── Exit reason category map ──────────────────────────────────────────────────
# Normalises the variable-format exit reason strings from pnl.py into stable
# categories used by the learning / diagnosis system.
_EXIT_CATEGORY_PREFIXES = [
    ("thesis_invalidat", "thesis_invalidation"),
    ("staged_profit",    "staged_profit"),
    ("trailing_stop",    "trailing_stop"),
    ("fair_value",       "fair_value"),
    ("salvage_stop",     "salvage"),
    ("daily_brake",      "daily_halt"),
    ("daily_halt",       "daily_halt"),
    ("expired",          "expired"),
]


def _categorize_exit_reason(exit_reason: str) -> str:
    """
    Converts a raw exit_reason string (which may contain embedded numbers like
    'staged_profit_42¢' or 'fair_value(exit_ev=38.1>hold_ev=32.0)') into a
    stable category key for accumulation and diagnosis.
    """
    if not exit_reason:
        return "unknown"
    er = exit_reason.lower()
    for prefix, category in _EXIT_CATEGORY_PREFIXES:
        if prefix in er:
            return category
    return "other"


def _build_day_diagnosis(
    *,
    date: str,
    starting_balance: float,
    pnl_dollars: float,
    pnl_pct: float,
    wins: int,
    losses: int,
    scratches: int,
    win_rate: float,
    scratch_heavy: bool,
    large_losers: list,
    fragile_trade_log: list,
    exit_reason_counts: dict,
    root_cause_summary: dict,
    halt_reason: str,
    daily_brake_level: int,
    p0_count: int,
) -> dict:
    """
    Builds a structured day diagnosis dict for storage in trade_history.json and
    learning_log.json. Answers the key diagnostic questions for a bad day:

      - What drove losses: one trade or many?
      - Were flat trades a warning sign (edge failure) or forced (halt cleanup)?
      - What exit reasons dominated?
      - Is the account balance too small for conclusions to be meaningful?
      - What root cause categories dominated today?
    """
    total_trades = wins + losses + scratches
    decided = wins + losses

    # ── Balance-size safeguard note ───────────────────────────────────────────
    safeguard_notes = []
    if starting_balance < 10.0:
        safeguard_notes.append(
            f"Account balance is very small (${starting_balance:.2f}). "
            "P&L percentages are highly volatile and should not drive strategy changes. "
            "Need larger sample before drawing conclusions."
        )
    if total_trades < 5:
        safeguard_notes.append(
            f"Only {total_trades} trade(s) today — insufficient data for statistical conclusions."
        )

    # ── Scratch attribution ───────────────────────────────────────────────────
    # Distinguish forced-halt scratches from genuine edge-failure scratches.
    halt_scratch_count = exit_reason_counts.get("daily_halt", 0)
    fair_value_count   = exit_reason_counts.get("fair_value", 0)
    scratch_attribution = "unknown"
    if scratches > 0:
        if halt_scratch_count >= scratches * 0.5:
            scratch_attribution = "halt_forced"
        elif fair_value_count >= scratches * 0.5:
            scratch_attribution = "fair_value_exit"
        elif scratches >= total_trades * 0.5 and decided == 0:
            scratch_attribution = "possible_entry_edge_failure"
        else:
            scratch_attribution = "mixed"

    # ── Loss concentration ────────────────────────────────────────────────────
    loss_concentration = "none"
    if losses == 1 and large_losers:
        loss_concentration = "single_large_loser"
    elif losses > 1 and large_losers:
        loss_concentration = "multiple_losers_with_large"
    elif losses > 1:
        loss_concentration = "distributed"

    # ── Primary root cause for the day ───────────────────────────────────────
    # Pick the most common root cause tag from trade-level analyses.
    # Default to "insufficient_telemetry" when no entry snapshot was stored.
    if root_cause_summary:
        primary_root_cause = max(root_cause_summary, key=lambda k: root_cause_summary[k])
    else:
        primary_root_cause = "insufficient_telemetry"

    # ── Recommended response ──────────────────────────────────────────────────
    recommended_response = _recommend_day_response(
        wins=wins,
        losses=losses,
        scratches=scratches,
        pnl_dollars=pnl_dollars,
        starting_balance=starting_balance,
        large_losers=large_losers,
        scratch_attribution=scratch_attribution,
        halt_reason=halt_reason,
        primary_root_cause=primary_root_cause,
        safeguard_notes=safeguard_notes,
    )

    return {
        "date": date,
        "outcome_distribution": {
            "wins": wins,
            "losses": losses,
            "scratches": scratches,
            "total": total_trades,
            "decided": decided,
        },
        "win_rate_decided": round(win_rate, 3),
        "scratch_rate": round(scratches / total_trades, 3) if total_trades > 0 else 0.0,
        "scratch_heavy": scratch_heavy,
        "scratch_attribution": scratch_attribution,
        "loss_concentration": loss_concentration,
        "large_losers": large_losers,
        "fragile_trades": fragile_trade_log,
        "fragile_trade_count": len(fragile_trade_log),
        "exit_reason_counts": exit_reason_counts,
        "root_cause_summary": root_cause_summary,
        "primary_root_cause": primary_root_cause,
        "halt_triggered": bool(halt_reason),
        "halt_reason": halt_reason,
        "daily_brake_level": daily_brake_level,
        "p0_count": p0_count,
        "safeguard_notes": safeguard_notes,
        "recommended_response": recommended_response,
    }


def _recommend_day_response(
    *,
    wins: int,
    losses: int,
    scratches: int,
    pnl_dollars: float,
    starting_balance: float,
    large_losers: list,
    scratch_attribution: str,
    halt_reason: str,
    primary_root_cause: str,
    safeguard_notes: list,
) -> str:
    """
    Returns a structured recommended response string for the day.
    Conservative: avoids strong recommendations on thin data or tiny accounts.
    """
    decided = wins + losses
    total = wins + losses + scratches

    if safeguard_notes:
        return (
            "OBSERVE ONLY — insufficient evidence for strategy changes. "
            f"{safeguard_notes[0]}"
        )

    if decided == 0 and scratches > 0:
        if scratch_attribution == "halt_forced":
            return (
                "Scratch-only day caused by halt cleanup. "
                "Risk controls functioned correctly. "
                "Review entry quality and halt threshold if this recurs."
            )
        return (
            "All trades exited flat — no decided outcomes. "
            "Investigate whether edge was too thin at entry or spread prevented monetization. "
            "Do not draw strategy conclusions from scratch-only data."
        )

    if losses == 1 and large_losers:
        loser = large_losers[0]
        rc = loser.get("primary_root_cause", "unknown")
        return (
            f"Single large loser drove the day's P&L: {loser.get('ticker', '?')} "
            f"(pnl=${loser.get('pnl_dollars', 0):+.2f}, root_cause={rc}). "
            "Review that specific trade's entry snapshot and signal breakdown "
            "before changing any strategy parameters. "
            "One-trade evidence is insufficient for systemic conclusions."
        )

    if pnl_dollars < 0 and primary_root_cause == "insufficient_telemetry":
        return (
            "Losses recorded but entry snapshots are not available for root-cause analysis. "
            "Activate full entry-snapshot logging to diagnose future days properly."
        )

    if pnl_dollars < 0 and primary_root_cause in ("model_error", "forecast_error"):
        return (
            "Losses appear model/forecast-driven. "
            "Review NWS forecast accuracy for affected cities before next session. "
            "Do not change edge thresholds without at least 15+ trade sample."
        )

    if pnl_dollars < 0 and primary_root_cause == "halt_side_effects":
        return (
            "Halt cleanup caused or amplified losses. "
            "Risk controls functioned as intended. "
            "Review whether halt threshold is appropriate for account size."
        )

    win_rate = wins / decided if decided > 0 else 0.0
    if win_rate < 0.35 and decided >= 5:
        return (
            f"Low win rate ({win_rate:.0%}) on {decided} decided trades. "
            "Review signal quality and edge thresholds. "
            "Collect at least 15 decided trades before making threshold changes."
        )

    return (
        "Review structured_lessons for per-trade root cause tags. "
        "No single clear driver identified. Collect more data."
    )


def _build_day_takeaway(diagnosis: dict, p0_count: int) -> str:
    """
    Builds a human-readable day takeaway string from the structured diagnosis.
    Replaces the generic "consider tightening edge requirements" with specific,
    evidence-based language that names the actual driver of today's outcome.
    """
    wins    = diagnosis["outcome_distribution"]["wins"]
    losses  = diagnosis["outcome_distribution"]["losses"]
    scratches = diagnosis["outcome_distribution"]["scratches"]
    total   = diagnosis["outcome_distribution"]["total"]
    decided = diagnosis["outcome_distribution"]["decided"]
    win_rate = diagnosis["win_rate_decided"]
    pnl     = diagnosis.get("pnl_dollars", 0.0)  # not stored but computed below
    halt    = diagnosis.get("halt_reason", "")
    scratch_heavy = diagnosis.get("scratch_heavy", False)
    large_losers = diagnosis.get("large_losers", [])
    primary_rc = diagnosis.get("primary_root_cause", "")
    safeguard_notes = diagnosis.get("safeguard_notes", [])
    recommended = diagnosis.get("recommended_response", "")

    parts = []

    if p0_count > 0:
        parts.append(f"[{p0_count} P0 FINDING(S) — REVIEW REQUIRED]")

    # Outcome summary line
    if decided == 0 and scratches > 0:
        parts.append(
            f"All {scratches} trade(s) exited flat (scratch). "
            f"No decided outcomes — win rate undefined."
        )
    else:
        wr_str = f"{win_rate:.0%}" if decided > 0 else "N/A"
        parts.append(
            f"W/L/S: {wins}/{losses}/{scratches} ({total} total). "
            f"Win rate (decided): {wr_str}."
        )

    # Scratch-heavy note
    if scratch_heavy:
        attr = diagnosis.get("scratch_attribution", "unknown")
        if attr == "halt_forced":
            parts.append(
                f"{scratches} scratch(es) — mostly halt-forced cleanup, not edge failure."
            )
        elif attr == "fair_value_exit":
            parts.append(
                f"{scratches} scratch(es) via fair-value exit — "
                "model and market agreed; thin edge prevented gain."
            )
        elif attr == "possible_entry_edge_failure":
            parts.append(
                f"{scratches} scratch(es) — possible entry edge failure: "
                "entries may have lacked true edge or spread ate the margin."
            )
        else:
            parts.append(f"{scratches} scratch(es) — mixed attribution (see day_diagnosis).")

    # Large losers note
    if large_losers:
        loser_names = ", ".join(
            f"{l.get('ticker','?')}(${l.get('pnl_dollars',0):+.2f})"
            for l in large_losers[:3]
        )
        parts.append(f"Large loser(s): {loser_names}.")

    # Primary root cause
    if primary_rc and primary_rc not in ("normal_variance", "insufficient_telemetry"):
        parts.append(f"Primary root cause: {primary_rc}.")
    elif primary_rc == "insufficient_telemetry":
        parts.append("Entry snapshots not available — root cause unknown.")

    # Safeguards
    if safeguard_notes:
        parts.append(safeguard_notes[0])

    # Halt note
    if halt:
        parts.append(f"Halted: {halt}.")

    # Recommended action
    if recommended:
        parts.append(f"Response: {recommended}")

    return " ".join(parts)


def _get_season_for_date(date_str: str) -> str:
    """Returns 'winter' / 'spring' / 'summer' / 'fall' for an ISO date string."""
    try:
        d = date.fromisoformat(date_str)
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
