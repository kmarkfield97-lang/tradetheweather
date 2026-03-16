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
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from src.analysis.classifier import (
    CityPerformancePenalty,
    classify_city_penalty,
    compute_evidence_strength,
    derive_structured_lesson,
    MIN_SAMPLES_MONITOR,
)

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")
LEARNING_LOG_FILE = os.path.join(DATA_DIR, "learning_log.json")
MISSED_OPP_FILE = os.path.join(DATA_DIR, "missed_opportunities.json")

WIN_THRESHOLD_PCT = 0.0


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
        with open(HISTORY_FILE, "w") as f:
            json.dump(records, f, indent=2)

    def _load_learning_log(self) -> list:
        try:
            with open(LEARNING_LOG_FILE) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_learning_log(self, log: list):
        with open(LEARNING_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2)

    def _load_missed_opportunities(self) -> list:
        try:
            with open(MISSED_OPP_FILE) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_missed_opportunities(self, records: list):
        with open(MISSED_OPP_FILE, "w") as f:
            json.dump(records, f, indent=2)

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
        Returns a dict with trade_analyses, structured_lessons, and day_takeaway.
        """
        from dataclasses import asdict

        trade_analyses = []
        structured_lessons = []
        wins = 0
        losses = 0
        total_pnl = 0.0
        p0_count = 0

        for pos in daily_state.positions:
            pd = asdict(pos) if hasattr(pos, "__dataclass_fields__") else dict(pos)
            pnl = pd.get("pnl_dollars", 0.0)
            status = pd.get("status", "open")

            if status in ("closed", "expired"):
                won = pnl > WIN_THRESHOLD_PCT
                wins += 1 if won else 0
                losses += 1 if not won else 0
                total_pnl += pnl

                # Generate structured lesson
                lesson = derive_structured_lesson(pd, won)
                if lesson.severity == "P0":
                    p0_count += 1

                # Legacy flat analysis (backward compat)
                analysis = {
                    "ticker": pd.get("ticker", ""),
                    "side": pd.get("side", ""),
                    "entry_price": pd.get("entry_price"),
                    "exit_price": pd.get("exit_price"),
                    "contracts": pd.get("contracts", 0),
                    "pnl_dollars": round(pnl, 2),
                    "outcome": "win" if won else "loss",
                    "lesson": lesson.lesson_text,
                    "severity": lesson.severity,
                    "suggested_action": lesson.suggested_action,
                }
                trade_analyses.append(analysis)
                structured_lessons.append(lesson.to_dict())

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        starting = daily_state.starting_balance
        ending = daily_state.current_balance
        pnl_pct = ((ending - starting) / starting * 100) if starting > 0 else 0.0

        # Takeaway with P0 alert
        if p0_count > 0:
            takeaway = (
                f"[{p0_count} P0 finding(s) detected — review required.] "
            )
        else:
            takeaway = ""

        if pnl_pct > 3:
            takeaway += f"Strong day: +{pnl_pct:.1f}%. Win rate {win_rate:.0%}. Keep current approach."
        elif pnl_pct > 0:
            takeaway += f"Slight gain: +{pnl_pct:.1f}%. Win rate {win_rate:.0%}. Watch for improving signals."
        elif pnl_pct > -2:
            takeaway += f"Small loss: {pnl_pct:.1f}%. Win rate {win_rate:.0%}. Review threshold selection."
        else:
            takeaway += f"Difficult day: {pnl_pct:.1f}%. Win rate {win_rate:.0%}. Consider tightening edge requirements."

        if daily_state.halt_reason:
            takeaway += f" Halted: {daily_state.halt_reason}."

        # Update history record with analyses
        history = self._load_history()
        for record in history:
            if record.get("date") == daily_state.date:
                record["trade_analyses"] = trade_analyses
                record["structured_lessons"] = structured_lessons
                record["day_takeaway"] = takeaway
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
            "p0_count": p0_count,
            "trade_analyses": trade_analyses,
            "structured_lessons": structured_lessons,
            "day_takeaway": takeaway,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        log = self._load_learning_log()
        log = [l for l in log if l.get("date") != daily_state.date]
        log.append(learning_entry)
        self._save_learning_log(log)

        logger.info(
            f"Day analyzed: {daily_state.date} | "
            f"W/L={wins}/{losses} pnl=${total_pnl:+.2f} P0={p0_count}"
        )

        return {
            "trade_analyses": trade_analyses,
            "structured_lessons": structured_lessons,
            "day_takeaway": takeaway,
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

        # Build performance dicts
        perf_city = {}
        for city, pnls in city_pnl.items():
            total = sum(pnls)
            w = sum(1 for p in pnls if p > 0)
            perf_city[city] = {
                "total_pnl": round(total, 2),
                "win_rate": round(w / len(pnls), 3),
                "trades": len(pnls),
            }

        perf_type = {}
        for mtype, pnls in type_pnl.items():
            total = sum(pnls)
            w = sum(1 for p in pnls if p > 0)
            perf_type[mtype] = {
                "total_pnl": round(total, 2),
                "win_rate": round(w / len(pnls), 3),
                "trades": len(pnls),
            }

        insights.performance_by_city = perf_city
        insights.performance_by_type = perf_type
        insights.open_p0_count = p0_total

        if perf_city:
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
