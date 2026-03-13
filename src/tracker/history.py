"""
Daily performance history tracker.
Records daily P&L summaries and trade outcomes for learning and analysis.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")
LEARNING_LOG_FILE = os.path.join(DATA_DIR, "learning_log.json")

WIN_THRESHOLD_PCT = 0.0         # > 0 pnl = win
AVOID_CITY_LOSS_THRESHOLD = -5.0  # dollars — avoid cities losing more than this


@dataclass
class HistoryInsights:
    days_recorded: int = 0
    win_rate_7d: float = 0.0
    avg_pnl_7d: float = 0.0
    best_city: str = ""
    worst_city: str = ""
    best_market_type: str = ""
    worst_market_type: str = ""
    avoid_cities: List[str] = field(default_factory=list)
    performance_by_city: Dict[str, dict] = field(default_factory=dict)
    performance_by_type: Dict[str, dict] = field(default_factory=dict)


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
            "trade_analyses": [],   # filled by analyze_day
            "day_takeaway": "",     # filled by analyze_day
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        history = self._load_history()
        # Remove any existing entry for the same date
        history = [h for h in history if h.get("date") != record["date"]]
        history.append(record)
        self._save_history(history)
        return record

    # -------------------------------------------------------------------------
    # Analyze a day
    # -------------------------------------------------------------------------

    def analyze_day(self, daily_state) -> dict:
        """
        Generates trade analyses and a day takeaway.
        Appends analysis to learning_log.json.
        Returns a dict with trade_analyses and day_takeaway.
        """
        from dataclasses import asdict

        trade_analyses = []
        wins = 0
        losses = 0
        total_pnl = 0.0

        for pos in daily_state.positions:
            pd = asdict(pos) if hasattr(pos, "__dataclass_fields__") else dict(pos)
            pnl = pd.get("pnl_dollars", 0.0)
            status = pd.get("status", "open")

            if status in ("closed", "expired"):
                won = pnl > WIN_THRESHOLD_PCT
                wins += 1 if won else 0
                losses += 1 if not won else 0
                total_pnl += pnl

                analysis = {
                    "ticker": pd.get("ticker", ""),
                    "side": pd.get("side", ""),
                    "entry_price": pd.get("entry_price"),
                    "exit_price": pd.get("exit_price"),
                    "contracts": pd.get("contracts", 0),
                    "pnl_dollars": round(pnl, 2),
                    "outcome": "win" if won else "loss",
                    "lesson": _derive_lesson(pd, won),
                }
                trade_analyses.append(analysis)

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        starting = daily_state.starting_balance
        ending = daily_state.current_balance
        pnl_pct = ((ending - starting) / starting * 100) if starting > 0 else 0.0

        if pnl_pct > 3:
            takeaway = f"Strong day: +{pnl_pct:.1f}%. Win rate {win_rate:.0%}. Keep current approach."
        elif pnl_pct > 0:
            takeaway = f"Slight gain: +{pnl_pct:.1f}%. Win rate {win_rate:.0%}. Watch for improving signals."
        elif pnl_pct > -2:
            takeaway = f"Small loss: {pnl_pct:.1f}%. Win rate {win_rate:.0%}. Review threshold selection."
        else:
            takeaway = f"Difficult day: {pnl_pct:.1f}%. Win rate {win_rate:.0%}. Consider tightening edge requirements."

        if daily_state.halt_reason:
            takeaway += f" Halted: {daily_state.halt_reason}."

        # Update history record with analyses
        history = self._load_history()
        for record in history:
            if record.get("date") == daily_state.date:
                record["trade_analyses"] = trade_analyses
                record["day_takeaway"] = takeaway
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
            "trade_analyses": trade_analyses,
            "day_takeaway": takeaway,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        log = self._load_learning_log()
        log = [l for l in log if l.get("date") != daily_state.date]
        log.append(learning_entry)
        self._save_learning_log(log)

        return {"trade_analyses": trade_analyses, "day_takeaway": takeaway}

    # -------------------------------------------------------------------------
    # Insights
    # -------------------------------------------------------------------------

    def get_insights(self) -> HistoryInsights:
        """
        Returns a HistoryInsights dataclass summarizing recent performance.
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

        # Per-city performance
        city_pnl: Dict[str, List[float]] = {}
        type_pnl: Dict[str, List[float]] = {}

        for record in history:
            for pos in record.get("positions", []):
                pnl = pos.get("pnl_dollars", 0.0)
                if pos.get("status") not in ("closed", "expired"):
                    continue
                city = pos.get("city", "UNKNOWN")
                mtype = pos.get("market_type", "UNKNOWN")
                city_pnl.setdefault(city, []).append(pnl)
                type_pnl.setdefault(mtype, []).append(pnl)

        # Build performance dicts
        perf_city = {}
        for city, pnls in city_pnl.items():
            total = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            perf_city[city] = {
                "total_pnl": round(total, 2),
                "win_rate": round(wins / len(pnls), 3),
                "trades": len(pnls),
            }

        perf_type = {}
        for mtype, pnls in type_pnl.items():
            total = sum(pnls)
            wins = sum(1 for p in pnls if p > 0)
            perf_type[mtype] = {
                "total_pnl": round(total, 2),
                "win_rate": round(wins / len(pnls), 3),
                "trades": len(pnls),
            }

        insights.performance_by_city = perf_city
        insights.performance_by_type = perf_type

        if perf_city:
            insights.best_city = max(perf_city, key=lambda c: perf_city[c]["total_pnl"])
            insights.worst_city = min(perf_city, key=lambda c: perf_city[c]["total_pnl"])
            insights.avoid_cities = [
                c for c, v in perf_city.items()
                if v["total_pnl"] < AVOID_CITY_LOSS_THRESHOLD
            ]

        if perf_type:
            insights.best_market_type = max(perf_type, key=lambda t: perf_type[t]["total_pnl"])
            insights.worst_market_type = min(perf_type, key=lambda t: perf_type[t]["total_pnl"])

        return insights

    def format_insights(self) -> str:
        """Returns a Telegram-ready summary of recent performance."""
        insights = self.get_insights()

        if insights.days_recorded == 0:
            return "No trading history yet."

        avoid_str = ", ".join(insights.avoid_cities) if insights.avoid_cities else "none"

        lines = [
            f"Performance Summary ({insights.days_recorded} days recorded)",
            f"7-day win rate: {insights.win_rate_7d:.0%} | Avg P&L: ${insights.avg_pnl_7d:+.2f}",
            f"Best city: {insights.best_city or 'N/A'} | Worst: {insights.worst_city or 'N/A'}",
            f"Best market type: {insights.best_market_type or 'N/A'} | Worst: {insights.worst_market_type or 'N/A'}",
            f"Avoid list: {avoid_str}",
        ]

        if insights.performance_by_city:
            lines.append("\nCity breakdown:")
            for city, perf in sorted(
                insights.performance_by_city.items(),
                key=lambda x: x[1]["total_pnl"],
                reverse=True,
            )[:5]:
                lines.append(
                    f"  {city}: {perf['total_pnl']:+.2f} ({perf['win_rate']:.0%} WR, {perf['trades']}t)"
                )

        return "\n".join(lines)


# -------------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------------

def _derive_lesson(position: dict, won: bool) -> str:
    """Generates a brief lesson string for a closed position."""
    ticker = position.get("ticker", "")
    pnl = position.get("pnl_dollars", 0.0)
    side = position.get("side", "")
    entry = position.get("entry_price", 0)
    exit_p = position.get("exit_price")

    if won:
        return (
            f"WIN: {ticker} {side.upper()} entry={entry}¢ "
            f"exit={exit_p}¢ pnl=${pnl:+.2f}. Signal was correct."
        )
    else:
        return (
            f"LOSS: {ticker} {side.upper()} entry={entry}¢ "
            f"exit={exit_p}¢ pnl=${pnl:+.2f}. "
            f"Review forecast accuracy or edge threshold for this market type."
        )
