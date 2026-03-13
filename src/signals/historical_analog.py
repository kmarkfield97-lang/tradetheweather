"""
Historical analog signal.
Looks for past days with similar forecast conditions and checks what the actual
outcome was. Uses data/trade_history.json to find analogs.
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")


def compute(report: dict, threshold: float, market_type: str, side: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.
    Positive = historical analogs support the YES outcome.
    """
    try:
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    except Exception:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no history data"}

    city = report.get("city", "UNKNOWN")
    records = history if isinstance(history, list) else history.get("records", [])

    # Find closed positions for this city + market_type
    analogs = []
    for day in records:
        for pos in day.get("positions", []):
            if (
                pos.get("city", "").upper() == city.upper()
                and pos.get("market_type") == market_type
                and pos.get("status") in ("closed", "expired")
                and pos.get("pnl_dollars") is not None
            ):
                analogs.append(pos)

    if len(analogs) < 3:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": f"insufficient analogs for {city}/{market_type}"}

    # Simple win rate from recent analogs
    recent = analogs[-10:]
    wins = sum(1 for p in recent if p.get("pnl_dollars", 0) > 0)
    win_rate = wins / len(recent)
    # Neutral is 0.5 — adjust based on deviation
    adj = (win_rate - 0.5) * 0.08
    adj = max(-0.06, min(0.06, adj))
    confidence = min(0.6, len(recent) / 10.0 * 0.5)

    note = f"historical_analog: {city} {market_type} win_rate={win_rate:.0%} over {len(recent)} trades"

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
