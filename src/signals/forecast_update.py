"""
Forecast update signal.
Detects whether the NWS forecast has shifted recently (warming/cooling revision)
by comparing the current forecast against a cached version from ~6 hours ago.
A recent revision toward the threshold is a strong signal.
"""

import json
import os
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
CACHE_FILE = os.path.join(DATA_DIR, "forecast_update_cache.json")


def _load_cache() -> dict:
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def compute(report: dict, threshold: float, market_type: str) -> dict:
    """
    Returns a signal dict with keys: prob_adjustment, confidence, note.
    Positive adjustment = forecast moved toward YES (threshold will be met).
    """
    city = report.get("city", "UNKNOWN")
    forecast = report.get("forecast", {}) or {}
    cache = _load_cache()
    cache_key = f"{city}:{market_type}"
    now_iso = datetime.now(timezone.utc).isoformat()

    current_val = None
    if market_type == "temp_high":
        current_val = forecast.get("high_temp_f")
    elif market_type == "temp_low":
        current_val = forecast.get("low_temp_f")
    elif market_type in ("rain", "snow"):
        current_val = forecast.get("precip_chance")

    if current_val is None:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no forecast value"}

    prev_entry = cache.get(cache_key)
    if prev_entry:
        prev_val = prev_entry.get("value")
        delta = (current_val - prev_val) if (prev_val is not None) else 0.0
    else:
        delta = 0.0

    # Update cache
    cache[cache_key] = {"value": current_val, "updated_at": now_iso}
    _save_cache(cache)

    if abs(delta) < 0.5:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no significant forecast shift"}

    if market_type in ("temp_high", "temp_low"):
        # A 3°F shift = roughly 0.05 probability adjustment
        adj = min(0.10, max(-0.10, delta / 3.0 * 0.05))
    else:
        # Precip chance: 10% shift = 0.05 adjustment
        adj = min(0.08, max(-0.08, delta / 10.0 * 0.05))

    confidence = min(0.75, abs(delta) / 5.0)
    direction = "warmer" if delta > 0 else "cooler"
    note = f"forecast_update: {city} {market_type} shifted {delta:+.1f} ({direction})"

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
