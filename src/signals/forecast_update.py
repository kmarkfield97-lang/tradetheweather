"""
Forecast update signal.
Detects whether the NWS forecast has shifted recently (warming/cooling revision)
by comparing the current forecast against a cached baseline that is updated at most
every CACHE_MIN_AGE_HOURS hours.  The delta therefore reflects a genuine multi-hour
revision, not the noise between consecutive 10-minute scans.

A recent revision toward the threshold is a strong signal.
"""

import json
import os
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
CACHE_FILE = os.path.join(DATA_DIR, "forecast_update_cache.json")

# Only replace a cache baseline once it is at least this old (hours).
# Prevents computing delta against 10-minute-old data.
CACHE_MIN_AGE_HOURS = 6.0

# Cache entries older than this are considered stale — delta is unreliable.
CACHE_STALE_HOURS = 10.0


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
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

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

    # Determine cache entry age and whether we should update the baseline
    should_update_cache = True
    prev_is_stale = False
    cache_age_h = None
    delta = 0.0

    if prev_entry:
        prev_val = prev_entry.get("value")
        prev_updated = prev_entry.get("updated_at")
        if prev_updated:
            try:
                prev_dt = datetime.fromisoformat(prev_updated.replace("Z", "+00:00"))
                cache_age_h = (now - prev_dt).total_seconds() / 3600
                # Only update cache baseline once it is sufficiently aged
                should_update_cache = cache_age_h >= CACHE_MIN_AGE_HOURS
                prev_is_stale = cache_age_h > CACHE_STALE_HOURS
            except Exception:
                should_update_cache = True
        if prev_val is not None and not prev_is_stale:
            delta = current_val - prev_val
    # If no previous entry, nothing to compare — write initial baseline
    # delta stays 0.0 for this first call

    # Write updated baseline only if the old one is aged enough
    if should_update_cache:
        cache[cache_key] = {"value": current_val, "updated_at": now_iso}
        _save_cache(cache)

    # Stale cache means the delta is too old to be actionable
    if prev_is_stale:
        return {
            "prob_adjustment": 0.0,
            "confidence": 0.0,
            "note": f"forecast_update: {city} cache is {cache_age_h:.0f}h old (stale baseline)",
        }

    if abs(delta) < 0.5:
        return {"prob_adjustment": 0.0, "confidence": 0.0, "note": "no significant forecast shift"}

    if market_type in ("temp_high", "temp_low"):
        # Large shifts toward threshold are a stronger signal — raise cap to 0.15 for >3°F
        abs_delta = abs(delta)
        cap = 0.15 if abs_delta > 3.0 else 0.10
        # 3°F shift ≈ 0.05 probability adjustment; scales linearly
        adj = min(cap, max(-cap, delta / 3.0 * 0.05))
    else:
        # Precip chance: 10% shift ≈ 0.05 adjustment
        adj = min(0.08, max(-0.08, delta / 10.0 * 0.05))

    # Confidence scales with magnitude; higher if cache is reasonably fresh
    confidence = min(0.75, abs(delta) / 5.0)
    if cache_age_h is not None and cache_age_h > CACHE_MIN_AGE_HOURS * 1.5:
        # Older baseline is still valid but slightly less reliable
        confidence = min(confidence, 0.55)

    direction = "warmer" if delta > 0 else "cooler"
    age_str = f"{cache_age_h:.1f}h" if cache_age_h is not None else "?"
    note = (
        f"forecast_update: {city} {market_type} shifted {delta:+.1f} ({direction}) "
        f"over {age_str}"
    )

    return {
        "prob_adjustment": round(adj, 4),
        "confidence": round(confidence, 3),
        "note": note,
    }
