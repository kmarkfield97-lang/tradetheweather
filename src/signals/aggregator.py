"""
Signal aggregator. Combines outputs from all signal modules into a single
AggregatedSignal that the analysis engine can apply to base probability estimates.

Supports context-aware cap selection:
  - high_confidence_live:  fresh obs past threshold + high agreement → ±30¢ cap
  - final_hour_crossing:   last hour + threshold already crossed → ±25¢ cap
  - noisy:                 stale data or low agreement → ±12¢ cap
  - default:               standard → ±20¢ cap

Cap values are loaded from data/signal_weights.json so they can be versioned and
reviewed without editing source code. The file is optional — all defaults are
hardcoded here as fallback so the bot never fails due to a missing config.
"""

import json
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Hardcoded fallback defaults (authoritative source of truth) ────────────────
# These values are ALSO written to data/signal_weights.json so they can be
# tracked and proposed-changed through the approval workflow. If the file is
# missing or corrupt the bot runs on these values unchanged.
_DEFAULT_CAPS = {
    "high_confidence_live": 0.30,
    "final_hour_crossing":  0.25,
    "noisy":                0.12,
    "default":              0.20,
}
_DEFAULT_THRESHOLDS = {
    "fresh_obs_age_hours_max": 1.0,
    "stale_obs_age_hours_min": 2.0,
    "final_hour_window_hours": 1.0,
    "high_agreement_min":      0.85,
    "low_agreement_max":       0.50,
}
_DEFAULT_AGGREGATION = {
    "uncertainty_signal_count_cap": 5,
    "min_active_threshold":         0.001,
    "default_confidence":           0.5,
}

_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
_WEIGHTS_FILE = os.path.join(_DATA_DIR, "signal_weights.json")

# Module-level cache: (file_mtime, config_dict).
# Re-read the file only when mtime changes so hot reloading is free.
_weights_cache: tuple = (0.0, None)


def _load_weights() -> dict:
    """
    Load signal_weights.json with mtime-based caching.
    Returns the full config dict, or {} on any error.
    Safe to call on every aggregation — cache hit is a float comparison.
    """
    global _weights_cache
    try:
        mtime = os.path.getmtime(_WEIGHTS_FILE)
        if _weights_cache[0] == mtime and _weights_cache[1] is not None:
            return _weights_cache[1]
        with open(_WEIGHTS_FILE) as f:
            data = json.load(f)
        _weights_cache = (mtime, data)
        return data
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"aggregator: failed to load signal_weights.json: {e}")
        return {}


def _get_caps() -> dict:
    w = _load_weights()
    caps = {**_DEFAULT_CAPS}
    caps.update({k: v for k, v in w.get("cap_regimes", {}).items()
                 if not k.startswith("_")})
    return caps


def _get_thresholds() -> dict:
    w = _load_weights()
    thresh = {**_DEFAULT_THRESHOLDS}
    thresh.update({k: v for k, v in w.get("cap_thresholds", {}).items()
                   if not k.startswith("_")})
    return thresh


def _get_aggregation() -> dict:
    w = _load_weights()
    agg = {**_DEFAULT_AGGREGATION}
    agg.update({k: v for k, v in w.get("signal_defaults", {}).items()
                if not k.startswith("_")})
    agg.update({k: v for k, v in w.get("aggregation", {}).items()
                if not k.startswith("_")})
    return agg


def get_active_weights_version() -> str:
    """
    Returns the version string from signal_weights.json, or 'fallback' if the
    file is absent. Used to stamp every trade-candidate log entry so weight
    changes can be correlated with outcome changes post-hoc.
    """
    w = _load_weights()
    return w.get("_meta", {}).get("version", "fallback")


def _select_cap(context: Optional[dict], agreement: float) -> tuple:
    """
    Returns (cap_value, cap_regime_label) based on data quality and market state.
    Values are sourced from signal_weights.json with hardcoded fallbacks.
    """
    caps   = _get_caps()
    thresh = _get_thresholds()

    if context is None:
        return caps["default"], "default"

    obs_age          = context.get("obs_age")
    hours_left       = context.get("hours_left")
    obs_past_threshold = context.get("obs_past_threshold", False)

    fresh_obs    = obs_age is not None and obs_age < thresh["fresh_obs_age_hours_max"]
    final_hour   = hours_left is not None and hours_left < thresh["final_hour_window_hours"]
    high_agreement = agreement >= thresh["high_agreement_min"]
    stale_data   = obs_age is not None and obs_age > thresh["stale_obs_age_hours_min"]

    # High-confidence live-data regime: fresh obs already past threshold, signals agree
    if fresh_obs and high_agreement and obs_past_threshold:
        return caps["high_confidence_live"], "high_confidence_live"

    # Final-hour crossing: threshold already crossed or nearly so, <1h left
    if final_hour and obs_past_threshold:
        return caps["final_hour_crossing"], "final_hour_crossing"

    # Noisy regime: stale data or low signal agreement
    if stale_data or agreement < thresh["low_agreement_max"]:
        return caps["noisy"], "noisy"

    return caps["default"], "default"


def aggregate(signals: List[dict],
              context: Optional[dict] = None) -> "AggregatedSignal":  # noqa: F821
    """
    Aggregates a list of signal dicts into an AggregatedSignal.

    Each signal dict should have:
        prob_adjustment: float  — signed probability nudge
        confidence: float       — 0-1 weight for this signal
        note: str               — human-readable label

    Optional context dict keys:
        hours_left:           float — hours to market settlement
        obs_age:              float — age of most recent METAR observation (hours)
        obs_past_threshold:   bool  — whether station obs is already past threshold
    """
    from src.signals import AggregatedSignal

    agg_cfg = _get_aggregation()
    default_conf      = float(agg_cfg.get("default_confidence", 0.5))
    min_active_thresh = float(agg_cfg.get("min_active_threshold", 0.001))
    signal_count_cap  = int(agg_cfg.get("uncertainty_signal_count_cap", 5))

    if not signals:
        cap, regime = _select_cap(context, 0.0)
        return AggregatedSignal(
            prob_adjustment=0.0,
            signal_agreement=0.0,
            active_signals=0,
            model_uncertainty=0.5,
            notes=[],
            suggested_cap=cap,
            cap_regime=regime,
            weights_version=get_active_weights_version(),
        )

    active = [s for s in signals if abs(s.get("prob_adjustment", 0.0)) > min_active_thresh]

    if not active:
        cap, regime = _select_cap(context, 0.5)
        return AggregatedSignal(
            prob_adjustment=0.0,
            signal_agreement=0.5,
            active_signals=0,
            model_uncertainty=0.3,
            notes=[],
            suggested_cap=cap,
            cap_regime=regime,
            weights_version=get_active_weights_version(),
        )

    # Weighted average probability adjustment
    total_weight = sum(s.get("confidence", default_conf) for s in active)
    if total_weight == 0:
        total_weight = max(len(active), 1)

    weighted_adj = sum(
        s.get("prob_adjustment", 0.0) * s.get("confidence", default_conf)
        for s in active
    ) / total_weight

    # Agreement: fraction of signals pointing same direction as net adjustment
    if weighted_adj >= 0:
        agreeing = [s for s in active if s.get("prob_adjustment", 0.0) >= 0]
    else:
        agreeing = [s for s in active if s.get("prob_adjustment", 0.0) < 0]
    agreement = len(agreeing) / len(active) if active else 0.5

    # Uncertainty: inversely proportional to agreement and number of signals
    uncertainty = max(0.1, 1.0 - agreement) * (1.0 / min(len(active), signal_count_cap))

    notes = [s.get("note", "") for s in active if s.get("note")]

    # Select contextual cap
    cap, regime = _select_cap(context, agreement)

    return AggregatedSignal(
        prob_adjustment=round(weighted_adj, 4),
        signal_agreement=round(agreement, 3),
        active_signals=len(active),
        model_uncertainty=round(uncertainty, 3),
        notes=notes,
        suggested_cap=cap,
        cap_regime=regime,
        weights_version=get_active_weights_version(),
    )
