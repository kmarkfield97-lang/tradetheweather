"""
Operating profile configuration.

Controls the trade-off between maximum defensiveness and profit-seeking.
All three profiles keep hard protections (kill switches, reserve, stale-data
guards, correlation caps, duplicate-order protection) completely intact.
Only the *degree of aggressiveness* on the best setups is adjusted.

Profiles
--------
protection_first
    Original conservative baseline.  No size uplift.  Stall and fair-value
    logic use the tightest tolerances.

balanced (default)
    Modest uplift for top-tier trades only.  Stall / fair-value logic gives
    a small quality bonus to strong, well-supported positions.

profit_tilted
    Larger uplift for top-tier trades.  Slightly more tolerance in holding
    logic for near-locked and high-quality live positions.  Still conservative
    — just the least defensive of the three.

How to change the active profile
---------------------------------
Set the OPERATING_PROFILE environment variable before launching the bot:

    OPERATING_PROFILE=balanced python -m src.main

Or edit ACTIVE_PROFILE directly below.
"""

import os

# ── Valid profile names ─────────────────────────────────────────────────────
PROFILE_PROTECTION_FIRST = "protection_first"
PROFILE_BALANCED         = "balanced"
PROFILE_PROFIT_TILTED    = "profit_tilted"

VALID_PROFILES = {PROFILE_PROTECTION_FIRST, PROFILE_BALANCED, PROFILE_PROFIT_TILTED}

# ── Active profile (env-override or hard-coded default) ────────────────────
_env_profile = os.environ.get("OPERATING_PROFILE", "balanced").strip().lower()
ACTIVE_PROFILE: str = _env_profile if _env_profile in VALID_PROFILES else "balanced"

# ── Per-profile parameter sets ─────────────────────────────────────────────

_PROFILE_PARAMS = {
    PROFILE_PROTECTION_FIRST: {
        # Position sizing uplift for top-tier trades (multiplier on top of baseline)
        # 1.0 = no uplift
        "top_tier_size_multiplier": 1.0,

        # Minimum conviction score thresholds to reach each tier
        # top_tier: must exceed this score (out of ~0.60–0.80 realistic max)
        "top_tier_conviction_min": 0.55,
        # standard_tier: must exceed this score; below → marginal
        "standard_tier_conviction_min": 0.30,

        # Minimum edge (cents) to classify as top-tier
        "top_tier_min_edge_cents": 15.0,
        # Minimum signal agreement to classify as top-tier
        "top_tier_min_signal_agreement": 0.75,
        # Maximum model uncertainty to classify as top-tier
        "top_tier_max_model_uncertainty": 0.35,
        # Minimum exec liquidity ($) to classify as top-tier
        "top_tier_min_exec_liq": 75.0,

        # Stall detection: extra stall cycles required before escalating
        # a high-quality position (0 = no bonus, same as baseline)
        "quality_stall_cycle_bonus": 0,

        # Fair-value exit: extra grace minutes for high-quality positions
        "quality_fv_grace_minutes_bonus": 0,

        # Stall hold-EV ceiling: lower = easier to trigger stall flag
        # (no change from baseline means use the module constant)
        "quality_stall_hold_ev_bonus_cents": 0.0,
    },

    PROFILE_BALANCED: {
        "top_tier_size_multiplier": 1.25,         # 25% more size for top-tier only
        "top_tier_conviction_min": 0.50,
        "standard_tier_conviction_min": 0.28,
        "top_tier_min_edge_cents": 13.0,
        "top_tier_min_signal_agreement": 0.70,
        "top_tier_max_model_uncertainty": 0.40,
        "top_tier_min_exec_liq": 60.0,
        "quality_stall_cycle_bonus": 1,           # 1 extra cycle before urgent on strong positions
        "quality_fv_grace_minutes_bonus": 10,     # 10 extra grace minutes for strong positions
        "quality_stall_hold_ev_bonus_cents": 3.0, # hold-EV threshold 3¢ lower for quality positions
    },

    PROFILE_PROFIT_TILTED: {
        "top_tier_size_multiplier": 1.40,         # 40% more size for top-tier only
        "top_tier_conviction_min": 0.45,
        "standard_tier_conviction_min": 0.25,
        "top_tier_min_edge_cents": 12.0,
        "top_tier_min_signal_agreement": 0.65,
        "top_tier_max_model_uncertainty": 0.45,
        "top_tier_min_exec_liq": 50.0,
        "quality_stall_cycle_bonus": 2,           # 2 extra cycles before urgent on strong positions
        "quality_fv_grace_minutes_bonus": 15,
        "quality_stall_hold_ev_bonus_cents": 5.0,
    },
}

# ── Public accessor ────────────────────────────────────────────────────────

def get_profile_params() -> dict:
    """Return the parameter dict for the currently active profile."""
    return _PROFILE_PARAMS[ACTIVE_PROFILE]


def get_param(key: str):
    """Return a single parameter from the active profile."""
    return _PROFILE_PARAMS[ACTIVE_PROFILE][key]
