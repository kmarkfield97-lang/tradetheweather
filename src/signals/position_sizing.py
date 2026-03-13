"""
Position sizing signal / calculator.
Returns a recommended position size in dollars using fractional Kelly criterion,
adjusted for signal agreement, model uncertainty, and liquidity.
"""


def compute(
    our_prob: float,
    side: str,
    price: int,              # market price in cents
    edge: float,             # our_prob minus market implied prob
    daily_budget: float,
    open_position_cost: float,
    signal_agreement: float,  # 0-1
    model_uncertainty: float, # 0-1, higher = less certain
    liquidity: dict,
) -> dict:
    """
    Returns a dict with keys:
        position_dollars, kelly_raw, fractional_kelly,
        confidence_multiplier, liquidity_multiplier, exposure_multiplier
    """
    # Basic Kelly criterion
    prob_win = our_prob if side == "yes" else (1.0 - our_prob)
    prob_lose = 1.0 - prob_win
    odds = (100 - price) / price   # net payout per dollar if we win

    if odds <= 0 or prob_win <= 0:
        return {
            "position_dollars": 0.0,
            "kelly_raw": 0.0,
            "fractional_kelly": 0.0,
            "confidence_multiplier": 0.0,
            "liquidity_multiplier": 0.0,
            "exposure_multiplier": 0.0,
        }

    kelly_raw = (prob_win * odds - prob_lose) / odds
    kelly_raw = max(0.0, kelly_raw)

    # Fractional Kelly (1/4 base)
    fractional_kelly = kelly_raw * 0.25

    # Signal agreement multiplier: 1.0 at full agreement, 0.5 at full disagreement
    confidence_multiplier = 0.5 + (signal_agreement * 0.5)

    # Liquidity multiplier: reduce size if thin orderbook
    total_volume = liquidity.get("total_volume", 0)
    if total_volume >= 500:
        liquidity_multiplier = 1.0
    elif total_volume >= 100:
        liquidity_multiplier = 0.75
    elif total_volume >= 50:
        liquidity_multiplier = 0.5
    else:
        liquidity_multiplier = 0.25

    # Exposure multiplier: limit over-concentration in one market
    remaining_budget = max(0.0, daily_budget - open_position_cost)
    exposure_multiplier = min(1.0, remaining_budget / daily_budget) if daily_budget > 0 else 0.0

    # Model uncertainty penalty
    uncertainty_penalty = 1.0 - (model_uncertainty * 0.5)

    raw_dollars = daily_budget * fractional_kelly
    position_dollars = (
        raw_dollars
        * confidence_multiplier
        * liquidity_multiplier
        * exposure_multiplier
        * uncertainty_penalty
    )

    # Hard caps: minimum $1, maximum 20% of daily budget
    max_position = daily_budget * 0.20
    position_dollars = max(1.0, min(position_dollars, max_position))
    position_dollars = round(position_dollars, 2)

    return {
        "position_dollars": position_dollars,
        "kelly_raw": round(kelly_raw, 4),
        "fractional_kelly": round(fractional_kelly, 4),
        "confidence_multiplier": round(confidence_multiplier, 3),
        "liquidity_multiplier": round(liquidity_multiplier, 3),
        "exposure_multiplier": round(exposure_multiplier, 3),
    }
