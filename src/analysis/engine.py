"""
Trade analysis engine.
Scores weather markets using NWS forecasts, historical trends, moon phase,
and Kalshi market pricing. Only surfaces high-confidence, liquid opportunities.
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from src.kalshi.client import KalshiClient
from src.weather.pipeline import WeatherPipeline


# Minimum confidence edge over market price to recommend a trade
MIN_EDGE = 0.07        # 7 cents edge over implied probability
MIN_LIQUIDITY = 50     # minimum total contracts in orderbook
MAX_SPREAD = 15        # maximum acceptable spread (cents)
MAX_TRADES_PER_DAY = 5
MAX_POSITION_PCT = 0.20  # 20% of daily budget per trade


@dataclass
class TradeRecommendation:
    ticker: str
    market_title: str
    side: str                    # "yes" or "no"
    our_probability: float       # our estimated probability (0–1)
    market_price: int            # kalshi price in cents
    edge: float                  # our_prob - market_implied_prob
    confidence: str              # "high" / "medium" / "low"
    contracts: int               # number of contracts to buy
    cost_dollars: float          # total cost in dollars
    reasoning: str               # human-readable explanation
    city: str
    market_type: str             # "temp_high" / "temp_low" / "rain" / "snow"
    forecast_summary: str
    alerts: list[str] = field(default_factory=list)


class TradeAnalysisEngine:
    def __init__(self, kalshi: KalshiClient, weather: WeatherPipeline):
        self.kalshi = kalshi
        self.weather = weather
        self._bias_cache: dict = {}  # city:market_type -> bias_f, refreshed daily

    def _get_warm_bias(self, city: str, market_type: str, lookback: int = 7) -> float:
        """
        Returns the recent mean forecast error (°F) for a city+market_type.
        Positive value = NWS ran too warm; subtract from forecast before scoring.
        Capped at ±8°F to avoid over-correcting on sparse data.
        Requires at least 3 records; returns 0.0 if insufficient data.
        """
        import os, json as _json
        path = os.path.join(os.path.dirname(__file__), "../../data/forecast_errors.json")
        try:
            with open(path) as f:
                data = _json.load(f)
            errors = data.get("forecast_errors", [])
        except Exception:
            return 0.0

        city_key = city.upper()
        relevant = [
            e["error"] for e in errors
            if e.get("city", "").upper() == city_key
            and e.get("market_type") == market_type
            and isinstance(e.get("error"), (int, float))
        ]
        # Take most recent `lookback` records
        recent = relevant[-lookback:]
        if len(recent) < 3:
            return 0.0
        bias = sum(recent) / len(recent)
        # Cap: never correct more than 8°F
        return max(-8.0, min(8.0, bias))

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def get_recommendations(self, daily_budget: float, trades_used: int = 0) -> list[TradeRecommendation]:
        """
        Scans all open Kalshi weather markets, scores them, and returns
        ranked trade recommendations that fit within the daily rules.
        """
        trades_remaining = MAX_TRADES_PER_DAY - trades_used
        if trades_remaining <= 0:
            return []

        markets = self.kalshi.get_weather_markets()
        recommendations = []

        for market in markets:
            ticker = market.get("ticker", "")
            try:
                rec = self._evaluate_market(market, daily_budget)
                if rec:
                    recommendations.append(rec)
            except Exception:
                continue

        # Sort by edge descending (best opportunity first)
        recommendations.sort(key=lambda r: r.edge, reverse=True)

        # Cap to remaining trades allowed today
        return recommendations[:trades_remaining]

    # -------------------------------------------------------------------------
    # Market evaluation
    # -------------------------------------------------------------------------

    def _evaluate_market(self, market: dict, daily_budget: float) -> Optional[TradeRecommendation]:
        ticker = market.get("ticker", "")
        title = market.get("title", "")

        # Parse market type and city from ticker/title
        parsed = self._parse_market(ticker, title)
        if not parsed:
            return None
        city, market_type, threshold = parsed

        # Check liquidity first — skip illiquid markets immediately
        liquidity = self.kalshi.get_liquidity(ticker)
        if not liquidity["is_liquid"]:
            return None
        if liquidity["spread"] and liquidity["spread"] > MAX_SPREAD:
            return None

        # Get weather data
        report = self.weather.get_full_report(city)
        if not report["forecast"] or "error" in report.get("forecast", {}):
            return None

        # Score the market
        our_prob, reasoning, forecast_summary = self._score_market(
            market_type, threshold, report, city
        )
        if our_prob is None:
            return None

        # Determine best side
        yes_price = liquidity["best_yes_price"]
        no_price = liquidity["best_no_price"]
        if not yes_price or not no_price:
            return None

        market_yes_prob = yes_price / 100.0
        market_no_prob = no_price / 100.0

        yes_edge = our_prob - market_yes_prob
        no_edge = (1 - our_prob) - market_no_prob

        if yes_edge >= no_edge and yes_edge >= MIN_EDGE:
            side = "yes"
            edge = yes_edge
            price = yes_price
        elif no_edge > yes_edge and no_edge >= MIN_EDGE:
            side = "no"
            edge = no_edge
            price = no_price
        else:
            return None  # no edge

        # Kelly criterion (fractional) for position sizing
        prob_win = our_prob if side == "yes" else (1 - our_prob)
        prob_lose = 1 - prob_win
        odds = (100 - price) / price  # net odds if we win
        kelly = (prob_win * odds - prob_lose) / odds
        fractional_kelly = kelly * 0.25  # use 1/4 Kelly for safety

        max_position = daily_budget * MAX_POSITION_PCT
        position_dollars = min(daily_budget * fractional_kelly, max_position)
        position_dollars = max(1.0, round(position_dollars, 2))  # minimum $1

        contracts = math.floor(position_dollars / (price / 100))
        if contracts < 1:
            return None

        actual_cost = contracts * (price / 100)

        # Confidence tier
        if edge >= 0.15:
            confidence = "high"
        elif edge >= 0.10:
            confidence = "medium"
        else:
            confidence = "low"

        alerts = [a["event"] for a in report.get("alerts", [])]

        return TradeRecommendation(
            ticker=ticker,
            market_title=title,
            side=side,
            our_probability=round(our_prob, 3),
            market_price=price,
            edge=round(edge, 3),
            confidence=confidence,
            contracts=contracts,
            cost_dollars=round(actual_cost, 2),
            reasoning=reasoning,
            city=city,
            market_type=market_type,
            forecast_summary=forecast_summary,
            alerts=alerts,
        )

    # -------------------------------------------------------------------------
    # Market parser
    # -------------------------------------------------------------------------

    # Map Kalshi series ticker prefix → city name (must match US_CITIES in weather/pipeline.py)
    SERIES_CITY_MAP = {
        "KXHIGHTPHX": "PHOENIX", "KXHIGHTHOU": "HOUSTON", "KXHIGHTMIN": "MINNEAPOLIS",
        "KXHIGHTDAL": "DALLAS", "KXHIGHTLV": "LAS_VEGAS", "KXHIGHTSATX": "SAN_ANTONIO",
        "KXHIGHTBOS": "BOSTON", "KXHIGHTNOLA": "NEW_ORLEANS", "KXHIGHTSFO": "LOS_ANGELES",
        "KXHIGHTSEA": "SEATTLE", "KXHIGHTDC": "NYC",  # DC not in pipeline, map to nearest
        "KXHIGHTATL": "ATLANTA", "KXHIGHTOKC": "NASHVILLE",
        "KXLOWTCHI": "CHICAGO", "KXLOWTDEN": "DENVER", "KXLOWTNYC": "NYC",
        "KXLOWTPHIL": "PHILADELPHIA", "KXLOWTMIA": "MIAMI", "KXLOWTLAX": "LOS_ANGELES",
        "KXLOWTAUS": "DALLAS",
        "KXRAINNYC": "NYC", "KXRAINHOU": "HOUSTON", "KXRAINCHIM": "CHICAGO",
        "KXRAINSEA": "SEATTLE",
    }

    def _parse_market(self, ticker: str, title: str) -> Optional[tuple]:
        """
        Attempts to extract (city, market_type, threshold) from ticker/title.
        Returns None if we can't parse it confidently.
        """
        ticker_upper = ticker.upper()
        title_lower = title.lower()

        # First: try series prefix map (most reliable)
        city = None
        for prefix, mapped_city in self.SERIES_CITY_MAP.items():
            if ticker_upper.startswith(prefix):
                city = mapped_city
                break

        # Fallback: city name in ticker or title
        if not city:
            for c in ["NYC", "LOS_ANGELES", "CHICAGO", "HOUSTON", "PHOENIX",
                       "PHILADELPHIA", "SAN_ANTONIO", "SAN_DIEGO", "DALLAS",
                       "MIAMI", "ATLANTA", "BOSTON", "SEATTLE", "DENVER",
                       "MINNEAPOLIS", "NEW_ORLEANS", "LAS_VEGAS", "KANSAS_CITY",
                       "CLEVELAND", "NASHVILLE"]:
                if c.replace("_", "") in ticker_upper or c.replace("_", " ").lower() in title_lower:
                    city = c
                    break
        if not city:
            city_aliases = {
                "new york": "NYC", "los angeles": "LOS_ANGELES",
                "chicago": "CHICAGO", "atlanta": "ATLANTA", "boston": "BOSTON",
                "seattle": "SEATTLE", "denver": "DENVER", "miami": "MIAMI",
                "houston": "HOUSTON", "phoenix": "PHOENIX", "dallas": "DALLAS",
                "minneapolis": "MINNEAPOLIS", "philadelphia": "PHILADELPHIA",
                "san antonio": "SAN_ANTONIO", "new orleans": "NEW_ORLEANS",
                "las vegas": "LAS_VEGAS",
            }
            for alias, mapped in city_aliases.items():
                if alias in title_lower:
                    city = mapped
                    break
        if not city:
            return None

        market_type = None
        threshold = None

        # Also infer from series prefix
        series_prefix = next((p for p in self.SERIES_CITY_MAP if ticker_upper.startswith(p)), "")
        if "HIGH" in series_prefix or "HIGHT" in series_prefix:
            market_type = "temp_high"
        elif "LOW" in series_prefix or "LOWT" in series_prefix:
            market_type = "temp_low"
        elif "RAIN" in series_prefix:
            market_type = "rain"
        elif "SNOW" in series_prefix:
            market_type = "snow"
        elif any(kw in title_lower for kw in ["high temp", "high temperature", "maximum temperature", "max temp", "maximum temp"]):
            market_type = "temp_high"
        elif any(kw in title_lower for kw in ["low temp", "low temperature", "lowest temperature", "minimum temp"]):
            market_type = "temp_low"
        elif any(kw in title_lower for kw in ["rain", "precipitation", "precip"]):
            market_type = "rain"
        elif any(kw in title_lower for kw in ["snow", "snowfall", "snowstorm"]):
            market_type = "snow"
        else:
            return None

        # Extract threshold from ticker suffix first (e.g. -T96, -T89, -B95.5)
        import re
        ticker_thresh = re.search(r"-[TB]([\d.]+)(?:-|$)", ticker_upper)
        if ticker_thresh:
            threshold = int(float(ticker_thresh.group(1)))
        else:
            # Fall back: extract temperature number from title (°F)
            m = re.search(r"(\d+)\s*°", title)
            if m:
                threshold = int(m.group(1))
            else:
                nums = re.findall(r"\b(\d{2,3})\b", title)
                if nums:
                    threshold = int(nums[0])

        return city, market_type, threshold

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    def _score_market(
        self, market_type: str, threshold: Optional[int], report: dict, city: str = ""
    ) -> tuple[Optional[float], str, str]:
        """
        Returns (probability, reasoning, forecast_summary).
        probability is our estimated P(YES) for the market.
        """
        forecast = report.get("forecast", {})
        hourly = report.get("hourly", [])
        moon = report.get("moon", {})
        temp_trend = report.get("temp_trend")
        alerts = report.get("alerts", [])

        high_temp = forecast.get("high_temp_f")
        low_temp = forecast.get("low_temp_f")
        precip_chance = forecast.get("precip_chance") or 0
        short_fc = forecast.get("short_forecast", "")
        detailed_fc = forecast.get("detailed_forecast", "")
        moon_modifier = moon.get("precip_modifier", 0)

        forecast_summary = f"{short_fc} | High: {high_temp}°F Low: {low_temp}°F | Precip: {precip_chance}%"

        if market_type == "temp_high" and threshold is not None and high_temp is not None:
            # Apply warm-bias correction: NWS consistently runs too warm for most cities.
            # Subtract the city's recent mean forecast error before computing probability.
            warm_bias = self._get_warm_bias(city, "temp_high") if city else 0.0
            corrected_high = high_temp - warm_bias
            diff = corrected_high - threshold
            # Logistic curve: sigmoid of difference scaled by uncertainty
            uncertainty = 3.0  # degrees F spread
            trend_adj = (temp_trend or 0) * 2  # warming trend nudges up
            adjusted_diff = diff + trend_adj
            prob = self._sigmoid(adjusted_diff / uncertainty)
            bias_note = f" [bias-corrected: {high_temp}°F - {warm_bias:+.1f}°F bias = {corrected_high:.1f}°F]" if warm_bias != 0.0 else ""
            reasoning = (
                f"NWS forecasts high of {high_temp}°F vs market threshold of {threshold}°F "
                f"(diff: {diff:+.1f}°F){bias_note}. "
                f"Temp trend: {'warming' if (temp_trend or 0) > 0 else 'cooling'} "
                f"({temp_trend:+.2f}°F/hr). Moon: {moon['phase_name']}."
            )
            return prob, reasoning, forecast_summary

        elif market_type == "temp_low" and threshold is not None and low_temp is not None:
            # Apply warm-bias correction to low temp too (inverted: warm bias raises lows)
            warm_bias = self._get_warm_bias(city, "temp_low") if city else 0.0
            corrected_low = low_temp - warm_bias
            diff = corrected_low - threshold
            uncertainty = 3.0
            trend_adj = (temp_trend or 0) * 2
            adjusted_diff = diff + trend_adj
            prob = self._sigmoid(adjusted_diff / uncertainty)
            bias_note = f" [bias-corrected: {low_temp}°F - {warm_bias:+.1f}°F bias = {corrected_low:.1f}°F]" if warm_bias != 0.0 else ""
            reasoning = (
                f"NWS forecasts low of {low_temp}°F vs market threshold of {threshold}°F "
                f"(diff: {diff:+.1f}°F){bias_note}. "
                f"Temp trend: {'warming' if (temp_trend or 0) > 0 else 'cooling'} "
                f"({temp_trend:+.2f}°F/hr). Moon: {moon['phase_name']}."
            )
            return prob, reasoning, forecast_summary

        elif market_type == "rain":
            # Probability of rain
            base_prob = precip_chance / 100.0
            # Moon phase adds slight modifier
            prob = min(1.0, base_prob + moon_modifier)
            # Check alerts for severe weather
            has_rain_alert = any("rain" in a["event"].lower() or "flood" in a["event"].lower() for a in alerts)
            if has_rain_alert:
                prob = min(1.0, prob + 0.10)
            reasoning = (
                f"NWS precipitation probability: {precip_chance}%. "
                f"Moon phase ({moon['phase_name']}) modifier: {moon_modifier:+.0%}. "
                f"{'Active rain/flood alerts. ' if has_rain_alert else ''}"
                f"Forecast: {short_fc}."
            )
            return prob, reasoning, forecast_summary

        elif market_type == "snow":
            # Similar to rain but check for snow keywords
            has_snow = any(kw in detailed_fc.lower() for kw in ["snow", "flurr", "blizzard", "winter storm"])
            snow_alert = any("snow" in a["event"].lower() or "winter" in a["event"].lower() for a in alerts)
            base_prob = precip_chance / 100.0 if has_snow else 0.05
            prob = min(1.0, base_prob + (0.15 if snow_alert else 0))
            reasoning = (
                f"Snow keywords in forecast: {'yes' if has_snow else 'no'}. "
                f"Precip chance: {precip_chance}%. "
                f"Winter weather alerts: {'yes' if snow_alert else 'no'}. "
                f"Moon: {moon['phase_name']}."
            )
            return prob, reasoning, forecast_summary

        return None, "", ""

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Logistic sigmoid, returns value in (0, 1)."""
        return 1.0 / (1.0 + math.exp(-x))

    def format_recommendation(self, rec: TradeRecommendation, rank: int) -> str:
        """Returns a formatted string for Telegram display."""
        edge_pct = f"{rec.edge * 100:.1f}%"
        our_prob_pct = f"{rec.our_probability * 100:.1f}%"
        market_implied = f"{rec.market_price}¢"
        alert_str = f"\n⚠️ Alerts: {', '.join(rec.alerts)}" if rec.alerts else ""

        return (
            f"Trade #{rank} — {rec.confidence.upper()} CONFIDENCE\n"
            f"Market: {rec.market_title}\n"
            f"Ticker: {rec.ticker}\n"
            f"Side: {rec.side.upper()} @ {market_implied}\n"
            f"Our probability: {our_prob_pct} | Edge: +{edge_pct}\n"
            f"Contracts: {rec.contracts} | Cost: ${rec.cost_dollars:.2f}\n"
            f"\nForecast: {rec.forecast_summary}\n"
            f"\nReasoning:\n{rec.reasoning}"
            f"{alert_str}"
        )
