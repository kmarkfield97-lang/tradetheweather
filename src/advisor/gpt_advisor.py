"""
GPT-powered strategy advisor.
Calls OpenAI once per day to review recent trading history and suggest
improvements to the trading strategy.
"""

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

GPT_MODEL = "gpt-4o"

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
ADVICE_LOCK_FILE = os.path.join(DATA_DIR, "last_gpt_advice_date.txt")
ADVICE_LOG_FILE = os.path.join(DATA_DIR, "gpt_advice_log.json")
CHANGELOG_FILE = os.path.join(DATA_DIR, "CHANGELOG.md")
ERRORS_FILE = os.path.join(DATA_DIR, "forecast_errors.json")
HISTORY_FILE = os.path.join(DATA_DIR, "trade_history.json")
LEARNING_LOG_FILE = os.path.join(DATA_DIR, "learning_log.json")


def _load_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _load_text(path: str, default: str = "") -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return default


def _save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _already_ran_today() -> bool:
    last_date_str = _load_text(ADVICE_LOCK_FILE)
    return last_date_str == date.today().isoformat()


def _mark_ran_today():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ADVICE_LOCK_FILE, "w") as f:
        f.write(date.today().isoformat())


def _build_prompt(history: list, learning_log: list, forecast_errors: dict) -> str:
    """Builds the GPT prompt from recent trading data."""
    recent_history = history[-7:] if history else []
    recent_log = learning_log[-7:] if learning_log else []
    recent_errors = (forecast_errors.get("forecast_errors", []) or [])[-20:]

    # Summarize performance
    if recent_history:
        total_pnl = sum(r.get("pnl_dollars", 0) for r in recent_history)
        wins = sum(1 for r in recent_history if r.get("pnl_dollars", 0) > 0)
        win_rate = wins / len(recent_history)
    else:
        total_pnl = 0.0
        wins = 0
        win_rate = 0.0

    # Summarize takeaways
    takeaways = [r.get("day_takeaway", "") for r in recent_log if r.get("day_takeaway")]

    # Summarize forecast errors
    city_errors: dict = {}
    for e in recent_errors:
        city = e.get("city", "?")
        mtype = e.get("market_type", "?")
        err = e.get("error", 0.0)
        key = f"{city}/{mtype}"
        city_errors.setdefault(key, []).append(err)

    error_summary = []
    for key, errs in city_errors.items():
        mean_err = sum(errs) / len(errs)
        error_summary.append(f"  {key}: mean_bias={mean_err:+.1f}°F over {len(errs)} days")

    prompt = f"""You are a quantitative trading advisor for a Kalshi weather prediction market bot.

The bot trades binary YES/NO contracts on US city weather outcomes (temperature highs/lows, rain, snow).

RECENT PERFORMANCE (last 7 trading days):
- Total P&L: ${total_pnl:+.2f}
- Win rate: {win_rate:.0%} ({wins}/{len(recent_history)} days profitable)

RECENT DAY TAKEAWAYS:
{chr(10).join(f"- {t}" for t in takeaways) if takeaways else "- No takeaways recorded"}

NWS FORECAST BIAS BY CITY/MARKET TYPE (positive = NWS ran too warm):
{chr(10).join(error_summary) if error_summary else "- No bias data recorded"}

CURRENT STRATEGY PARAMETERS:
- MIN_EDGE = 0.07 (7 cents edge over market implied probability)
- MIN_LIQUIDITY = 50 contracts
- MAX_SPREAD = 15 cents
- MAX_TRADES_PER_DAY = 5
- MAX_POSITION_PCT = 20% of daily budget
- STOP_LOSS = -5% daily
- PROFIT_TARGET = +5% daily
- Using fractional Kelly (1/4 Kelly) for position sizing
- Bias correction: NWS forecast errors applied to probability estimates

OPEN QUESTION FOR TODAY:
Should we implement a market-disagreement filter that skips trades where the
Kalshi market price disagrees with our model by more than 25 percentage points?
Arguments for: large disagreements often mean the market has information we lack.
Arguments against: large edges are exactly what we want to capture.

Please provide 3-5 specific, actionable suggestions to improve the bot's profitability.
For each suggestion, include:
1. A short title
2. Priority: HIGH / MEDIUM / LOW
3. Category: additive (new feature) / update (change existing param) / replace (replace logic)
4. A clear description of what to change and why
5. Expected impact (dollars/percentage improvement)

Format your response as a JSON object with this structure:
{{
  "summary": "One-sentence overall assessment",
  "suggestions": [
    {{
      "id": "suggestion_1",
      "title": "...",
      "priority": "HIGH|MEDIUM|LOW",
      "category": "additive|update|replace",
      "description": "...",
      "expected_impact": "..."
    }}
  ],
  "market_disagreement_filter_recommendation": "yes|no|conditional",
  "market_disagreement_filter_reasoning": "..."
}}"""
    return prompt


def get_suggestions(force: bool = False) -> Optional[dict]:
    """
    Calls OpenAI once per day to get strategy suggestions.
    Returns structured JSON advice dict, or None on failure.
    Uses force=True to bypass the once-per-day lock.
    """
    if not force and _already_ran_today():
        logger.info("GPT advisor already ran today; skipping.")
        return load_latest_suggestions()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; skipping GPT advisor.")
        return None

    history = _load_json(HISTORY_FILE, [])
    learning_log = _load_json(LEARNING_LOG_FILE, [])
    forecast_errors = _load_json(ERRORS_FILE, {})

    prompt = _build_prompt(history, learning_log, forecast_errors)

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a quantitative trading strategy advisor. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        raw_content = response.choices[0].message.content or ""

        # Strip markdown code fences if present
        content = raw_content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        advice = json.loads(content)
        advice["generated_at"] = datetime.now(timezone.utc).isoformat()
        advice["model"] = GPT_MODEL

        # Append to log
        log = _load_json(ADVICE_LOG_FILE, [])
        if not isinstance(log, list):
            log = []
        log.append(advice)
        _save_json(ADVICE_LOG_FILE, log)

        _mark_ran_today()
        logger.info(f"GPT advisor completed: {len(advice.get('suggestions', []))} suggestions")
        return advice

    except json.JSONDecodeError as e:
        logger.error(f"GPT advisor: failed to parse JSON response: {e}")
        return None
    except Exception as e:
        logger.error(f"GPT advisor error: {e}")
        return None


def load_latest_suggestions() -> Optional[dict]:
    """Reads the most recent advice from the log without making an API call."""
    log = _load_json(ADVICE_LOG_FILE, [])
    if not isinstance(log, list) or not log:
        return None
    return log[-1]
