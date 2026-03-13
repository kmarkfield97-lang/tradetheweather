"""
Entry point. Run this to start the bot.
"""

import logging
import asyncio
import os
import sys

from src.orchestrator import Orchestrator
from src.telegram.bot import TradeTheWeatherBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

PID_FILE = os.path.join(os.path.dirname(__file__), "data", "bot.pid")


def _acquire_pid_lock():
    """Ensures only one instance runs at a time. Exits if another instance is already running."""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                old_pid = int(f.read().strip())
            # Check if process is actually still running
            os.kill(old_pid, 0)
            logger.error(f"Another bot instance is already running (PID {old_pid}). Exiting.")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            # Stale PID file — old process is gone
            pass
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _release_pid_lock():
    try:
        os.remove(PID_FILE)
    except FileNotFoundError:
        pass


def main():
    _acquire_pid_lock()
    try:
        orchestrator = Orchestrator()
        bot = TradeTheWeatherBot(orchestrator=orchestrator)
        orchestrator.set_bot(bot)

        # Start scheduler inside the event loop that python-telegram-bot creates
        async def post_init(application):
            orchestrator.start_scheduler()

        from telegram.ext import Application
        orchestrator.bot.app.post_init = post_init

        bot.run()
    finally:
        _release_pid_lock()


if __name__ == "__main__":
    main()
