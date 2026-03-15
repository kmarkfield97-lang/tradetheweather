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
    """
    Ensures only one instance runs at a time using an atomic file creation.
    O_CREAT | O_EXCL guarantees only one process wins the race.
    Falls back to checking for a stale PID if the file already exists.
    """
    while True:
        try:
            # Atomic: fails immediately if file already exists
            fd = os.open(PID_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return  # We hold the lock
        except FileExistsError:
            # File exists — check if the owning process is still alive
            try:
                with open(PID_FILE) as f:
                    old_pid = int(f.read().strip())
                os.kill(old_pid, 0)  # raises if process is gone
                logger.error(f"Another bot instance is already running (PID {old_pid}). Exiting.")
                sys.exit(1)
            except (ProcessLookupError, ValueError):
                # Stale PID file — remove it and retry atomically
                logger.warning("Removing stale PID file and retrying.")
                try:
                    os.remove(PID_FILE)
                except FileNotFoundError:
                    pass


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
