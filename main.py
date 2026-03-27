"""
Entry point. Run this to start the bot.
"""

import logging
import asyncio
import os
import signal
import sys

from src.orchestrator import Orchestrator
from src.telegram.bot import TradeTheWeatherBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# Use an absolute path derived from this file's real location so the lock path
# is identical regardless of how the script is invoked (relative vs absolute,
# launchd vs terminal). os.path.dirname(__file__) returns "" when invoked as
# "python3 main.py", which resolved to a different CWD-relative path than the
# launchd absolute-path invocation — allowing two instances to hold separate
# "locks" on two different files simultaneously.
_HERE = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(_HERE, "data", "bot.pid")


def _acquire_pid_lock():
    """
    Ensures only one instance runs at a time using an atomic PID file.
    O_CREAT | O_EXCL guarantees only one process wins the race.
    Handles stale files left by SIGKILL or crash without finally: running.
    """
    while True:
        try:
            fd = os.open(PID_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return  # We hold the lock
        except FileExistsError:
            try:
                with open(PID_FILE) as f:
                    content = f.read().strip()
                old_pid = int(content)
                os.kill(old_pid, 0)  # raises ProcessLookupError if gone
                # Process is alive — verify it's actually our bot, not a recycled PID
                try:
                    with open(f"/proc/{old_pid}/cmdline") as f:
                        cmdline = f.read()
                except FileNotFoundError:
                    # macOS: use ps instead
                    import subprocess
                    result = subprocess.run(
                        ["ps", "-p", str(old_pid), "-o", "command="],
                        capture_output=True, text=True
                    )
                    cmdline = result.stdout
                if "main.py" not in cmdline and "tradetheweather" not in cmdline:
                    # PID was recycled by an unrelated process — treat as stale
                    logger.warning(
                        f"PID {old_pid} in lock file belongs to unrelated process "
                        f"({cmdline.strip()!r}). Treating as stale."
                    )
                    raise ProcessLookupError
                logger.error(
                    f"Another bot instance is already running (PID {old_pid}). Exiting."
                )
                sys.exit(1)
            except (ProcessLookupError, ValueError, OSError):
                logger.warning("Removing stale PID file and retrying.")
                try:
                    os.remove(PID_FILE)
                except FileNotFoundError:
                    pass


def _release_pid_lock():
    try:
        # Only remove the file if it still contains our PID — don't clobber a
        # lock that a new instance wrote after we were signalled.
        with open(PID_FILE) as f:
            pid_in_file = int(f.read().strip())
        if pid_in_file == os.getpid():
            os.remove(PID_FILE)
    except (FileNotFoundError, ValueError, OSError):
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
