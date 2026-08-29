"""Load backend/.env into the process environment at entry points."""

import os

from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")


def load_environment() -> None:
    """Populate ``os.environ`` from ``backend/.env`` if present.

    Existing process variables win (``override=False``), so this never clobbers
    values set by the shell or an orchestration runtime. Safe to call
    repeatedly; subsequent calls are essentially no-ops on the same values.
    """
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
