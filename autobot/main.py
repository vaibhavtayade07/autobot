"""Entrypoint: `python -m autobot` / `python -m autobot.main` -> cli.main()."""
from __future__ import annotations

import sys

from .cli import main


def run() -> None:
    sys.exit(main())


if __name__ == "__main__":
    run()
