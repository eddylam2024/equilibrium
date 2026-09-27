"""Console entry point: ``equilibrium`` (and the short alias ``eq``)."""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING, format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), show_time=False, show_path=False)],
    )


def run() -> None:
    _configure_logging()
    if len(sys.argv) == 1 and sys.stdin.isatty() and sys.stdout.isatty():
        from equilibrium.cli.shell import run_shell

        run_shell()
        return
    from equilibrium.cli.commands import app

    app()


if __name__ == "__main__":
    run()
