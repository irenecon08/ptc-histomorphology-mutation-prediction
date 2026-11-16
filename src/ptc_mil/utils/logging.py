"""Logging setup used by the command-line entry points."""

import logging


def setup_logging(verbose: bool = False) -> None:
    """Send log messages to the terminal with a timestamp and the module name."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
