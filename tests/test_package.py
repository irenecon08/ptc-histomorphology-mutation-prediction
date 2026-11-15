"""Smoke tests: the package installs, imports and exposes its command-line entry point."""

import ptc_mil
from ptc_mil.cli import main


def test_version_is_defined() -> None:
    assert isinstance(ptc_mil.__version__, str)


def test_cli_runs_without_arguments() -> None:
    assert main([]) == 0
