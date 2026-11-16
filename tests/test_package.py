"""Smoke tests: the package imports and its command-line interface parses."""

import pytest

import ptc_mil
from ptc_mil.cli import COMMANDS, build_parser, main


def test_version_is_defined() -> None:
    assert isinstance(ptc_mil.__version__, str)


def test_cli_requires_a_step(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "STEP" in capsys.readouterr().err


@pytest.mark.parametrize("step", sorted(COMMANDS))
def test_every_step_has_help(step: str) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([step, "--help"])
    assert exc.value.code == 0
