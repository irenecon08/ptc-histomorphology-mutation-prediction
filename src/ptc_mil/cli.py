"""Command-line entry point for the ptc_mil pipeline.

Pipeline stages (split, tile, embed, train, calibrate, evaluate) are added as
subcommands in later steps of the refactor.
"""

import argparse

from ptc_mil import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ptc-mil",
        description="Driver-alteration prediction from H&E whole-slide images.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
