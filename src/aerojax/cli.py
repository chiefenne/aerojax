"""Command line: ``aerojax run case.toml`` and ``aerojax check case.toml``."""

import argparse
import sys

from aerojax import __version__
from aerojax.config import ConfigError, load_case


def main(argv=None):
    """Entry point of the ``aerojax`` command."""
    parser = argparse.ArgumentParser(
        prog="aerojax",
        description="2D unstructured finite-volume Navier-Stokes solver",
    )
    parser.add_argument("--version", action="version", version=f"aerojax {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a case")
    run.add_argument("case", help="TOML case file")
    run.add_argument("--quiet", action="store_true", help="no console output")
    check = sub.add_parser("check", help="validate a case and print the resolved file")
    check.add_argument("case", help="TOML case file")
    args = parser.parse_args(argv)

    try:
        if args.command == "check":
            config = load_case(args.case)
            print(config.to_toml())
            for note in config.notices:
                print(f"# notice: {note}")
            return 0
        from aerojax.run import run_case

        result = run_case(args.case, console=not args.quiet)
        return 0 if result.converged else 2
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
