"""CLI wiring scaffold for Praxis orchestrator."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Praxis CLI scaffold")
    parser.add_argument("command", help="Placeholder command")
    return parser


def main() -> None:
    raise NotImplementedError("CLI wiring is not implemented in scaffold.")


if __name__ == "__main__":
    main()
