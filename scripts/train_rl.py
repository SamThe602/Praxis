"""CLI entry point for the lightweight PPO trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from packages.guidance.trainer_rl import train_rl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional override for the training config"
    )
    args = parser.parse_args()

    config_path = str(args.config) if args.config is not None else None
    metrics = train_rl(config_path=config_path)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
