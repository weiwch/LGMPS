from __future__ import annotations

import argparse
import json
from pathlib import Path

from lgmps.configuration import PUBLIC_DATASETS, load_experiment
from lgmps.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LGMPS model locally or with torchrun.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=PUBLIC_DATASETS)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--resume-mode", choices=("full", "weights_only"), default="full")
    arguments = parser.parse_args()
    config = load_experiment(arguments.config, dataset=arguments.dataset)
    if arguments.resume_from:
        config.training.resume_from = str(Path(arguments.resume_from).expanduser().resolve())
        config.training.resume_mode = arguments.resume_mode
    result = run_training(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
