from __future__ import annotations

import argparse
import json

from lgmps.configuration import PUBLIC_DATASETS, load_experiment
from lgmps.inference import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run beam retrieval evaluation and paper-style metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=PUBLIC_DATASETS)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--result-file", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--index-stride", type=int, default=1)
    parser.add_argument("--index-offset", type=int, default=0)
    parser.add_argument("--capture-artifacts", action="store_true")
    parser.add_argument(
        "--moe-intervention",
        choices=(
            "average_categories",
            "full",
            "shared_only",
            "routed_only",
            "swap_categories",
        ),
        default="full",
    )
    arguments = parser.parse_args()
    config = load_experiment(arguments.config, dataset=arguments.dataset)
    if arguments.batch_size is not None:
        if arguments.batch_size <= 0:
            parser.error("--batch-size must be positive")
        config.inference.batch_size = arguments.batch_size
    if arguments.result_file:
        config.inference.result_file = arguments.result_file
    summary = evaluate_checkpoint(
        config,
        checkpoint_override=arguments.checkpoint,
        evaluation_limit=arguments.limit,
        start_index=arguments.start_index,
        end_index=arguments.end_index,
        index_stride=arguments.index_stride,
        index_offset=arguments.index_offset,
        capture_artifacts=arguments.capture_artifacts,
        moe_intervention=arguments.moe_intervention,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
