from __future__ import annotations

import argparse

from lgmps.visualization import render_moe


def main() -> None:
    parser = argparse.ArgumentParser(description="Render expert-load heatmaps from MoE JSONL logs.")
    parser.add_argument("log_file")
    parser.add_argument("output_dir")
    arguments = parser.parse_args()
    for output in render_moe(arguments.log_file, arguments.output_dir):
        print(output)


if __name__ == "__main__":
    main()
