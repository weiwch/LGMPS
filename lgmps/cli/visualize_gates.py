from __future__ import annotations

import argparse
import json

from lgmps.visualization import render_gates


def main() -> None:
    parser = argparse.ArgumentParser(description="Render transformed GDN alpha/beta research plots.")
    parser.add_argument("source")
    parser.add_argument("output_dir")
    parser.add_argument("--smoothing", type=int, default=21)
    arguments = parser.parse_args()
    print(
        json.dumps(
            render_gates(arguments.source, arguments.output_dir, smoothing=arguments.smoothing), indent=2
        )
    )


if __name__ == "__main__":
    main()
