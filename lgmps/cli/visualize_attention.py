from __future__ import annotations

import argparse

from lgmps.visualization import render_attention


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an attention heatmap from an evaluation artifact.")
    parser.add_argument("artifact_file")
    parser.add_argument("output_file")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--session", type=int, default=-1)
    parser.add_argument("--head", type=int, default=-1)
    arguments = parser.parse_args()
    print(
        render_attention(
            arguments.artifact_file,
            arguments.output_file,
            layer=arguments.layer,
            session=arguments.session,
            head=arguments.head,
        )
    )


if __name__ == "__main__":
    main()
