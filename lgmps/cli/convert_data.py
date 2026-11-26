from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from lgmps.data.conversion import convert_legacy_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert unversioned KuaiSAR session JSONL with public field names into the canonical schema."
    )
    parser.add_argument("source_file")
    parser.add_argument("output_file")
    parser.add_argument("--index-file", default=None)
    parser.add_argument("--minimum-sessions", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    arguments = parser.parse_args()
    summary = convert_legacy_jsonl(
        arguments.source_file,
        arguments.output_file,
        index_file=arguments.index_file,
        minimum_sessions=arguments.minimum_sessions,
        overwrite=arguments.overwrite,
        skip_invalid=arguments.skip_invalid,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
