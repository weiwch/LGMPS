from __future__ import annotations

import argparse

from lgmps.data.indexed_jsonl import build_offset_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persistent mmap byte-offset index for JSONL data.")
    parser.add_argument("data_file")
    parser.add_argument("--index-file", default=None)
    arguments = parser.parse_args()
    print(build_offset_index(arguments.data_file, arguments.index_file))


if __name__ == "__main__":
    main()
