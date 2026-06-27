#!/usr/bin/env python3
"""Extract text from a WARC file.

$ uv run python scripts/extract_warc.py --help
usage: extract_warc.py [-h] -i INPUT -o OUTPUT [-n LIMIT]

Extract clean plain text from HTML records inside a WARC file.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input WARC file (supports .warc or .warc.gz)
  -o OUTPUT, --output OUTPUT
                        Path to the output text/jsonl file
  -n LIMIT, --limit LIMIT
                        Maximum number of documents to extract (default: 10)

Example:
uv run python scripts/extract_warc.py \
    -i local-shared-data/CC/example.warc.gz \
    -o data/CC/example_warc_extracted.txt
"""

import argparse
import os
import sys

from cs336_data.data_processing import extract_text_from_warc


def main():
    parser = argparse.ArgumentParser(
        description="Extract clean plain text from HTML records inside a WARC file."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the input WARC file (supports .warc or .warc.gz)",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output text/jsonl file"
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        help="Maximum number of documents to extract (default: 10)",
    )

    args = parser.parse_args()

    # Create the output directory if it does not exist.
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        texts = extract_text_from_warc(args.input, args.limit)
        # Write to output.
        with open(args.output, "w", encoding="utf-8") as out_f:
            for item in texts:
                out_f.write(item["text"] + "\n")

        print(f"Successfully extracted {len(texts)} documents to {args.output}")
    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
