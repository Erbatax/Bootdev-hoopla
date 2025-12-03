#!/usr/bin/env python3

import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.multimodal_search import image_search_command, verify_image_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verify_image_embedding_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to verify embedding",
    )

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search movies using an image"
    )
    image_search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to search",
    )
    image_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return",
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            result = image_search_command(args.image_path, limit=args.limit)
            for i, search_result in enumerate(result["search_results"], start=1):
                print(
                    f"{i}. {search_result['title']} (similarity: {search_result['similarity']:.3f})"
                )
                print(f"   {search_result['description'][:100]}...\n")
        case _:
            parser.exit(2, parser.format_help())


if __name__ == "__main__":
    main()
