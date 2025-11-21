import argparse
from lib.semantic_search import (
    search_command,
    verify_model,
    verify_embeddings,
    embed_text,
    chunk_command,
    semantic_chunk_command,
    embed_chunks_command,
    search_chunked_command,
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies semantically")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit results"
    )

    subparsers.add_parser("verify", help="Verify model")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_parser = subparsers.add_parser("embed_text", help="Embed given text")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Embed given query text"
    )
    embed_query_parser.add_argument("text", type=str, help="Query text to embed")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk given text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        required=False,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        required=False,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantic chunk given text"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to semantic chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        required=False,
        default=DEFAULT_SEMANTIC_CHUNK_SIZE,
        help="Max semantic chunk size",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        required=False,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap",
    )

    subparsers.add_parser("embed_chunks", help="Embed chunked documents")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search chunked documents semantically"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Limit results"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            query = args.query
            limit = args.limit
            search_command(query, limit)
        case "verify":
            verify_command()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            text = args.text
            embed_text(text)
        case "embedquery":
            text = args.text
            embed_text(text)
        case "chunk":
            text = args.text
            chunk_size = args.chunk_size
            overlap = args.overlap
            chunk_command(text, chunk_size, overlap)
        case "semantic_chunk":
            text = args.text
            max_chunk_size = args.max_chunk_size
            overlap = args.overlap
            semantic_chunk_command(text, max_chunk_size, overlap)
        case "embed_chunks":
            embeddings = embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            query = args.query
            limit = args.limit
            search_chunked_command(query, limit)
        case _:
            parser.print_help()


def verify_command():
    verify_model()


if __name__ == "__main__":
    main()
