import argparse
from lib.hybrid_search import (
    normalize_scores,
    weighted_score_command,
    rrf_score_command,
)
from lib.search_utils import (
    DEFAULT_HYBRID_SEARCH_ALPHA,
    DEFAULT_HYBRID_SEARCH_LIMIT,
    DEFAULT_RRF_K,
)
import os
from dotenv import load_dotenv
from google import genai


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Scores")

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_HYBRID_SEARCH_ALPHA,
        help="Weight for BM25 scores",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_HYBRID_SEARCH_LIMIT, help="Limit results"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform RRF hybrid search"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "k", type=int, nargs="?", default=DEFAULT_RRF_K, help="RRF k parameter"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_HYBRID_SEARCH_LIMIT, help="Limit results"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Rerank method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_scores(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_score_command(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                print(
                    f"{i}. {result[1]['document']['title']}\n"
                    f"Hybrid Score: {result[1]['hybrid_score']:.4f}\n"
                    f"BM25 Score: {result[1]['bm25_score']:.4f}, Semantic Score: {result[1]['semantic_score']:.4f}\n"
                    f"{result[1]['document']['description'][:100]}..."
                )
        case "rrf-search":
            result = rrf_score_command(
                args.query, args.k, args.enhance, args.rerank_method, args.limit
            )
            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )

            if result["reranked"]:
                print(
                    f"Reranking top {len(result['results'])} results using {result['rerank_method']} method...\n"
                )

            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )

            for i, res in enumerate(result["results"], 1):
                metadata = res.get("metadata", {})
                print(f"{i}. {res['title']}")
                if res.get("reranked_individual_score"):
                    print(
                        f"   Reranked Score: {res.get("reranked_individual_score", 0):.3f}/10"
                    )
                if res.get("reranked_batch_rank"):
                    print(f"   Rerank Rank: {res.get('reranked_batch_rank')}")
                if res.get("reranked_cross_encoder_score"):
                    print(
                        f"   Cross Encoder Score: {res.get('reranked_cross_encoder_score', 0):.3f}"
                    )
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
