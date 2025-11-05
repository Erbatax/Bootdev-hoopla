#!/usr/bin/env python3

import argparse
from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
    tfidf_command,
    bm25_idf_command,
    bm25_tf_command,
    bm25_search_command,
    BM25_K1,
    BM25_B,
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term to get frequency for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get tfidf of a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get score for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return",
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            display_search_results(results)
        case "tf":
            doc_id = args.doc_id
            term = args.term
            tf = tf_command(doc_id, term)
            print(f"Term frequency for {term} in document {doc_id}: {tf}")
        case "idf":
            term = args.term
            idf = idf_command(term)
            print(f"Inverse document frequency of '{term}': {idf:.2f}")
        case "tfidf":
            doc_id = args.doc_id
            term = args.term
            tfidf = tfidf_command(doc_id, term)
            print(f"TFIDF score of '{term}' in document {doc_id}: {tfidf:.2f}")
        case "bm25idf":
            term = args.term
            bm25_idf = bm25_idf_command(term)
            print(f"BM25 IDF score of '{term}': {bm25_idf:.2f}")
        case "bm25tf":
            doc_id = args.doc_id
            term = args.term
            k1 = args.k1
            b = args.b
            bm25_tf = bm25_tf_command(doc_id, term, k1, b)
            print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25_tf:.2f}")
        case "bm25search":
            query = args.query
            limit = args.limit
            print(f"Searching for: {query}")
            results = bm25_search_command(query, limit)
            display_search_results_with_score(results)
        case _:
            parser.print_help()


def display_search_results(results: list[dict]) -> None:
    for i, movie in enumerate(results, start=1):
        print(f"{i}. ({movie['id']}) {movie['title']}")


def display_search_results_with_score(results: list[(dict, float)]) -> None:
    for movie, score in results:
        print(f"({movie['id']}) {movie['title']} - Score: {score:.2f}")


if __name__ == "__main__":
    main()
