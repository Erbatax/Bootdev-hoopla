import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.augmented_generation import (
    citations_command,
    question_command,
    rag_command,
    summarize_command,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results using LLM"
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to evaluate",
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Generate answer with citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to cite",
    )

    question_parser = subparsers.add_parser(
        "question", help="Ask a question based on search results"
    )
    question_parser.add_argument(
        "query", type=str, help="Search query for question answering"
    )
    question_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to use for question answering",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)

            print(f"Search Results:")
            for res in result["search_results"]:
                print(f"- {res['title']}")
            if result["rag_answer"]:
                print(f"\nRAG Response:\n{result['rag_answer']}")
        case "summarize":
            result = summarize_command(args.query, limit=args.limit)

            print(f"Search Results:")
            for res in result["search_results"]:
                print(f"- {res['title']}")
            if result["summary"]:
                print(f"\nLLM Summary:\n{result['summary']}")
        case "citations":
            result = citations_command(args.query, limit=args.limit)

            print(f"Search Results:")
            for res in result["search_results"]:
                print(f"- {res['title']}")
            if result["citations"]:
                print(f"\nLLM Answer:\n{result['citations']}")
        case "question":
            result = question_command(args.query, limit=args.limit)

            print(f"Search Results:")
            for res in result["search_results"]:
                print(f"- {res['title']}")
            if result["question_answer"]:
                print(f"\nAnswer:\n{result['question_answer']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
