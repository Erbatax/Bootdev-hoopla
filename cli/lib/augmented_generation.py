import os
from dotenv import load_dotenv
from lib.hybrid_search import HybridSearch
from lib.search_utils import DEFAULT_SEARCH_LIMIT, RRF_K, SEARCH_MULTIPLIER, load_movies
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def generate_answer(
    search_results: list[dict], query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> str:
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{context}

Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model=model, contents=prompt)
    answer = (response.text or "").strip()
    return answer


def rag_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    searcher = HybridSearch(movies)

    search_results = searcher.rrf_search(
        query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = generate_answer(search_results, query, limit=limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "rag_answer": answer,
    }


def generate_summary(
    search_results: list[dict], query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> str:
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{context}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    response = client.models.generate_content(model=model, contents=prompt)
    answer = (response.text or "").strip()
    return answer


def summarize_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    searcher = HybridSearch(movies)

    search_results = searcher.rrf_search(query, k=60, limit=limit)

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    summary = generate_summary(search_results, query, limit=limit)

    return {
        "query": query,
        "search_results": search_results,
        "summary": summary,
    }


def generate_citations(
    search_results: list[dict], query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> str:
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{context}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    answer = (response.text or "").strip()
    return answer


def citations_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    searcher = HybridSearch(movies)

    search_results = searcher.rrf_search(query, k=60, limit=limit)

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    citations = generate_citations(search_results, query, limit=limit)

    return {
        "query": query,
        "search_results": search_results,
        "citations": citations,
    }


def generate_question_answer(
    search_results: list[dict], question: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> str:
    context = ""

    for result in search_results[:limit]:
        context += f"{result['title']}: {result['document']}\n\n"

    #     prompt = f"""Answer the following question based on the provided documents.

    # Question: {question}

    # Documents:
    # {context}

    # General instructions:
    # - Answer directly and concisely
    # - Use only information from the documents
    # - If the answer isn't in the documents, say "I don't have enough information"
    # - Cite sources when possible

    # Guidance on types of questions:
    # - Factual questions: Provide a direct answer
    # - Analytical questions: Compare and contrast information from the documents
    # - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

    # Answer:"""

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    answer = (response.text or "").strip()
    return answer


def question_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    searcher = HybridSearch(movies)

    search_results = searcher.rrf_search(query, k=60, limit=limit)

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    question_answer = generate_question_answer(search_results, query, limit=limit)

    return {
        "query": query,
        "search_results": search_results,
        "question_answer": question_answer,
    }
