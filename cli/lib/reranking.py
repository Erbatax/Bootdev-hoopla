import json
import time
from typing import Optional

from sentence_transformers import CrossEncoder

from .search_utils import format_search_result
from .llm import llm_client, DEFAULT_LLM_MODEL


def individual_rerank(
    query: str, documents: list[dict], limit: Optional[int] = None
) -> list[dict]:
    reranked_results = []
    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

        response = llm_client.models.generate_content(
            model=DEFAULT_LLM_MODEL, contents=prompt
        )
        reranked_score = float((response.text or "").strip().strip('"'))
        reranked_results.append({**doc, "reranked_individual_score": reranked_score})

        time.sleep(3)  # To avoid rate limiting

    sorted_results = sorted(
        reranked_results, key=lambda x: x["reranked_individual_score"], reverse=True
    )

    if limit is None:
        limit = len(sorted_results)
    limited_results = sorted_results[:limit]

    return limited_results


def batch_rerank(
    query: str, documents: list[dict], limit: Optional[int] = None
) -> list[dict]:
    doc_list_str = ""
    for doc in documents:
        doc_list_str += f'- ID: {doc["id"]}, Title: {doc.get("title", "")}, Description: {doc.get("document", "")}\n'

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""

    response = llm_client.models.generate_content(
        model=DEFAULT_LLM_MODEL, contents=prompt
    )
    response_text = (
        (response.text or "")
        .strip()
        .strip('"')
        .removeprefix("```json")
        .removesuffix("```")
    )
    new_ranks = json.loads(response_text)

    sorted_results = []
    id_to_doc = {doc["id"]: doc for doc in documents}
    for i, doc_id in enumerate(new_ranks, 1):
        if doc_id in id_to_doc:
            sorted_results.append({**id_to_doc[doc_id], "reranked_batch_rank": i})

    if limit is None:
        limit = len(sorted_results)
    limited_results = sorted_results[:limit]

    return limited_results


def cross_encoder_rerank(
    query: str, documents: list[dict], limit: Optional[int] = None
) -> list[dict]:
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    zipped_docs: list[dict] = []
    for doc, score in zip(documents, scores):
        zipped_docs.append({**doc, "reranked_cross_encoder_score": score})

    sorted_results = sorted(
        zipped_docs, key=lambda x: x["reranked_cross_encoder_score"], reverse=True
    )

    if limit is None:
        limit = len(sorted_results)
    limited_results = sorted_results[:limit]

    return limited_results


def rerank(
    query: str,
    documents: list[dict],
    method: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    match method:
        case "individual":
            return individual_rerank(query, documents, limit)
        case "batch":
            return batch_rerank(query, documents, limit)
        case "cross_encoder":
            return cross_encoder_rerank(query, documents, limit)
        case _:
            return documents
