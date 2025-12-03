import json
import os
from time import sleep

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5, debug: bool = False
) -> list[dict]:
    scored_docs = []

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

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)

    if debug:
        print(f"\nReranked Documents (Individual LLM Rerank) ({len(scored_docs)}):")
        for i, doc in enumerate(scored_docs[: limit * 2], 1):
            print(f" {i}. {doc['title']} (Score: {doc['individual_score']})")

    return scored_docs[:limit]


def llm_rerank_batch(
    query: str, documents: list[dict], limit: int = 5, debug: bool = False
) -> list[dict]:
    if not documents:
        return []

    doc_map = {}
    doc_list = []
    for doc in documents:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""

    response = client.models.generate_content(model=model, contents=prompt)
    ranking_text = (response.text or "").strip()

    parsed_ids = json.loads(ranking_text)

    reranked = []
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            reranked.append({**doc_map[doc_id], "batch_rank": i + 1})

    if debug:
        print(f"\nReranked Documents (Batch LLM Rerank) ({len(reranked)}):")
        for i, doc in enumerate(reranked[: limit * 2], 1):
            print(f" {i}. {doc['title']} (Rank: {doc['batch_rank']})")

    return reranked[:limit]


def cross_encoder_rerank(
    query: str, documents: list[dict], limit: int = 5, debug: bool = False
) -> list[dict]:
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    scores = cross_encoder.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["crossencoder_score"] = float(score)

    documents.sort(key=lambda x: x["crossencoder_score"], reverse=True)

    if debug:
        print(f"\nReranked Documents (Cross-Encoder Rerank) ({len(documents)}):")
        for i, doc in enumerate(documents[: limit * 2], 1):
            print(f" {i}. {doc['title']} (Score: {doc['crossencoder_score']})")

    return documents[:limit]


def rerank(
    query: str,
    documents: list[dict],
    method: str = "batch",
    limit: int = 5,
    debug: bool = False,
) -> list[dict]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit=limit, debug=debug)
    if method == "batch":
        return llm_rerank_batch(query, documents, limit=limit, debug=debug)
    if method == "cross_encoder":
        return cross_encoder_rerank(query, documents, limit=limit, debug=debug)
    else:
        return documents[:limit]
