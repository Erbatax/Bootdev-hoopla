import os

from .reranking import rerank

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    DEFAULT_HYBRID_SEARCH_LIMIT,
    DEFAULT_HYBRID_SEARCH_ALPHA,
    DEFAULT_RRF_K,
    load_movies,
    format_search_result,
)
from collections import defaultdict
from typing import Optional
from .query_enhancement import enhance_query


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=DEFAULT_HYBRID_SEARCH_LIMIT):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunk(query, limit * 500)

        combined = combine_search_results(bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunk(query, limit * 500)

        fused = reciprocal_rank_fusion(bm25_results, semantic_results, k)
        return fused[:limit]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))
    return normalized_scores


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[dict],
    semantic_results: list[dict],
    alpha: float = DEFAULT_HYBRID_SEARCH_ALPHA,
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)

    combined_scores = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25_score"],
            semantic_score=data["semantic_score"],
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def weighted_score_command(
    query: str,
    alpha: float = DEFAULT_HYBRID_SEARCH_ALPHA,
    limit: int = DEFAULT_HYBRID_SEARCH_LIMIT,
) -> list[dict]:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)
    return results


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def reciprocal_rank_fusion(
    bm25_results: list[dict], semantic_results: list[dict], k: int = DEFAULT_RRF_K
) -> list[dict]:
    rrf_scores = {}

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["bm25_rank"] is None:
            rrf_scores[doc_id]["bm25_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["semantic_rank"] is None:
            rrf_scores[doc_id]["semantic_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    rrf_results = []
    for doc_id, data in rrf_scores.items():
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=data["rrf_score"],
            rrf_score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        rrf_results.append(result)

    return sorted(rrf_results, key=lambda x: x["score"], reverse=True)


def rrf_score_command(
    query: str,
    k: int = DEFAULT_RRF_K,
    enhance: Optional[str] = None,
    rerank_method: Optional[str] = None,
    limit: int = DEFAULT_HYBRID_SEARCH_LIMIT,
) -> list[dict]:
    documents = load_movies()
    searcher = HybridSearch(documents)

    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query

    search_limit = limit
    if rerank_method:
        search_limit = limit * 5
    results = searcher.rrf_search(query, k, search_limit)

    reranked = False
    if rerank_method:
        reranked = True
        rekanked_results = rerank(query, results, method=rerank_method, limit=limit)
        results = rekanked_results

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "rerank_method": rerank_method,
        "reranked": reranked,
        "query": query,
        "k": k,
        "results": results,
    }
