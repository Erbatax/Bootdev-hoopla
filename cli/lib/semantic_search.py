from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
    MOVIE_EMBEDDINGS_PATH,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    SCORE_PRECISION,
    format_search_result,
    load_movies,
)
import re
import json
from collections import defaultdict


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list[dict[str, str]] = None
        self.document_map = {}

    def build_embeddings(self, documents: list[dict[str, str]]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict[str, str]]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        self.embeddings = self.build_embeddings(documents)
        return self.embeddings

    def generate_embedding(self, text):
        if len(text.strip()) == 0:
            raise ValueError("Text is required")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def search(self, query: str, limit: int) -> list[dict[str, str | float]]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(load_movies())
    results = semantic_search.search_chunk(query, limit)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description']}\n")


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def chunk_command(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def semantic_chunk_command(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, sentences in enumerate(chunks, 1):
        print(f"{i}. {sentences}")


def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    stripped_text = text.strip()
    if len(stripped_text) == 0:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", stripped_text)
    processed_sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(processed_sentences) == 1 and processed_sentences[0].endswith(
        (".", "?", "!")
    ):
        processed_sentences = stripped_text  # Treat all text as a single sentence
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict[str, str]]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk(
                text,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": idx, "chunk_idx": i, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            if len(self.chunk_embeddings) == len(self.chunk_metadata):
                return self.chunk_embeddings

        self.chunk_embeddings = self.build_chunk_embeddings(documents)
        return self.chunk_embeddings

    def search_chunk(
        self, query: str, limit: int = DEFAULT_SEMANTIC_LIMIT
    ) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )

        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": i,
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": similarity,
                }
            )

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for movie_idx, score in sorted_movies[:limit]:
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )

        return results


def embed_chunks_command() -> np.ndarray:
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    return chunked_semantic_search.load_or_create_chunk_embeddings(movies)


def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    results = chunked_semantic_search.search_chunk(query, limit)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")

    return results
