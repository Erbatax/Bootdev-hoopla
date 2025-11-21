import json
import os
from typing import Any

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

MOVIES_PATH = os.path.join(DATA_DIR, "movies.json")
STOP_WORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(DATA_DIR, "golden_dataset.json")


DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_SEMANTIC_LIMIT = 10
DOCUMENT_PREVIEW_LENGTH = 100

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")
SCORE_PRECISION = 4

DEFAULT_HYBRID_SEARCH_ALPHA = 0.5
DEFAULT_HYBRID_SEARCH_LIMIT = 5
DEFAULT_RRF_K = 60


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_words() -> list[str]:
    with open(STOP_WORDS_PATH, "r") as f:
        data = f.read()
    return data.splitlines()


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        data = json.load(f)
    return data


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
