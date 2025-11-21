import os
import pickle
import string
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    CACHE_DIR,
    format_search_result,
    load_movies,
    load_stop_words,
)

BM25_K1 = 1.5  # Saturation effect parameter
BM25_B = 0.75  # Normalization parameter

stemmer = PorterStemmer()
stop_words = load_stop_words()


class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, dict]
    term_frequency: dict[int, Counter]
    doc_lengths: dict[int, int]

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequency = defaultdict(Counter)
        self.doc_lengths = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_text = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_text)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequency, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                "Inverted index file not found in cache. Please run the 'build' command first."
            )
        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(
                "Docmap file not found in cache. Please run the 'build' command first."
            )
        if not os.path.exists(self.term_frequencies_path):
            raise FileNotFoundError(
                "Term frequencies file not found in cache. Please run the 'build' command first."
            )

        if not os.path.exists(self.doc_lengths_path):
            raise FileNotFoundError(
                "Doc lengths file not found in cache. Please run the 'build' command first."
            )

        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequency = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        term_tokens = tokenize_text(term)
        if len(term_tokens) > 1:
            raise ValueError("get_documents only supports single terms.")
        doc_ids = self.index.get(term_tokens[0], set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        term_tokens = tokenize_text(term)
        if len(term_tokens) > 1:
            raise ValueError("get_tf only supports single terms.")
        tf = 0
        tf += self.term_frequency[doc_id][term_tokens[0]]
        return tf

    def get_idf(self, term: str) -> float:
        total_docs = len(self.docmap)
        term_doc_ids = self.get_documents(term)
        term_doc_freq = len(term_doc_ids)
        import math

        # Basic IDF calculation. log(N / df)
        # we "solved" division by zero by adding 1 to numerator and denominator
        idf = math.log((total_docs + 1) / (term_doc_freq + 1))
        return idf

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        tfidf = tf * idf
        return tfidf

    def get_bm25_idf(self, term: str) -> float:
        term_tokens = tokenize_text(term)
        if len(term_tokens) > 1:
            raise ValueError("get_tf only supports single terms.")
        total_docs = len(self.docmap)
        term_doc_ids = self.get_documents(term)
        term_doc_freq = len(term_doc_ids)

        import math

        # Better IDF calculation.
        # 1. Avoids negative values for high-frequency terms
        # 2. Avoid that the really rare terms have too much weight
        # 3. Avoid division by zero
        bm25_idf = math.log(
            (total_docs - term_doc_freq + 0.5) / (term_doc_freq + 0.5) + 1
        )
        return bm25_idf

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        # Length normalization factor
        # Scale term frequency by inverse document length.
        # Shorter documents score are boosted, longer documents score are reduced
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        # Term Frequency Saturation.
        # Many occurences of the same word score less than the same number of occurence of various words
        saturated_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return saturated_tf

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit) -> list[(dict, float)]:
        query_tokens = tokenize_text(query)
        score_map = defaultdict(float)
        for doc_id in self.docmap.keys():
            doc_score = 0.0
            for query_token in query_tokens:
                doc_score += self.bm25(doc_id, query_token)
            score_map[doc_id] = doc_score
        # Sort by score descending
        sorted_docs = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequency[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        total_docs = len(self.doc_lengths)
        if total_docs == 0:
            return 0.0
        return total_length / total_docs


# End of InvertedIndex class


def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    inverted_index.load()

    result_ids = set()
    query_tokens = tokenize_text(query)
    for query_token in query_tokens:
        doc_ids = inverted_index.get_documents(query_token)
        result_ids.update(doc_ids)
        if len(result_ids) >= limit:
            break

    result_docs = []
    for doc_id in sorted(list(result_ids))[:limit]:
        result_docs.append(inverted_index.docmap[doc_id])
    return result_docs


# Get term frequency in a document
def tf_command(doc_id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tf(doc_id, term)


# Get inverse document frequency
def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_idf(term)


# Get TF-IDF score
def tfidf_command(doc_id: int, term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_tfidf(doc_id, term)


# Get BM25 IDF score
def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_idf(term)


# Get BM25 TF score
def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_tf(doc_id, term, k1, b)


def bm25_search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[(dict, float)]:
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.bm25_search(query, limit)


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


import unicodedata


def preprocess_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # # Replace accented characters
    # text = unicodedata.normalize("NFKD", text)
    # # Convert special characters
    # chars_to_replace_by_space = "—–‐-."
    # text = text.translate(str.maketrans({c: " " for c in chars_to_replace_by_space}))
    # text = text.translate(str.maketrans("", "", "“”‘’«»₣€₹£¡¿"))  # Remove special characters
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    # Split words
    tokens = text.split()
    # Remove empty tokens
    tokens = filter(lambda token: len(token) > 0, tokens)
    # Remove stop words
    tokens = filter(lambda token: token not in stop_words, tokens)
    # Apply stemming
    tokens = map(lambda token: stemmer.stem(token), tokens)
    return list(tokens)
