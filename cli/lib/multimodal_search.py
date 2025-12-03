from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies
from lib.semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(
        self,
        documents: list[dict] = [],
        model_name="clip-ViT-B-32",
    ):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        image_embedding = self.model.encode([image])[0]
        return image_embedding

    def search_with_image(
        self, image_path: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        similarities: list[dict] = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, text_embedding)
            similarities.append({**self.documents[i], "similarity": similarity})

        results = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[
            :limit
        ]
        return results


def verify_image_embedding(image_path: str):
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image_path: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    searcher = MultimodalSearch(documents=movies)

    search_results = searcher.search_with_image(image_path, limit=limit)

    return {
        "image_path": image_path,
        "search_results": search_results,
    }
