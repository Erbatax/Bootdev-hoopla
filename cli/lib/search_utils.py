import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

MOVIES_PATH = os.path.join(DATA_DIR, "movies.json")
STOP_WORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stop_words() -> list[str]:
    with open(STOP_WORDS_PATH, "r") as f:
        data = f.read()
    return data.splitlines()
