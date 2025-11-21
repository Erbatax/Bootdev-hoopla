import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
llm_client = genai.Client(api_key=api_key)
DEFAULT_LLM_MODEL = "gemini-2.0-flash"
