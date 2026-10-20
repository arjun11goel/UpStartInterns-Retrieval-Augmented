"""Central configuration."""
import os
from dotenv import load_dotenv

load_dotenv()
APP_TITLE = "LocalDoc QA"
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))
# FAISS squared L2 distance; smaller is closer.
MAX_L2_DISTANCE = float(os.getenv("MAX_L2_DISTANCE", "1.25"))
