import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq / LLM ──────────────────────────────────────────────
GROQ_API_KEY        = os.environ["GROQ_API_KEY"]
GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Embedding ────────────────────────────────────────────────
EMBED_MODEL         = os.getenv(
    "EMBED_MODEL",
    "intfloat/multilingual-e5-base"
)

# ── Data paths ───────────────────────────────────────────────
FAISS_INDEX_PATH    = os.getenv("FAISS_INDEX_PATH", "medical_index.faiss")
KNOWLEDGE_PATH      = os.getenv("KNOWLEDGE_PATH",   "chatbot_knowledge_chunked.json")

# ── RAG params ───────────────────────────────────────────────
TOP_K               = int(os.getenv("TOP_K",                "3"))
SIMILARITY_THRESHOLD= float(os.getenv("SIMILARITY_THRESHOLD","0.55"))
CONTEXT_CHAR_LIMIT  = int(os.getenv("CONTEXT_CHAR_LIMIT",   "3000"))
MAX_TOKENS_ANSWER   = int(os.getenv("MAX_TOKENS_ANSWER",    "600"))
MAX_TOKENS_SUMMARY  = int(os.getenv("MAX_TOKENS_SUMMARY",   "400"))

# ── Fuzzy match ──────────────────────────────────────────────
FUZZY_THRESHOLD     = int(os.getenv("FUZZY_THRESHOLD",      "70"))

# ── Flask ────────────────────────────────────────────────────
FLASK_SECRET_KEY    = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
FLASK_ENV           = os.getenv("FLASK_ENV",         "development")
FLASK_PORT          = int(os.getenv("FLASK_PORT",    "5000"))