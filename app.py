import sys
import json
import faiss
import unicodedata
import logging
from flask import Flask, render_template, request, jsonify
from groq import Groq
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import config

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Flask ────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY

# ── Model & Index setup ──────────────────────────────────────
logger.info("Loading models and index...")
client   = Groq(api_key=config.GROQ_API_KEY)
embedder = SentenceTransformer(config.EMBED_MODEL)
index    = faiss.read_index(config.FAISS_INDEX_PATH)

with open(config.KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    full_data = json.load(f)

id2doc = {i: doc for i, doc in enumerate(full_data)}
logger.info(f"Loaded {len(full_data)} documents.")

# ── Stopwords ────────────────────────────────────────────────
STOPWORDS = {
    "trieu chung", "dau hieu", "benh", "thuoc",
    "cach chua", "dieu tri", "phong ngua"
}

# ── Helpers ──────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def is_query_too_generic(query: str) -> bool:
    q_norm = normalize_text(query)
    return any(q_norm == sw or q_norm.strip() in sw for sw in STOPWORDS)


def extract_disease(query: str, kb):
    q_norm = normalize_text(query)
    best_doc, best_score = None, 0
    for doc in kb:
        candidates = [doc.get("main_name", "")] + doc.get("synonyms", [])
        for cand in candidates:
            score = fuzz.partial_ratio(q_norm, normalize_text(cand))
            if score > best_score:
                best_score, best_doc = score, doc
    return best_doc if best_score >= config.FUZZY_THRESHOLD else None


def retrieve_context(query, top_k=None, threshold=None):
    top_k     = top_k     or config.TOP_K
    threshold = threshold or config.SIMILARITY_THRESHOLD
    query_vec = embedder.encode([query])
    D, I      = index.search(query_vec, top_k)
    return [id2doc[idx] for score, idx in zip(D[0], I[0])
            if idx in id2doc and score >= threshold]


def summarize_context(context: str) -> str:
    prompt = f"Hãy tóm tắt ngắn gọn, giữ lại ý chính, dễ hiểu:\n\n{context}"
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.MAX_TOKENS_SUMMARY,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def prepare_context(docs) -> str:
    context_text = "\n\n".join(
        f"{d.get('main_name','')}\n{d.get('knowledge_text','')}" for d in docs
    )
    if len(context_text) <= config.CONTEXT_CHAR_LIMIT:
        return context_text
    chunks    = [context_text[i:i+config.CONTEXT_CHAR_LIMIT]
                 for i in range(0, len(context_text), config.CONTEXT_CHAR_LIMIT)]
    summaries = [summarize_context(chunk) for chunk in chunks]
    return "\n\n".join(summaries)


def ask_llama(query: str, context: str, lang_mode: str = "Auto") -> str:
    if not context:
        return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp."

    lang_map = {
        "Vietnamese": "Always answer ONLY in Vietnamese.",
        "English":    "Always answer ONLY in English.",
        "Auto":       ("If the question is in Vietnamese, answer ONLY in Vietnamese. "
                       "If the question is in English, answer ONLY in English."),
    }
    lang_instruction = lang_map.get(lang_mode, lang_map["Auto"])

    prompt = f"""You are a helpful bilingual medical assistant.
{lang_instruction}
Use the following medical context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.MAX_TOKENS_ANSWER,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ── Routes ───────────────────────────────────────────────────
@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    user_query = data.get("message", "").strip()
    lang_mode  = data.get("lang_mode", "Auto")

    if not user_query:
        return jsonify({"type": "error", "text": "Vui lòng nhập câu hỏi."})

    logger.info(f"Query: {user_query!r} | lang={lang_mode}")

    if is_query_too_generic(user_query):
        return jsonify({
            "type": "answer",
            "text": "Câu hỏi chưa đủ cụ thể. Vui lòng nhập tên bệnh bạn muốn tra cứu."
        })

    doc = extract_disease(user_query, full_data)
    if doc:
        context = prepare_context([doc])
        answer  = ask_llama(user_query, context, lang_mode)
        logger.info(f"Direct match: {doc.get('main_name')}")
        return jsonify({"type": "answer", "text": answer})

    suggestions = retrieve_context(user_query)
    if not suggestions:
        return jsonify({
            "type": "answer",
            "text": "Xin lỗi, tôi chưa tìm thấy thông tin về bệnh này trong dữ liệu."
        })

    sug_names = [d.get("main_name", "Không rõ") for d in suggestions]
    logger.info(f"Suggestions: {sug_names}")
    return jsonify({
        "type": "suggestions",
        "text": "Mình chưa tìm thấy chính xác. Bạn có muốn nói đến một trong các bệnh sau không?",
        "suggestions": sug_names
    })


@app.route("/disease/<name>")
def disease_detail(name):
    lang_mode = request.args.get("lang", "Auto")
    doc = extract_disease(name, full_data)
    if doc:
        context = prepare_context([doc])
        return jsonify({"type": "answer", "text": ask_llama(name, context, lang_mode)})
    return jsonify({"type": "error", "text": "Không tìm thấy thông tin."})


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        debug=(config.FLASK_ENV == "development"),
        port=config.FLASK_PORT
    )