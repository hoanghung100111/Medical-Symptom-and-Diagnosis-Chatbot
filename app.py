import sys
import io
import json
import faiss
import unicodedata
import logging
import re
import math
import time
import numpy as np
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from groq import Groq
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import config

# ── Fix Unicode logging trên Windows ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(stream=io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        ))
    ]
)
logger = logging.getLogger(__name__)

# ── Text helpers ─────────────────────────────────────────────
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")

def _tokenize(text: str) -> list:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r'\w+', text)

# ── Flask ────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY

# ── Model & Index ────────────────────────────────────────────
logger.info("Loading models and index...")
client   = Groq(api_key=config.GROQ_API_KEY)
embedder = SentenceTransformer(config.EMBED_MODEL)
index    = faiss.read_index(config.FAISS_INDEX_PATH)

with open(config.KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
    full_data = json.load(f)

id2doc        = {i: doc for i, doc in enumerate(full_data)}
all_doc_names = [normalize_text(d.get("main_name", "")) for d in full_data]
_N            = len(full_data)
logger.info(f"Loaded {_N} documents.")

# ── BM25 ─────────────────────────────────────────────────────
_corpus_tokens = [
    _tokenize(d.get("knowledge_text", "") + " " + d.get("main_name", ""))
    for d in full_data
]
_df = defaultdict(int)
for tokens in _corpus_tokens:
    for t in set(tokens):
        _df[t] += 1

_avgdl  = sum(len(t) for t in _corpus_tokens) / max(_N, 1)
_k1, _b = 1.5, 0.75

def bm25_scores(query_tokens: list) -> np.ndarray:
    scores = np.zeros(_N)
    for term in query_tokens:
        if term not in _df:
            continue
        idf = math.log((_N - _df[term] + 0.5) / (_df[term] + 0.5) + 1)
        for i, tokens in enumerate(_corpus_tokens):
            tf  = tokens.count(term)
            dl  = len(tokens)
            scores[i] += idf * (tf * (_k1 + 1)) / (tf + _k1 * (1 - _b + _b * dl / _avgdl))
    return scores

logger.info("BM25 index built.")

# ════════════════════════════════════════════════════════════
# SEMANTIC CACHE
# ════════════════════════════════════════════════════════════
# Lưu (query_embedding, answer) — cosine sim >= threshold thì cache hit
_CACHE_EMBEDDINGS: list = []   # list of np.ndarray (normalized)
_CACHE_ANSWERS:    list = []   # list of str
_CACHE_THRESHOLD   = 0.92      # cao để chỉ hit khi query thực sự giống nhau

def _cache_lookup(query: str) -> str | None:
    if not _CACHE_EMBEDDINGS:
        return None
    q_vec = embedder.encode([query], convert_to_numpy=True)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    mat   = np.vstack(_CACHE_EMBEDDINGS)
    sims  = (mat @ q_vec.T).flatten()
    best  = int(np.argmax(sims))
    if sims[best] >= _CACHE_THRESHOLD:
        logger.info(f"Cache HIT (sim={sims[best]:.3f}): {query!r}")
        return _CACHE_ANSWERS[best]
    return None

def _cache_store(query: str, answer: str) -> None:
    q_vec = embedder.encode([query], convert_to_numpy=True)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    _CACHE_EMBEDDINGS.append(q_vec)
    _CACHE_ANSWERS.append(answer)

# ════════════════════════════════════════════════════════════
# TẦNG 1 — INTENT CLASSIFICATION (keyword + embedding)
# ════════════════════════════════════════════════════════════
_INTENT_KEYWORDS = {
    "symptom": [
        "triệu chứng", "dấu hiệu", "biểu hiện", "nhận biết", "nhận ra",
        "phát hiện", "bị gì", "có gì", "bị đau", "bị ho", "bị sốt",
        "bị mệt", "bị ngứa", "bị nổi", "bị chảy", "bị khó", "bị tê",
        "bị sưng", "cảm thấy", "hay bị", "mãi không khỏi", "liên tục",
        "kéo dài", "không hết",
        "symptom", "sign", "feel", "feeling", "hurt", "pain",
    ],
    "treatment": [
        "điều trị", "chữa", "chữa trị", "chữa khỏi", "khỏi bệnh",
        "thuốc", "uống thuốc", "dùng thuốc", "phác đồ", "trị liệu",
        "phẫu thuật", "mổ", "xử lý", "xử trí", "làm gì",
        "phải làm sao", "làm thế nào", "khắc phục", "hồi phục",
        "treatment", "treat", "cure", "medicine", "medication",
        "drug", "therapy", "surgery", "how to treat", "what to do",
    ],
    "prevention": [
        "phòng ngừa", "phòng tránh", "ngăn ngừa", "ngăn chặn",
        "tránh", "tránh được không", "phòng bệnh", "bảo vệ",
        "giảm nguy cơ", "để không bị", "cách phòng",
        "prevent", "prevention", "avoid", "protect", "reduce risk",
    ],
    "cause": [
        "nguyên nhân", "do đâu", "do gì", "tại sao", "vì sao",
        "lý do", "gây ra", "gây nên", "dẫn đến", "ai dễ bị",
        "yếu tố nguy cơ", "nguy cơ", "cơ chế", "bệnh sinh",
        "cause", "reason", "why", "what causes", "risk factor",
    ],
}

_INTENT_EXAMPLES = {
    "symptom": [
        "Triệu chứng của bệnh này là gì?", "Bệnh này biểu hiện như thế nào?",
        "Làm sao nhận biết được bệnh này?", "Người bệnh thường có những dấu hiệu gì?",
        "Bị đau đầu liên tục phải làm sao?", "Ho mãi không khỏi là bệnh gì?",
        "What are the symptoms of this disease?", "What signs should I look for?",
    ],
    "treatment": [
        "Bệnh này điều trị như thế nào?", "Có thuốc gì chữa bệnh này không?",
        "Làm gì khi bị mắc bệnh này?", "Bệnh này có chữa khỏi được không?",
        "How is this disease treated?", "What medication is used for this condition?",
    ],
    "prevention": [
        "Làm thế nào để phòng ngừa bệnh này?", "Có thể tránh được bệnh này không?",
        "Làm sao để không bị mắc bệnh?", "How can I prevent this disease?",
        "What are the best ways to avoid getting sick?",
    ],
    "cause": [
        "Nguyên nhân gây ra bệnh này là gì?", "Bệnh này do đâu mà có?",
        "Tại sao lại bị bệnh này?", "Yếu tố nguy cơ của bệnh này là gì?",
        "What causes this disease?", "What are the risk factors?",
    ],
    "general": [
        "Bệnh này là gì?", "Cho tôi biết về bệnh này.",
        "Bệnh này có nguy hiểm không?", "Tell me about this disease.",
    ],
}

logger.info("Building intent embedding index...")
_intent_labels, _intent_sentences = [], []
for intent, examples in _INTENT_EXAMPLES.items():
    for ex in examples:
        _intent_labels.append(intent)
        _intent_sentences.append(ex)

_intent_embs = embedder.encode(_intent_sentences, convert_to_numpy=True)
_intent_embs = _intent_embs / (np.linalg.norm(_intent_embs, axis=1, keepdims=True) + 1e-9)
logger.info(f"Intent index: {len(_intent_sentences)} examples.")


def classify_intent(query: str) -> str:
    q = query.lower()
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in kws):
            logger.info(f"Intent (keyword): {intent}")
            return intent

    q_vec  = embedder.encode([query], convert_to_numpy=True)
    q_vec  = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    sims   = (_intent_embs @ q_vec.T).flatten()
    scores = {}
    for label, sim in zip(_intent_labels, sims):
        scores.setdefault(label, []).append(float(sim))
    avg    = {k: np.mean(v) for k, v in scores.items()}
    best   = max(avg, key=avg.__getitem__)
    logger.info(f"Intent (embedding): {best} {avg}")
    return best if avg[best] >= 0.45 else "general"


# ════════════════════════════════════════════════════════════
# TẦNG 2 — FUZZY LOOKUP (không dùng LLM)
# ════════════════════════════════════════════════════════════
def fuzzy_lookup(query: str, kb: list) -> dict | None:
    """
    Tìm tên bệnh trực tiếp từ query bằng fuzzy match.
    Không gọi LLM — zero API call.
    """
    q_norm = normalize_text(query)
    best_doc, best_score = None, 0
    for doc in kb:
        candidates = [doc.get("main_name", "")] + doc.get("synonyms", [])
        for cand in candidates:
            cand_norm = normalize_text(cand)
            if len(cand_norm) < 3:
                continue
            score     = fuzz.token_set_ratio(q_norm, cand_norm)
            threshold = 92 if len(cand_norm) <= 5 else config.FUZZY_THRESHOLD
            if score > best_score and score >= threshold:
                best_score, best_doc = score, doc
    if best_doc:
        logger.info(f"Fuzzy: {query!r} → {best_doc.get('main_name')} (score={best_score})")
    return best_doc


# ════════════════════════════════════════════════════════════
# TẦNG 3 — HYBRID RETRIEVAL (FAISS + BM25)
# ════════════════════════════════════════════════════════════
def hybrid_retrieve(query: str, top_k: int = None, alpha: float = 0.6) -> list:
    top_k        = top_k or config.TOP_K
    q_vec        = embedder.encode([query])
    D, I         = index.search(q_vec, min(top_k * 3, _N))
    faiss_scores = np.zeros(_N)
    for score, idx in zip(D[0], I[0]):
        if idx >= 0:
            faiss_scores[idx] = 1 / (1 + score)

    bm25      = bm25_scores(_tokenize(query))
    bm25_max  = bm25.max()
    bm25_norm = bm25 / bm25_max if bm25_max > 0 else bm25

    combined    = alpha * faiss_scores + (1 - alpha) * bm25_norm
    top_indices = np.argsort(combined)[::-1][:top_k]
    results     = [id2doc[i] for i in top_indices if combined[i] > 0.01 and i in id2doc]
    logger.info(f"Hybrid retrieve: {[d.get('main_name') for d in results]}")
    return results


# ════════════════════════════════════════════════════════════
# TẦNG 4 — SINGLE LLM CALL với intent-aware prompt
# (entity extraction + answering gộp lại 1 call duy nhất)
# ════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are MediChat, a professional medical information assistant.

STRICT RULES:
1. GROUNDING: Answer ONLY based on the provided context. Do NOT add information not present.
2. UNCERTAINTY: If context is insufficient, say: "Tôi không tìm thấy đủ thông tin. Vui lòng tham khảo bác sĩ."
3. NO HALLUCINATION: Never invent drug names, dosages, or procedures not in the context.
4. DISCLAIMER: Always end treatment/diagnosis responses with a medical disclaimer.
5. TONE: Professional, clear, empathetic.
6. SCOPE: Health information only — not diagnosis or prescription.
"""

LANG_MAP = {
    "Vietnamese": "Respond ONLY in Vietnamese.",
    "English":    "Respond ONLY in English.",
    "Auto":       "Detect the language of the question and respond in the SAME language.",
}

PROMPT_TEMPLATES = {
    "symptom": """{lang_instruction}

The user is asking about SYMPTOMS of a disease.

CONTEXT:
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

Answer structure:
**Triệu chứng thường gặp:**
- [List symptoms from context only, one per line]

**Dấu hiệu cảnh báo nguy hiểm:**
- [Red-flag symptoms if mentioned in context]

**Khi nào cần gặp bác sĩ:**
[1-2 sentences. Omit sections not in context. Do NOT fabricate.]""",

    "treatment": """{lang_instruction}

The user is asking about TREATMENT of a disease.

CONTEXT:
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

Answer structure:
**Phương pháp điều trị:**
[Treatment from context only]

**Thuốc thường dùng:**
[Medications only if in context. Otherwise: "Việc dùng thuốc cần được bác sĩ chỉ định."]

**Lưu ý quan trọng:**
- Phác đồ điều trị cần bác sĩ chỉ định dựa trên tình trạng từng bệnh nhân.
- Không tự ý dùng thuốc mà không có hướng dẫn của bác sĩ.""",

    "prevention": """{lang_instruction}

The user is asking about PREVENTION of a disease.

CONTEXT:
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

Answer structure:
**Biện pháp phòng ngừa:**
[Prevention from context, grouped: lifestyle / medical / environmental]

**Đối tượng nguy cơ cao:**
[Risk groups if in context. If no prevention info: "Vui lòng tham khảo bác sĩ."]""",

    "cause": """{lang_instruction}

The user is asking about CAUSES of a disease.

CONTEXT:
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

Answer structure:
**Nguyên nhân chính:**
[Causes from context only]

**Yếu tố nguy cơ:**
[Risk factors if mentioned]

**Cơ chế bệnh sinh:**
[Pathophysiology if in context. Skip if not mentioned. Do NOT add causes not in context.]""",

    "general": """{lang_instruction}

The user has a general medical question.

CONTEXT:
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

Answer comprehensively from context only. Use clear headings if needed.
End with: "⚕️ Lưu ý: Thông tin chỉ mang tính tham khảo. Vui lòng tham khảo bác sĩ hoặc chuyên gia y tế." """,
}


def prepare_context(docs: list) -> str:
    """Truncate trực tiếp — không gọi LLM để summarize."""
    context_text = "\n\n".join(
        f"[{d.get('main_name','')}]\n{d.get('knowledge_text','')}" for d in docs
    )
    return context_text[:config.CONTEXT_CHAR_LIMIT]


def build_prompt(query: str, context: str, lang_mode: str, intent: str) -> tuple[str, str]:
    """Trả về (system_prompt, user_prompt)."""
    template         = PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES["general"])
    lang_instruction = LANG_MAP.get(lang_mode, LANG_MAP["Auto"])
    user_prompt      = template.format(
        lang_instruction=lang_instruction,
        context=context,
        query=query,
    )
    return SYSTEM_PROMPT, user_prompt


# ════════════════════════════════════════════════════════════
# STREAMING
# ════════════════════════════════════════════════════════════
def stream_answer(system_prompt: str, user_prompt: str):
    """
    Generator: yield từng token từ Groq streaming API.
    Format: Server-Sent Events (SSE) để frontend nhận dễ.
    """
    full_text = []
    try:
        stream = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=config.MAX_TOKENS_ANSWER,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text.append(delta)
                yield f"data: {json.dumps({'token': delta})}\n\n"

        # Lưu cache sau khi stream xong
        yield f"data: {json.dumps({'done': True})}\n\n"
        return "".join(full_text)

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def ask_llama_sync(system_prompt: str, user_prompt: str) -> str:
    """Non-streaming call — dùng cho cache store sau khi stream xong."""
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=config.MAX_TOKENS_ANSWER,
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════
STOPWORDS = {
    "trieu chung", "dau hieu", "benh", "thuoc",
    "cach chua", "dieu tri", "phong ngua"
}

def process_query_stream(user_query: str, lang_mode: str = "Auto"):
    """
    Generator pipeline — stream tokens về frontend.
    Query → Cache? → Intent → Fuzzy → Hybrid Retrieve → 1 LLM call (stream)
    """
    t0 = time.time()

    # 0. Semantic cache check
    cached = _cache_lookup(user_query)
    if cached:
        logger.info(f"Cache hit in {time.time()-t0:.3f}s")
        yield f"data: {json.dumps({'token': cached})}\n\n"
        yield f"data: {json.dumps({'done': True, 'cached': True})}\n\n"
        return

    # 1. Intent (0 API call)
    intent = classify_intent(user_query)

    # 2. Fuzzy lookup (0 API call) — thay thế extract_entity_llm
    doc = fuzzy_lookup(user_query, full_data)

    if doc:
        related  = hybrid_retrieve(user_query, top_k=2)
        all_docs = [doc] + [d for d in related if d.get("main_name") != doc.get("main_name")]
        context  = prepare_context(all_docs[:3])
    else:
        # 3. Hybrid retrieval fallback
        suggestions = hybrid_retrieve(user_query, top_k=config.TOP_K)

        if not suggestions:
            yield f"data: {json.dumps({'token': 'Xin lỗi, không tìm thấy thông tin phù hợp.'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # Nếu nhiều kết quả không rõ → trả suggestions (không stream)
        if len(suggestions) > 1 and not doc:
            sug_names = [d.get("main_name", "") for d in suggestions]
            yield f"data: {json.dumps({'suggestions': sug_names})}\n\n"
            return

        context = prepare_context(suggestions)

    # 4. Single LLM call — stream
    system_prompt, user_prompt = build_prompt(user_query, context, lang_mode, intent)
    logger.info(f"LLM call | intent={intent} | {time.time()-t0:.3f}s elapsed")

    full_text = []
    try:
        stream = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=config.MAX_TOKENS_ANSWER,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_text.append(delta)
                yield f"data: {json.dumps({'token': delta})}\n\n"

        answer = "".join(full_text)
        logger.info(f"Done in {time.time()-t0:.3f}s | {len(answer)} chars")

        # Lưu vào semantic cache
        _cache_store(user_query, answer)
        yield f"data: {json.dumps({'done': True, 'cached': False})}\n\n"

    except Exception as e:
        logger.error(f"LLM error: {e}")
        err_msg = "Có lỗi xảy ra. Vui lòng thử lại."
        if "rate_limit" in str(e).lower() or "429" in str(e):
            err_msg = "Hệ thống đang bận, vui lòng thử lại sau vài giây."
        yield f"data: {json.dumps({'error': err_msg})}\n\n"


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

    q_norm = normalize_text(user_query)
    if any(q_norm == sw or q_norm.strip() == sw for sw in STOPWORDS):
        return jsonify({
            "type": "answer",
            "text": "Câu hỏi chưa đủ cụ thể. Vui lòng nhập tên bệnh bạn muốn tra cứu."
        })

    return Response(
        stream_with_context(process_query_stream(user_query, lang_mode)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/disease/<name>")
def disease_detail(name):
    lang_mode = request.args.get("lang", "Auto")
    return Response(
        stream_with_context(process_query_stream(name, lang_mode)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        debug=(config.FLASK_ENV == "development"),
        port=config.FLASK_PORT,
        threaded=True,
    )