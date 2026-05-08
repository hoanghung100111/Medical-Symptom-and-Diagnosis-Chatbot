import sys
import io
import json
import faiss
import unicodedata
import logging
import re
import numpy as np
from flask import Flask, render_template, request, jsonify
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

id2doc       = {i: doc for i, doc in enumerate(full_data)}
all_doc_names = [normalize_text(d.get("main_name", "")) for d in full_data]
logger.info(f"Loaded {len(full_data)} documents.")

# ── BM25 lightweight (không cần thư viện ngoài) ──────────────
# Precompute inverted index cho hybrid retrieval
from collections import defaultdict, Counter
import math

def _tokenize(text: str):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r'\w+', text)

# Build BM25 index
_corpus_tokens = [_tokenize(d.get("knowledge_text", "") + " " + d.get("main_name", ""))
                  for d in full_data]
_df = defaultdict(int)
for tokens in _corpus_tokens:
    for t in set(tokens):
        _df[t] += 1

_N      = len(full_data)
_avgdl  = sum(len(t) for t in _corpus_tokens) / max(_N, 1)
_k1, _b = 1.5, 0.75

def bm25_scores(query_tokens: list[str]) -> np.ndarray:
    scores = np.zeros(_N)
    for term in query_tokens:
        if term not in _df:
            continue
        idf = math.log((_N - _df[term] + 0.5) / (_df[term] + 0.5) + 1)
        for i, tokens in enumerate(_corpus_tokens):
            tf  = tokens.count(term)
            dl  = len(tokens)
            num = tf * (_k1 + 1)
            den = tf + _k1 * (1 - _b + _b * dl / _avgdl)
            scores[i] += idf * num / den
    return scores

logger.info("BM25 index built.")

# ════════════════════════════════════════════════════════════
# TẦNG 1 — INTENT CLASSIFICATION (Hybrid: keyword + embedding)
# ════════════════════════════════════════════════════════════

# ── Keyword rules (fast path) ────────────────────────────────────
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "symptom": [
        "triệu chứng", "dấu hiệu", "biểu hiện", "nhận biết",
        "nhận ra", "phát hiện", "bị gì", "có gì",
        "bị đau", "bị ho", "bị sốt", "bị mệt", "bị ngứa",
        "bị nổi", "bị chảy", "bị khó", "bị tê", "bị sưng",
        "đau như thế nào", "cảm thấy", "hay bị", "mãi không khỏi",
        "liên tục", "kéo dài", "không hết",
        "symptom", "sign", "feel", "feeling", "hurt", "pain",
        "suffering", "what does it feel",
    ],
    "treatment": [
        "điều trị", "chữa", "chữa trị", "chữa khỏi", "khỏi bệnh",
        "thuốc", "uống thuốc", "dùng thuốc", "phác đồ", "trị liệu",
        "can thiệp", "phẫu thuật", "mổ", "xử lý", "xử trí",
        "làm gì", "phải làm sao", "làm thế nào", "giải quyết",
        "khắc phục", "cải thiện", "hồi phục",
        "treatment", "treat", "cure", "medicine", "medication",
        "drug", "therapy", "surgery", "how to treat", "what to do",
        "manage", "remedy",
    ],
    "prevention": [
        "phòng ngừa", "phòng tránh", "ngăn ngừa", "ngăn chặn",
        "tránh", "tránh được không", "có thể tránh", "phòng bệnh",
        "bảo vệ", "giảm nguy cơ", "không bị", "để không bị",
        "làm sao để không", "cách phòng",
        "prevent", "prevention", "avoid", "protect", "reduce risk",
        "how to avoid", "how to prevent",
    ],
    "cause": [
        "nguyên nhân", "do đâu", "do gì", "tại sao", "vì sao",
        "lý do", "nguyên do", "gây ra", "gây nên", "dẫn đến",
        "ai dễ bị", "đối tượng nào", "yếu tố nguy cơ",
        "nguy cơ", "cơ chế", "bệnh sinh",
        "cause", "reason", "why", "what causes", "risk factor",
        "who gets", "mechanism", "pathogen",
    ],
}

def _keyword_classify(query: str) -> str | None:
    """Fast path: keyword match. Returns intent or None."""
    q = query.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return None


# ── Embedding similarity (slow path) ────────────────────────────────────
_INTENT_EXAMPLES: dict[str, list[str]] = {
    "symptom": [
        "Triệu chứng của bệnh này là gì?",
        "Bệnh này biểu hiện như thế nào?",
        "Làm sao nhận biết được bệnh này?",
        "Người bệnh thường có những dấu hiệu gì?",
        "Bệnh này gây ra cảm giác như thế nào?",
        "Bị đau đầu liên tục phải làm sao?",
        "Ho mãi không khỏi là bệnh gì?",
        "What are the symptoms of this disease?",
        "How does this condition present itself?",
        "What signs should I look for?",
    ],
    "treatment": [
        "Bệnh này điều trị như thế nào?",
        "Có thuốc gì chữa bệnh này không?",
        "Phác đồ điều trị của bệnh này là gì?",
        "Làm gì khi bị mắc bệnh này?",
        "Bệnh này có chữa khỏi được không?",
        "Phải xử trí thế nào?",
        "How is this disease treated?",
        "What medication is used for this condition?",
        "What should I do if I have this disease?",
    ],
    "prevention": [
        "Làm thế nào để phòng ngừa bệnh này?",
        "Có thể tránh được bệnh này không?",
        "Cách bảo vệ bản thân khỏi bệnh này?",
        "Làm sao để không bị mắc bệnh?",
        "Biện pháp phòng bệnh hiệu quả nhất là gì?",
        "How can I prevent this disease?",
        "What are the best ways to avoid getting sick?",
        "How to protect myself from this condition?",
    ],
    "cause": [
        "Nguyên nhân gây ra bệnh này là gì?",
        "Bệnh này do đâu mà có?",
        "Tại sao lại bị bệnh này?",
        "Yếu tố nguy cơ của bệnh này là gì?",
        "Ai dễ bị mắc bệnh này nhất?",
        "Cơ chế gây bệnh là gì?",
        "What causes this disease?",
        "Why do people get this condition?",
        "What are the risk factors?",
    ],
    "general": [
        "Bệnh này là gì?",
        "Cho tôi biết về bệnh này.",
        "Thông tin về bệnh này.",
        "Bệnh này có nguy hiểm không?",
        "Tell me about this disease.",
        "What is this condition?",
        "Give me information about this disease.",
    ],
}

logger.info("Building intent embedding index...")
_intent_labels: list[str] = []
_intent_sentences: list[str] = []
for intent, examples in _INTENT_EXAMPLES.items():
    for ex in examples:
        _intent_labels.append(intent)
        _intent_sentences.append(ex)

_intent_embeddings = embedder.encode(_intent_sentences, convert_to_numpy=True)
_intent_embeddings = _intent_embeddings / (
    np.linalg.norm(_intent_embeddings, axis=1, keepdims=True) + 1e-9
)
logger.info(f"Intent embedding index built: {len(_intent_sentences)} examples.")


def _embedding_classify(query: str, threshold: float = 0.45) -> str:
    """Slow path: cosine similarity vs example sentences."""
    q_vec  = embedder.encode([query], convert_to_numpy=True)
    q_vec  = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)
    scores = (_intent_embeddings @ q_vec.T).flatten()

    intent_scores: dict[str, list[float]] = {k: [] for k in _INTENT_EXAMPLES}
    for label, score in zip(_intent_labels, scores):
        intent_scores[label].append(float(score))

    avg_scores  = {k: np.mean(v) for k, v in intent_scores.items()}
    best_intent = max(avg_scores, key=avg_scores.__getitem__)
    best_score  = avg_scores[best_intent]

    logger.info(f"Embedding intent scores: { {k: round(v,3) for k,v in avg_scores.items()} }")
    return best_intent if best_score >= threshold else "general"


def classify_intent(query: str) -> str:
    """
    Hybrid:
    1. Keyword match (fast, ~0ms)
    2. Embedding similarity if keyword miss (~20ms, no API call)
    """
    intent = _keyword_classify(query)
    if intent:
        logger.info(f"Intent (keyword): {intent}")
        return intent
    intent = _embedding_classify(query)
    logger.info(f"Intent (embedding): {intent}")
    return intent

# ════════════════════════════════════════════════════════════
# TẦNG 2 — ENTITY EXTRACTION (LLM-based)
# ════════════════════════════════════════════════════════════
_ENTITY_CACHE: dict[str, str | None] = {}

def extract_entity_llm(query: str) -> str | None:
    """Dùng LLM để extract tên bệnh ra khỏi câu hỏi."""
    if query in _ENTITY_CACHE:
        return _ENTITY_CACHE[query]

    prompt = f"""Extract the disease or medical condition name from this query.
Return ONLY the disease name in Vietnamese, nothing else.
If no disease is mentioned, return: NONE

Query: {query}
Disease name:"""

    try:
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        result = resp.choices[0].message.content.strip()
        entity = None if result.upper() == "NONE" or not result else result
        _ENTITY_CACHE[query] = entity
        logger.info(f"Entity extracted: {entity!r} from {query!r}")
        return entity
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return None


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def fuzzy_lookup(entity: str, kb) -> dict | None:
    """Lookup chính xác tên bệnh đã được extract."""
    if not entity:
        return None
    e_norm = normalize_text(entity)
    best_doc, best_score = None, 0
    for doc in kb:
        candidates = [doc.get("main_name", "")] + doc.get("synonyms", [])
        for cand in candidates:
            cand_norm = normalize_text(cand)
            if len(cand_norm) < 3:
                continue
            # Dùng token_set_ratio để robust hơn với word order
            score = fuzz.token_set_ratio(e_norm, cand_norm)
            threshold = 92 if len(cand_norm) <= 5 else config.FUZZY_THRESHOLD
            if score > best_score and score >= threshold:
                best_score, best_doc = score, doc
    logger.info(f"Fuzzy lookup: {entity!r} → {best_doc.get('main_name') if best_doc else None} (score={best_score})")
    return best_doc


# ════════════════════════════════════════════════════════════
# TẦNG 3 — HYBRID RETRIEVAL (FAISS + BM25)
# ════════════════════════════════════════════════════════════
def hybrid_retrieve(query: str, top_k: int = None, alpha: float = 0.6) -> list[dict]:
    """
    Hybrid retrieval: alpha * FAISS_score + (1-alpha) * BM25_score
    alpha=0.6 → nghiêng về semantic; tăng lên nếu muốn keyword match mạnh hơn
    """
    top_k = top_k or config.TOP_K

    # FAISS semantic scores
    query_vec      = embedder.encode([query])
    faiss_D, faiss_I = index.search(query_vec, min(top_k * 3, _N))
    faiss_scores   = np.zeros(_N)
    for score, idx in zip(faiss_D[0], faiss_I[0]):
        if idx >= 0:
            # Normalize FAISS score (L2 distance → similarity)
            faiss_scores[idx] = 1 / (1 + score)

    # BM25 keyword scores
    query_tokens  = _tokenize(query)
    bm25          = bm25_scores(query_tokens)
    bm25_max      = bm25.max()
    bm25_norm     = bm25 / bm25_max if bm25_max > 0 else bm25

    # Combine
    combined      = alpha * faiss_scores + (1 - alpha) * bm25_norm
    top_indices   = np.argsort(combined)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if combined[idx] > 0.01 and idx in id2doc:
            results.append(id2doc[idx])

    logger.info(f"Hybrid retrieve: {[d.get('main_name') for d in results]}")
    return results


# ════════════════════════════════════════════════════════════
# TẦNG 4 — INTENT-AWARE PROMPTS (nâng cấp)
# ════════════════════════════════════════════════════════════

# ── System prompt chung — áp dụng cho mọi intent ────────────
SYSTEM_PROMPT = """You are MediChat, a professional medical information assistant.

STRICT RULES you must always follow:
1. GROUNDING: Answer ONLY based on the provided context. Do NOT add information not present in the context.
2. UNCERTAINTY: If the context does not contain enough information to answer, say explicitly:
   "Tôi không tìm thấy đủ thông tin về vấn đề này trong cơ sở dữ liệu. Vui lòng tham khảo bác sĩ."
3. NO HALLUCINATION: Never invent drug names, dosages, statistics, or medical procedures not mentioned in the context.
4. DISCLAIMER: Always end responses about treatment or diagnosis with a medical disclaimer.
5. TONE: Professional, clear, and empathetic. Avoid overly technical jargon unless necessary.
6. SCOPE: You provide health information only, not diagnosis or prescription.
"""

LANG_MAP = {
    "Vietnamese": "Respond ONLY in Vietnamese.",
    "English":    "Respond ONLY in English.",
    "Auto":       "Detect the language of the question and respond in the SAME language.",
}

# ── User prompt templates theo intent ───────────────────────
PROMPT_TEMPLATES = {

    "symptom": """{lang_instruction}

The user is asking about SYMPTOMS of a disease.

CONTEXT (from medical knowledge base):
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

Answer following this EXACT structure:
**Triệu chứng thường gặp:**
- [List each main symptom clearly, one per line, based ONLY on context]

**Dấu hiệu cảnh báo nguy hiểm:**
- [List red-flag symptoms that require immediate medical attention, if mentioned in context]

**Khi nào cần gặp bác sĩ:**
[1-2 sentences on when to seek care]

If context does not mention a category above, omit that section entirely. Do NOT fabricate symptoms.""",

    "treatment": """{lang_instruction}

The user is asking about TREATMENT of a disease.

CONTEXT (from medical knowledge base):
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

Answer following this EXACT structure:
**Phương pháp điều trị:**
[Describe treatment approaches mentioned in context only]

**Thuốc thường dùng:**
[List medications only if explicitly mentioned in context. If not mentioned, write: "Việc dùng thuốc cần được bác sĩ chỉ định cụ thể."]

**Lưu ý quan trọng:**
- Phác đồ điều trị cần được bác sĩ hoặc chuyên gia y tế chỉ định dựa trên tình trạng cụ thể của từng bệnh nhân.
- Không tự ý dùng thuốc hoặc thay đổi liều lượng mà không có hướng dẫn của bác sĩ.""",

    "prevention": """{lang_instruction}

The user is asking about PREVENTION of a disease.

CONTEXT (from medical knowledge base):
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

Answer following this EXACT structure:
**Biện pháp phòng ngừa:**
[List prevention measures from context, grouped logically: lifestyle / medical / environmental]

**Đối tượng nguy cơ cao cần chú ý:**
[Mention risk groups if stated in context]

If context lacks prevention details, state: "Thông tin phòng ngừa chi tiết không có trong cơ sở dữ liệu. Vui lòng tham khảo bác sĩ." """,

    "cause": """{lang_instruction}

The user is asking about CAUSES or RISK FACTORS of a disease.

CONTEXT (from medical knowledge base):
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

Answer following this EXACT structure:
**Nguyên nhân chính:**
[List causes stated in context only]

**Yếu tố nguy cơ:**
[List risk factors if mentioned]

**Cơ chế bệnh sinh (nếu có):**
[Brief pathophysiology if context mentions it. Skip this section if not in context.]

Do NOT add causes not mentioned in the context.""",

    "general": """{lang_instruction}

The user has a general medical question.

CONTEXT (from medical knowledge base):
\"\"\"
{context}
\"\"\"

USER QUESTION: {query}

Instructions:
- Answer comprehensively but concisely based ONLY on the context provided.
- Use clear headings if the answer covers multiple aspects.
- If the context is insufficient to answer fully, acknowledge this explicitly.
- End your response with this disclaimer on a new line:
  "⚕️ Lưu ý: Thông tin trên chỉ mang tính tham khảo. Để được chẩn đoán và điều trị chính xác, vui lòng tham khảo ý kiến bác sĩ hoặc chuyên gia y tế." """,
}


def ask_llama(query: str, context: str, lang_mode: str = "Auto", intent: str = "general") -> str:
    if not context:
        return (
            "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu. "
            "Vui lòng thử lại với từ khóa khác hoặc tham khảo ý kiến bác sĩ."
        )

    template         = PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES["general"])
    lang_instruction = LANG_MAP.get(lang_mode, LANG_MAP["Auto"])
    user_prompt      = template.format(
        lang_instruction=lang_instruction,
        context=context[:config.CONTEXT_CHAR_LIMIT],
        query=query,
    )

    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=config.MAX_TOKENS_ANSWER,
        temperature=0.2,   # giảm từ 0.3 → 0.2 để ít hallucinate hơn
    )
    return resp.choices[0].message.content


# ── Context preparation ──────────────────────────────────────
def summarize_context(context: str) -> str:
    prompt = f"Tóm tắt ngắn gọn, giữ ý chính:\n\n{context}"
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.MAX_TOKENS_SUMMARY,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def prepare_context(docs: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[{d.get('main_name','')}]\n{d.get('knowledge_text','')}" for d in docs
    )
    if len(context_text) <= config.CONTEXT_CHAR_LIMIT:
        return context_text
    chunks    = [context_text[i:i+config.CONTEXT_CHAR_LIMIT]
                 for i in range(0, len(context_text), config.CONTEXT_CHAR_LIMIT)]
    summaries = [summarize_context(chunk) for chunk in chunks]
    return "\n\n".join(summaries)


# ════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════
def process_query(user_query: str, lang_mode: str = "Auto") -> dict:
    """
    Pipeline:
    Query → Intent → Entity Extract → Fuzzy Lookup → Hybrid Retrieve → Answer
    """
    # Tầng 1: Intent
    intent = classify_intent(user_query)
    logger.info(f"Intent: {intent}")

    # Tầng 2: Entity extraction
    entity = extract_entity_llm(user_query)

    # Tầng 2b: Fuzzy lookup nếu có entity
    doc = fuzzy_lookup(entity, full_data) if entity else None

    if doc:
        # Direct match — retrieve thêm related docs để context phong phú hơn
        related   = hybrid_retrieve(entity or user_query, top_k=2)
        all_docs  = [doc] + [d for d in related if d.get("main_name") != doc.get("main_name")]
        context   = prepare_context(all_docs[:3])
        answer    = ask_llama(user_query, context, lang_mode, intent)
        logger.info(f"Match: {doc.get('main_name')} | intent={intent}")
        return {"type": "answer", "text": answer}

    # Tầng 3: Hybrid retrieval fallback
    suggestions = hybrid_retrieve(user_query, top_k=config.TOP_K)
    if not suggestions:
        return {
            "type": "answer",
            "text": "Xin lỗi, tôi chưa tìm thấy thông tin về bệnh này trong dữ liệu."
        }

    # Nếu chỉ có 1 kết quả với score cao → trả lời luôn
    if len(suggestions) == 1:
        context = prepare_context(suggestions)
        answer  = ask_llama(user_query, context, lang_mode, intent)
        return {"type": "answer", "text": answer}

    # Nhiều kết quả → hỏi lại người dùng
    sug_names = [d.get("main_name", "Không rõ") for d in suggestions]
    logger.info(f"Suggestions: {sug_names}")
    return {
        "type": "suggestions",
        "text": "Mình chưa xác định chính xác bệnh bạn muốn hỏi. Bạn có muốn nói đến một trong các bệnh sau không?",
        "suggestions": sug_names
    }


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

    # Generic check
    q_norm = normalize_text(user_query)
    STOPWORDS = {"trieu chung", "dau hieu", "benh", "thuoc", "cach chua", "dieu tri", "phong ngua"}
    if any(q_norm == sw or q_norm.strip() == sw for sw in STOPWORDS):
        return jsonify({
            "type": "answer",
            "text": "Câu hỏi chưa đủ cụ thể. Vui lòng nhập tên bệnh bạn muốn tra cứu."
        })

    result = process_query(user_query, lang_mode)
    return jsonify(result)


@app.route("/disease/<name>")
def disease_detail(name):
    lang_mode = request.args.get("lang", "Auto")
    result    = process_query(name, lang_mode)
    return jsonify(result)


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(
        debug=(config.FLASK_ENV == "development"),
        port=config.FLASK_PORT
    )