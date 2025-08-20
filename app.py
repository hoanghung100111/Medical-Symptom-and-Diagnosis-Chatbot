import os
import json
import faiss
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import unicodedata

# ============================================================
# Setup
# ============================================================
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Embedding model
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Load FAISS index
index = faiss.read_index("medical_index.faiss")

# Load dữ liệu JSON
try:
    with open("chatbot_knowledge_updated.json", "r", encoding="utf-8") as f:
        full_data = json.load(f)
except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'chatbot_knowledge_updated.json'")
    st.stop()

id2doc = {i: doc for i, doc in enumerate(full_data)}

# ============================================================
# Helper functions
# ============================================================
STOPWORDS = {"trieu chung", "dau hieu", "benh", "thuoc", "cach chua", "dieu tri", "phong ngua"}

def normalize_text(text: str) -> str:
    """Bỏ dấu, lowercase"""
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join([c for c in text if unicodedata.category(c) != "Mn"])
    return text

def is_query_too_generic(query: str) -> bool:
    """Check xem query chỉ chứa stopwords không"""
    q_norm = normalize_text(query)
    return any(q_norm == sw or q_norm.strip() in sw for sw in STOPWORDS)

def extract_disease(query: str, kb):
    """Tìm bệnh trong query bằng fuzzy partial match"""
    q_norm = normalize_text(query)
    best_doc, best_score = None, 0

    for doc in kb:
        candidates = [doc.get("main_name", "")] + doc.get("synonyms", [])
        for cand in candidates:
            cand_norm = normalize_text(cand)
            score = fuzz.partial_ratio(q_norm, cand_norm)
            if score > best_score:
                best_score, best_doc = score, doc

    if best_score >= 70:  # threshold fuzzy
        return best_doc
    return None

def retrieve_context(query, top_k=3, threshold=0.55):
    """Tìm trong FAISS, có ngưỡng similarity"""
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, top_k)

    docs = []
    for score, idx in zip(D[0], I[0]):
        if idx in id2doc and score >= threshold:
            docs.append(id2doc[idx])
    return docs

def summarize_context(context, max_tokens=400):
    """Tóm tắt context dài"""
    summary_prompt = f"Hãy tóm tắt ngắn gọn, giữ lại ý chính, dễ hiểu:\n\n{context}"
    summary = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return summary.choices[0].message.content

def prepare_context(docs, char_limit=3000):
    context_text = "\n\n".join(
        f"{d.get('main_name','')}\n{d.get('knowledge_text','')}" for d in docs
    )
    if len(context_text) > char_limit:
        chunks = [
            context_text[i : i + char_limit] for i in range(0, len(context_text), char_limit)
        ]
        summaries = [summarize_context(chunk) for chunk in chunks]
        return "\n\n".join(summaries)
    return context_text

def ask_llama(query, context, lang_mode="Auto"):
    if not context:
        return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp."

    if lang_mode == "Vietnamese":
        lang_instruction = "Always answer ONLY in Vietnamese."
    elif lang_mode == "English":
        lang_instruction = "Always answer ONLY in English."
    else:
        lang_instruction = """If the question is in Vietnamese, answer ONLY in Vietnamese.
If the question is in English, answer ONLY in English."""

    prompt = f"""You are a helpful bilingual medical assistant.
{lang_instruction}

Use the following medical context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.3,
    )
    return response.choices[0].message.content

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Medical Chatbot", page_icon="💊")
st.title("💊 Bilingual Medical Chatbot")

st.sidebar.title("⚙️ Settings")
lang_mode = st.sidebar.radio("Ngôn ngữ trả lời:", ("Auto", "Vietnamese", "English"), index=0)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_disease" in st.session_state:
    disease_name = st.session_state.pop("selected_disease")
    doc = extract_disease(disease_name, full_data)
    if doc:
        context = prepare_context([doc])
        answer = ask_llama(disease_name, context, lang_mode)
        st.session_state.messages.append({"role": "user", "content": disease_name})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.experimental_rerun()

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_query := st.chat_input("Nhập câu hỏi (tiếng Việt hoặc English):"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):
            # Nếu query quá chung chung → yêu cầu rõ hơn
            if is_query_too_generic(user_query):
                answer = "Câu hỏi của bạn chưa đủ cụ thể. Vui lòng nhập tên bệnh bạn muốn tra cứu."
                st.write(answer)
            else:
                # Ưu tiên fuzzy match
                doc = extract_disease(user_query, full_data)
                if doc:
                    docs = [doc]
                    context = prepare_context(docs)
                    answer = ask_llama(user_query, context, lang_mode)
                    st.write(answer)
                else:
                    # fallback FAISS
                    suggestions = retrieve_context(user_query, top_k=3)
                    if not suggestions:
                        answer = "Xin lỗi, tôi chưa tìm thấy thông tin về bệnh này trong dữ liệu."
                        st.write(answer)
                    else:
                        sug_names = [d.get("main_name", "Không rõ") for d in suggestions]
                        answer = "Mình chưa tìm thấy chính xác. Bạn có muốn nói đến một trong các bệnh sau không?"
                        st.write(answer)
                        for sug in sug_names:
                            if st.button(sug, key=f"sug_{sug}"):
                                st.session_state.selected_disease = sug
                                st.experimental_rerun()

        st.session_state.messages.append({"role": "assistant", "content": answer})
