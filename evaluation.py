"""
evaluation.py — Đánh giá chất lượng hệ thống RAG Medical Chatbot
================================================================
Metrics:
  - Retrieval Accuracy  : correct doc có trong top-K results không?
  - MRR (Mean Reciprocal Rank): rank trung bình của correct doc
  - Answer Relevance    : cosine similarity giữa câu trả lời và ground truth
  - Avg Latency         : thời gian phản hồi trung bình

Usage:
  python evaluation.py
  python evaluation.py --top_k 5 --output results/eval_report.json
"""

import json
import time
import argparse
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Ground truth test set ─────────────────────────────────────────────────────
# Mỗi sample: query → expected disease name + expected keywords trong answer
TEST_SET = [
    {
        "query": "Triệu chứng của bệnh tiểu đường type 2 là gì?",
        "expected_disease": "tiểu đường",
        "expected_keywords": ["đường huyết", "insulin", "khát nước", "mệt mỏi"],
    },
    {
        "query": "Cách điều trị tăng huyết áp",
        "expected_disease": "tăng huyết áp",
        "expected_keywords": ["huyết áp", "thuốc", "lối sống", "natri"],
    },
    {
        "query": "Bệnh cảm cúm lây qua đường nào?",
        "expected_disease": "cảm cúm",
        "expected_keywords": ["virus", "lây", "hô hấp", "ho"],
    },
    {
        "query": "Sốt xuất huyết có nguy hiểm không?",
        "expected_disease": "sốt xuất huyết",
        "expected_keywords": ["dengue", "tiểu cầu", "sốt", "muỗi"],
    },
    {
        "query": "Triệu chứng viêm phổi ở người lớn",
        "expected_disease": "viêm phổi",
        "expected_keywords": ["ho", "sốt", "khó thở", "phổi"],
    },
    {
        "query": "Migraine đau đầu điều trị như thế nào?",
        "expected_disease": "migraine",
        "expected_keywords": ["đau đầu", "buồn nôn", "ánh sáng", "thuốc"],
    },
    {
        "query": "Viêm loét dạ dày ăn gì kiêng gì?",
        "expected_disease": "viêm loét dạ dày",
        "expected_keywords": ["dạ dày", "H. pylori", "axit", "ăn uống"],
    },
    {
        "query": "Dị ứng thức ăn biểu hiện như thế nào?",
        "expected_disease": "dị ứng",
        "expected_keywords": ["dị ứng", "phản ứng", "miễn dịch", "ngứa"],
    },
]


# ── Setup ─────────────────────────────────────────────────────────────────────
def load_resources():
    logger.info("Loading models and index...")
    embedder = SentenceTransformer(config.EMBED_MODEL)
    index    = faiss.read_index(config.FAISS_INDEX_PATH)
    client   = Groq(api_key=config.GROQ_API_KEY)

    with open(config.KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    id2doc = {i: doc for i, doc in enumerate(full_data)}

    logger.info(f"Loaded {len(full_data)} documents.")
    return embedder, index, client, full_data, id2doc


# ── Retrieval helpers ─────────────────────────────────────────────────────────
def retrieve(query, embedder, index, id2doc, top_k):
    vec  = embedder.encode([query])
    D, I = index.search(vec, top_k)
    return [id2doc[i] for i in I[0] if i in id2doc]


def fuzzy_match(query, doc):
    """Return fuzzy score between query and doc main_name / synonyms."""
    import unicodedata
    def norm(t):
        t = t.lower().strip()
        t = unicodedata.normalize("NFD", t)
        return "".join(c for c in t if unicodedata.category(c) != "Mn")

    q = norm(query)
    candidates = [doc.get("main_name", "")] + doc.get("synonyms", [])
    return max(fuzz.partial_ratio(q, norm(c)) for c in candidates)


# ── Answer generation ─────────────────────────────────────────────────────────
def generate_answer(query, docs, client):
    context = "\n\n".join(
        f"{d.get('main_name','')}\n{d.get('knowledge_text','')}" for d in docs
    )
    if not context:
        return "Không tìm thấy thông tin."

    prompt = f"""You are a helpful medical assistant. Answer in Vietnamese.
Use the following context to answer the question.

Context:
{context[:config.CONTEXT_CHAR_LIMIT]}

Question: {query}
Answer:"""

    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.MAX_TOKENS_ANSWER,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ── Metrics ───────────────────────────────────────────────────────────────────
def retrieval_accuracy(retrieved_docs, expected_disease):
    """1 nếu expected disease xuất hiện trong retrieved docs."""
    import unicodedata
    def norm(t):
        t = t.lower().strip()
        t = unicodedata.normalize("NFD", t)
        return "".join(c for c in t if unicodedata.category(c) != "Mn")

    exp = norm(expected_disease)
    for doc in retrieved_docs:
        names = [doc.get("main_name", "")] + doc.get("synonyms", [])
        if any(fuzz.partial_ratio(exp, norm(n)) >= 70 for n in names):
            return 1
    return 0


def reciprocal_rank(retrieved_docs, expected_disease):
    """1/rank nếu tìm thấy, 0 nếu không."""
    import unicodedata
    def norm(t):
        t = t.lower().strip()
        t = unicodedata.normalize("NFD", t)
        return "".join(c for c in t if unicodedata.category(c) != "Mn")

    exp = norm(expected_disease)
    for rank, doc in enumerate(retrieved_docs, start=1):
        names = [doc.get("main_name", "")] + doc.get("synonyms", [])
        if any(fuzz.partial_ratio(exp, norm(n)) >= 70 for n in names):
            return 1 / rank
    return 0.0


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Tỉ lệ expected keywords xuất hiện trong answer."""
    import unicodedata
    def norm(t):
        t = t.lower()
        t = unicodedata.normalize("NFD", t)
        return "".join(c for c in t if unicodedata.category(c) != "Mn")

    ans_norm = norm(answer)
    hits = sum(1 for kw in keywords if norm(kw) in ans_norm)
    return hits / len(keywords) if keywords else 0.0


def answer_relevance_cosine(query, answer, embedder) -> float:
    """Cosine similarity giữa embedding của query và answer."""
    vecs = embedder.encode([query, answer])
    sim  = cosine_similarity([vecs[0]], [vecs[1]])[0][0]
    return float(sim)


# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation(top_k=3):
    embedder, index, client, full_data, id2doc = load_resources()

    results   = []
    acc_list  = []
    mrr_list  = []
    kw_list   = []
    cos_list  = []
    lat_list  = []

    for i, sample in enumerate(TEST_SET, 1):
        query    = sample["query"]
        exp_dis  = sample["expected_disease"]
        exp_kws  = sample["expected_keywords"]

        logger.info(f"[{i}/{len(TEST_SET)}] {query}")

        # Retrieval
        t0   = time.time()
        docs = retrieve(query, embedder, index, id2doc, top_k)
        ret_time = time.time() - t0

        # Answer generation
        t1     = time.time()
        answer = generate_answer(query, docs, client)
        gen_time = time.time() - t1

        total_latency = ret_time + gen_time

        # Metrics
        acc  = retrieval_accuracy(docs, exp_dis)
        rr   = reciprocal_rank(docs, exp_dis)
        kw   = keyword_coverage(answer, exp_kws)
        cos  = answer_relevance_cosine(query, answer, embedder)

        acc_list.append(acc)
        mrr_list.append(rr)
        kw_list.append(kw)
        cos_list.append(cos)
        lat_list.append(total_latency)

        result = {
            "query":              query,
            "expected_disease":   exp_dis,
            "retrieved_diseases": [d.get("main_name") for d in docs],
            "retrieval_hit":      bool(acc),
            "reciprocal_rank":    round(rr, 4),
            "keyword_coverage":   round(kw, 4),
            "answer_cosine_sim":  round(cos, 4),
            "latency_sec":        round(total_latency, 3),
            "answer_preview":     answer[:200] + "..." if len(answer) > 200 else answer,
        }
        results.append(result)

        logger.info(
            f"  hit={acc} | RR={rr:.2f} | kw={kw:.2f} | cos={cos:.3f} | {total_latency:.2f}s"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "top_k":                    top_k,
        "n_samples":                len(TEST_SET),
        "retrieval_accuracy":       round(np.mean(acc_list), 4),
        "mrr":                      round(np.mean(mrr_list), 4),
        "avg_keyword_coverage":     round(np.mean(kw_list),  4),
        "avg_answer_cosine_sim":    round(np.mean(cos_list), 4),
        "avg_latency_sec":          round(np.mean(lat_list), 3),
        "p95_latency_sec":          round(float(np.percentile(lat_list, 95)), 3),
    }

    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)
    print(f"  Samples evaluated   : {summary['n_samples']}")
    print(f"  Top-K               : {top_k}")
    print(f"  Retrieval Accuracy  : {summary['retrieval_accuracy']:.2%}")
    print(f"  MRR                 : {summary['mrr']:.4f}")
    print(f"  Keyword Coverage    : {summary['avg_keyword_coverage']:.2%}")
    print(f"  Answer Cosine Sim   : {summary['avg_answer_cosine_sim']:.4f}")
    print(f"  Avg Latency         : {summary['avg_latency_sec']}s")
    print(f"  P95 Latency         : {summary['p95_latency_sec']}s")
    print("="*55 + "\n")

    return {"summary": summary, "details": results}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Medical Chatbot RAG")
    parser.add_argument("--top_k",  type=int, default=3,            help="Top-K retrieval")
    parser.add_argument("--output", type=str, default="eval_report.json", help="Output JSON path")
    args = parser.parse_args()

    report = run_evaluation(top_k=args.top_k)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Report saved to {out_path}")