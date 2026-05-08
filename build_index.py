"""
build_index.py
--------------
Build FAISS index từ chatbot_knowledge_chunked.json.
Mỗi chunk được embed độc lập → semantic search chính xác theo từng chủ đề.

Chạy 1 lần trước khi khởi động app:
    python build_index.py
"""

import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────
CHUNKED_JSON   = "chatbot_knowledge_chunked.json"
INDEX_PATH     = "medical_index.faiss"
ID2CHUNK_PATH  = "id2chunk.pkl"          # thay id2doc.pkl cũ
MODEL_NAME     = "intfloat/multilingual-e5-base"  # tốt cho tiếng Việt
BATCH_SIZE     = 64
# ───────────────────────────────────────────────────────────────────────────


def main():
    print(f"📂 Loading chunks từ {CHUNKED_JSON}...")
    with open(CHUNKED_JSON, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   → {len(chunks)} chunks")

    print(f"\n🤖 Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Dùng embed_text (đã augment tên bệnh + nhãn chủ đề) để embed
    print("\n⚙️  Encoding chunks...")
    texts = [c["embed_text"] for c in chunks]

    # multilingual-e5 cần prefix "query: " và "passage: "
    # Khi build index dùng "passage: "
    passage_texts = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        passage_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # chuẩn hóa để dùng dot product = cosine
        convert_to_numpy=True,
    )

    dim = embeddings.shape[1]
    print(f"\n📐 Embedding dim: {dim}")

    # Build FAISS index với Inner Product (= cosine khi đã normalize)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"✅ FAISS index: {index.ntotal} vectors")

    # Lưu index
    faiss.write_index(index, INDEX_PATH)
    print(f"💾 Saved index → {INDEX_PATH}")

    # Lưu mapping id → chunk metadata (không lưu embed_text để tiết kiệm RAM)
    id2chunk = {}
    for i, c in enumerate(chunks):
        id2chunk[i] = {
            "chunk_id":    c["chunk_id"],
            "disease_name": c["disease_name"],
            "synonyms":    c["synonyms"],
            "label":       c["label"],
            "label_vi":    c["label_vi"],
            "text":        c["text"],
            "reference":   c["reference"],
        }

    with open(ID2CHUNK_PATH, "wb") as f:
        pickle.dump(id2chunk, f)
    print(f"💾 Saved metadata → {ID2CHUNK_PATH}")
    print(f"\n🎉 Done! Index sẵn sàng cho app.py")


if __name__ == "__main__":
    main()