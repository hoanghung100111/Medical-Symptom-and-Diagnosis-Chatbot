import json
import pickle
import sys
import faiss
from sentence_transformers import SentenceTransformer

def build_faiss_index(json_path, index_path="medical_index.faiss", map_path="id2doc.pkl"):
    # Load dữ liệu từ JSON
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Kiểm tra format dữ liệu
    if not isinstance(docs, list) or "main_name" not in docs[0]:
        raise ValueError("❌ JSON phải là list các dict có key 'main_name'!")

    texts = [d["main_name"] for d in docs]

    print(f"✅ Loaded {len(texts)} documents from {json_path}")

    # Load SentenceTransformer để encode
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Tạo FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Lưu index
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index to {index_path}")

    # Lưu mapping id -> doc
    id2doc = {i: docs[i] for i in range(len(docs))}
    with open(map_path, "wb") as f:
        pickle.dump(id2doc, f)

    print(f"✅ Saved ID-to-doc mapping to {map_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Usage: python build_index.py chatbot_knowledge_updated.json")
        sys.exit(1)

    json_file = sys.argv[1]
    build_faiss_index(json_file)
