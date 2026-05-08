# 🏥 MediChat — AI Medical Symptom & Diagnosis Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Flask-3.x-lightgrey?style=flat-square&logo=flask" />
  <img src="https://img.shields.io/badge/LLM-Llama_3.1_8B-orange?style=flat-square&logo=meta" />
  <img src="https://img.shields.io/badge/RAG-FAISS_+_SentenceTransformers-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

> A bilingual (Vietnamese / English) AI chatbot that answers medical questions using **Retrieval-Augmented Generation (RAG)**: dense vector search over a curated medical knowledge base, powered by Llama 3.1 via Groq API.

---

## ✨ Features

- **RAG pipeline** — FAISS vector index + Sentence Transformers for semantic retrieval
- **Fuzzy matching** — RapidFuzz for robust disease name lookup (handles typos, synonyms)
- **Bilingual** — Auto-detects Vietnamese / English, or user-selectable
- **Clean clinical UI** — Custom Flask + HTML/CSS frontend (no Streamlit)
- **Configurable** — All parameters via `.env`, zero hardcoded secrets
- **Evaluation suite** — `evaluation.py` reports Retrieval Accuracy, MRR, Keyword Coverage, Cosine Similarity, Latency
- **Docker ready** — Single command deployment

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│                  Flask API                  │
│  POST /chat                                 │
│   ├── Generic query filter (stopwords)      │
│   ├── Fuzzy disease name matching           │
│   └── Fallback: FAISS semantic retrieval    │
└───────────────┬─────────────────────────────┘
                │  Top-K documents
                ▼
┌─────────────────────────────────────────────┐
│           Knowledge Base (RAG)              │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │  FAISS Index     │  │  JSON Knowledge  │ │
│  │  (dense vectors) │  │  Base (~N docs)  │ │
│  └──────────────────┘  └──────────────────┘ │
│  Sentence Transformers (multilingual MiniLM) │
└───────────────┬─────────────────────────────┘
                │  Retrieved context
                ▼
┌─────────────────────────────────────────────┐
│         Groq API — Llama 3.1 8B             │
│  Generates grounded, bilingual answer       │
└───────────────┬─────────────────────────────┘
                │
                ▼
           JSON Response
```

---

## 📁 Project Structure

```
Medical-Symptom-and-Diagnosis-Chatbot/
├── app.py                          # Flask app & RAG logic
├── config.py                       # Centralised config (loads .env)
├── evaluation.py                   # RAG evaluation suite
├── .env.example                    # Environment variable template
├── Dockerfile                      # Production Docker image
├── docker-compose.yml              # One-command deployment
├── requirements.txt
├── templates/
│   └── index.html                  # Chat UI (clinical light theme)
├── medical_index.faiss             # FAISS vector index
└── chatbot_knowledge_updated.json  # Medical knowledge base
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/hoanghung100111/Medical-Symptom-and-Diagnosis-Chatbot.git
cd Medical-Symptom-and-Diagnosis-Chatbot
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your GROQ_API_KEY
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000)

---

## 🐳 Docker

```bash
# Build & run
docker compose up --build

# Run in background
docker compose up -d

# Stop
docker compose down
```

---

## 📊 Evaluation

Run the full evaluation suite against the built-in test set:

```bash
python evaluation.py
# With custom top-K and output path:
python evaluation.py --top_k 5 --output results/eval_report.json
```

Example output:

```
=======================================================
  EVALUATION SUMMARY
=======================================================
  Samples evaluated   : 8
  Top-K               : 3
  Retrieval Accuracy  : 87.50%
  MRR                 : 0.8333
  Keyword Coverage    : 74.25%
  Answer Cosine Sim   : 0.7841
  Avg Latency         : 2.134s
  P95 Latency         : 3.210s
=======================================================
```

---

## ⚙️ Configuration

All parameters are controlled via `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | LLM model |
| `EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `TOP_K` | `3` | Number of retrieved documents |
| `SIMILARITY_THRESHOLD` | `0.55` | Minimum FAISS similarity score |
| `FUZZY_THRESHOLD` | `70` | Minimum fuzzy match score (0–100) |
| `FLASK_PORT` | `5000` | Server port |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.1 8B via [Groq](https://groq.com) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector Search | [FAISS](https://github.com/facebookresearch/faiss) |
| Fuzzy Match | [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) |
| Backend | [Flask](https://flask.palletsprojects.com/) |
| Frontend | HTML / CSS / Vanilla JS |
| Deployment | Docker + Gunicorn |

---

## ⚠️ Disclaimer

> This chatbot is for **informational purposes only** and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.