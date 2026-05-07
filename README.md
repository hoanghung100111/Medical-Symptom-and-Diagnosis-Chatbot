# 🩺 SympCheck — Medical Information Assistant

> A bilingual (Vietnamese/English) RAG-powered chatbot that helps users look up symptom and disease information from a curated medical knowledge base.  
> **Disclaimer: This tool is for informational purposes only and does not replace professional medical advice.**

---

## 📸 Demo

![SympCheck Demo](assets/demo.png)

> Example Streamlit interface showing bilingual medical information retrieval.

---

## 📌 Problem Statement

Many Vietnamese users turn to unreliable sources (forums, social media) when they experience unfamiliar symptoms. SympCheck provides a structured, verifiable alternative — grounding every answer in a curated medical knowledge base rather than relying solely on LLM hallucination.

---

## 🏗️ Architecture

```text
User Query
    │
    ▼
┌─────────────────────┐
│   Fuzzy Matching    │  ← rapidfuzz, normalized Vietnamese text
│  (disease name)     │
└────────┬────────────┘
         │ hit / miss
         ▼
┌─────────────────────┐
│   FAISS Retrieval   │  ← sentence-transformers (multilingual MiniLM)
│  (semantic search)  │
└────────┬────────────┘
         │ top-k docs
         ▼
┌─────────────────────┐
│   LLM Generation    │  ← Groq API (Llama 3.1 8B Instant)
│  (answer synthesis) │
└────────┬────────────┘
         │
         ▼
    Streamlit UI
```

**Two-stage retrieval strategy:**
1. **Fuzzy match first** — handles typos and Vietnamese diacritic variations (e.g., "tieu duong" → "Tiểu đường")
2. **Semantic FAISS fallback** — catches queries that don't match any disease name directly but describe symptoms

---

## ✨ Features

- **Bilingual support** — auto-detects Vietnamese / English, or user can force a language via sidebar
- **Two-stage retrieval** — fuzzy match → semantic search, minimizes "not found" failures
- **Grounded answers** — LLM only answers from retrieved context, not from parametric memory
- **Generic query guard** — detects vague inputs (e.g., "triệu chứng") and prompts for clarification
- **Suggestion buttons** — when exact match fails, surfaces top-3 candidate diseases as clickable options
