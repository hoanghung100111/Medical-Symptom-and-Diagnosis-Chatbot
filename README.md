# Medical Symptom and Diagnosis Chatbot

### Project Overview

This is a medical Q&A chatbot built on a **Retrieval-Augmented Generation (RAG)** architecture. The project's goal is to provide an intelligent, reliable tool that helps users find information about symptoms and disease diagnoses.

Unlike traditional chatbots that rely solely on a Large Language Model (LLM), this system uses a private, curated medical knowledge base. This ensures that the information provided is accurate, specific, and verifiable.

---

### Key Features

* **AI Integration:** Leverages a Large Language Model (LLM) to understand and generate natural, user-friendly responses.
* **Intelligent Retrieval:** Integrates a retrieval system based on **FAISS** and **Sentence-Transformers** to find relevant information from a predefined dataset.
* **Flexible Search:** Supports both **Semantic Search** and **Fuzzy Matching** to handle complex queries and typos.
* **Web Interface:** Features a simple and intuitive user interface built with **Streamlit** for easy interaction.
* **Accurate Information:** Prioritizes returning information from the verified database, while also including a safety disclaimer that the answers are not professional medical advice.

---

### Workflow

The system operates in two main phases:

#### Phase 1: Index Building

1.  Medical data from `chatbot_knowledge_updated.json` is processed and encoded into vector embeddings.
2.  These vectors are stored in a high-efficiency search index (**FAISS**). This process is handled by the `build_index.py` script.

#### Phase 2: Chatbot Loop

1.  The user inputs a query.
2.  The system first performs a Fuzzy Match to find the most relevant disease names in the database.
3.  If no exact match is found, the system uses the FAISS index to retrieve semantically similar passages.
4.  The retrieved information is combined with the user's query and sent to an LLM (e.g., Llama) to generate a detailed and easy-to-understand answer.
5.  The final response is displayed to the user via the web interface.

---

### Technologies & Libraries

* **Language:** Python
* **Web Framework:** Streamlit
* **Embedding Model:** `sentence-transformers`
* **Vector Search:** FAISS
* **LLM:** Groq API (using Llama 3)
* **Fuzzy Search:** `rapidfuzz`
* **Data Science:** `json`, `os`, `numpy`, `unicodedata`

---

### Future Directions

* **Multimodal Integration:** Allow users to input images of symptoms to assist with diagnosis.
* **Conversational Memory:** Build conversational memory so the chatbot can ask follow-up questions based on previously provided symptoms.
* **Dataset Expansion:** Add more data from reputable sources to improve the chatbot's accuracy and coverage.
* **Deployment:** Optimize the system for deployment on cloud platforms.
