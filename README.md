# rag-complaint-chatbot
Internal RAG-powered complaint analytics chatbot for financial services using CFPB data
# RAG Complaint Analysis Chatbot

This project builds a Retrieval-Augmented Generation (RAG) system
to analyze CFPB consumer complaints and enable internal teams to
query customer pain points using natural language.

## Project Structure
project_root/ │ ├─ README.md ├─ requirements.txt ├─ .gitignore │ ├─ data/ │ ├─ raw/ # Original raw complaint datasets │ ├─ processed/ # Cleaned and preprocessed complaint data │ └─ vector_store/ # FAISS index and metadata │ ├─ src/ # Source code modules │ └─ vector_store_builder.py │ ├─ notebooks/ # Exploratory notebooks │ └─ 02_build_vector_store.ipynb │ ├─ tests/ # Unit tests │ └─ test_vector_store.py │ └─ .github/workflows/ # Optional CI/CD pipelines
