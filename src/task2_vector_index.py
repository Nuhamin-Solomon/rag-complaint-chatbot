"""
Task 2: Sampling, Chunking, Embedding, Vector Store Creation
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from pathlib import Path

DATA_PATH = Path("data/processed/filtered_complaints.csv")
VECTOR_DIR = Path("vector_store")

# ---------------------------
# Load cleaned data
# ---------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------
# Stratified sampling (10k–15k)
# ---------------------------
sampled_df = (
    df.groupby("Product", group_keys=False)
      .apply(lambda x: x.sample(min(len(x), 3000), random_state=42))
)

print(f"Sampled complaints: {len(sampled_df)}")

# ---------------------------
# Chunking
# ---------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = []
metadatas = []

for _, row in sampled_df.iterrows():
    chunks = splitter.split_text(row["cleaned_narrative"])
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "complaint_id": row.get("Complaint ID"),
            "product": row["Product"],
            "issue": row.get("Issue"),
            "sub_issue": row.get("Sub-issue"),
            "chunk_index": i
        })

print(f"Generated {len(documents)} chunks")

# ---------------------------
# Embeddings
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, show_progress_bar=True)

# ---------------------------
# Vector store (ChromaDB)
# ---------------------------
VECTOR_DIR.mkdir(exist_ok=True)

client = chromadb.Client(
    settings=chromadb.Settings(
        persist_directory=str(VECTOR_DIR)
    )
)

collection = client.get_or_create_collection("complaints")

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=[str(i) for i in range(len(documents))]
)

client.persist()
print("Vector store persisted to vector_store/")
