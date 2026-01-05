import os
import pandas as pd
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -------------------
# Paths
# -------------------
DATA_PATH = Path("data/processed/filtered_complaints.csv")
VECTOR_STORE_DIR = Path("vector_store")
VECTOR_STORE_DIR.mkdir(exist_ok=True)

INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.csv"

# -------------------
# Parameters
# -------------------
SAMPLE_SIZE = 2000
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# -------------------
# Helper: chunk text
# -------------------
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap

    return chunks

# -------------------
# Main
# -------------------
def main():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["cleaned_narrative"])
    df = df.sample(SAMPLE_SIZE, random_state=42)

    all_chunks = []
    all_metadata = []

    print("✂️ Chunking narratives...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        chunks = chunk_text(row["cleaned_narrative"])

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "product": row["Product"]
            })

    print("🧠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔢 Creating embeddings...")
    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("📦 Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("💾 Saving vector store...")
    faiss.write_index(index, str(INDEX_PATH))
    pd.DataFrame(all_metadata).to_csv(METADATA_PATH, index=False)

    print("✅ Task 2 complete: vector store created")

if __name__ == "__main__":
    main()
