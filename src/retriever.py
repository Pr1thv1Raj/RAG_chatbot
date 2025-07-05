import os
import json
from sentence_transformers import SentenceTransformer
import faiss

# Model to generate embeddings from the query
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load index
index_path = os.path.join(os.path.dirname(__file__), "..", "vectordb", "faiss_index.index")
index_path = os.path.abspath(index_path)

index = faiss.read_index(index_path)

# Load chunk texts
chunks_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chunks", "chunks.json"))
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks= json.load(f)


# # Print top results
# for i, idx in enumerate(indices[0]):
#     print(f"\nRank {i+1}:")
#     print(chunks[idx])

def retrieve_chunks(query):
    query_embedding = model.encode([query]).astype("float32")
    k = 5  # top-k results
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks





