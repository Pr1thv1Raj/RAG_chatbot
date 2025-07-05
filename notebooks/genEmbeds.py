import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Load the chunks from chunks folder
def load_chunks(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

chunk_file = "../chunks/chunks.json"
chunks = load_chunks(chunk_file)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks,show_progress_bar=True)
embeddings_np = np.array(embeddings).astype("float32")

dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# store the indices (vector database) in vectordb
faiss.write_index(index, "../vectordb/faiss_index.index")
