from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss
# import os

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
with open('data/fake_novel_dataset.json') as f:
    data = json.load(f)

contexts = [dict['context'] for dict in data]
embeddings = model.encode(contexts)
print(embeddings.shape)

# Save embeddings with FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, "context.index")
