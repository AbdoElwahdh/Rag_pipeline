from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss


def retrieve_top_k(q, k):
    # Load model and index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("context.index")

    # Load original data
    with open('data/fake_novel_dataset.json') as f:
        data = json.load(f)
    contexts = [dict['context'] for dict in data]

    # Input question
    q_embedding = model.encode([q])

    # Search
    D, indices = index.search(np.array(q_embedding), k=k)

    top_k_contexts = []
    print("\nTop relevant contexts:")
    for idx in indices[0]:
        top_k_contexts.append(contexts[idx])
        print("-", contexts[idx])
    return top_k_contexts


if __name__ == "__main__":
    print("topk:", retrieve_top_k(input("Enter your question: "), 5), end="\n")
