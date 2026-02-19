import os
import numpy as np
import faiss

OUTPUT_DIR = "task3_resnet/outputs"
INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss_index.bin")

embeddings = np.load(os.path.join(OUTPUT_DIR, "embeddings.npy"))
labels = np.load(os.path.join(OUTPUT_DIR, "labels.npy"))

index = faiss.read_index(INDEX_FILE)

def compute_precision_at_k(k):
    total_precision = 0

    for i in range(len(embeddings)):
        query_emb = embeddings[i].reshape(1, -1)
        distances, indices = index.search(query_emb, k + 1)

        retrieved = indices[0][1:]  # remove self
        correct = sum(labels[idx] == labels[i] for idx in retrieved)

        total_precision += correct / k

    return total_precision / len(embeddings)

for k in [1, 5, 10]:
    precision = compute_precision_at_k(k)
    print(f"Precision@{k}: {precision:.4f}")

