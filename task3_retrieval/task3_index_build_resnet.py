import os
import faiss
import numpy as np

OUTPUT_DIR = "task3_resnet/outputs"
INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss_index.bin")

embeddings = np.load(os.path.join(OUTPUT_DIR, "embeddings.npy")).astype("float32")

dim = embeddings.shape[1]

# Use Inner Product for cosine similarity
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)

print("FAISS index built successfully.")

