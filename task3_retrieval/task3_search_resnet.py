import os
import torch
import numpy as np
import faiss
import argparse
import matplotlib.pyplot as plt
from torchvision import models, transforms
from medmnist import PneumoniaMNIST

parser = argparse.ArgumentParser()
parser.add_argument("--query_index", type=int, required=True)
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()

OUTPUT_DIR = "task3_resnet/outputs"
INDEX_FILE = os.path.join(OUTPUT_DIR, "faiss_index.bin")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
test_dataset = PneumoniaMNIST(split='test', download=True)

# Load index and embeddings
index = faiss.read_index(INDEX_FILE)
embeddings = np.load(os.path.join(OUTPUT_DIR, "embeddings.npy"))
labels = np.load(os.path.join(OUTPUT_DIR, "labels.npy"))

# Query embedding
query_emb = embeddings[args.query_index].reshape(1, -1)

distances, indices = index.search(query_emb, args.k + 1)

# Remove self-match
retrieved_indices = indices[0][1:]

query_img, query_label = test_dataset[args.query_index]

plt.figure(figsize=(15, 3))
plt.subplot(1, args.k + 1, 1)
plt.imshow(query_img, cmap='gray')
plt.title(f"Query (Label {query_label})")
plt.axis('off')

for i, idx in enumerate(retrieved_indices):
    img, label = test_dataset[idx]
    plt.subplot(1, args.k + 1, i + 2)
    plt.imshow(img, cmap='gray')
    plt.title(f"Rank {i+1} (L {label})")
    plt.axis('off')

plt.tight_layout()
plt.show()

