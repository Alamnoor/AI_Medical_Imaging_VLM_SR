import os
import torch
import numpy as np
import faiss
from torchvision import models, transforms
from medmnist import PneumoniaMNIST

OUTPUT_DIR = "task3_resnet/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load PneumoniaMNIST test split
test_dataset = PneumoniaMNIST(split='test', download=True)

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


embeddings = []
labels = []

print("Extracting embeddings...")

for img, label in test_dataset:
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(img)

    embeddings.append(emb.cpu().numpy().squeeze())
    labels.append(label.item())

embeddings = np.array(embeddings).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), np.array(labels))

print("Embeddings saved successfully.")

