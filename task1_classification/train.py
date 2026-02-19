import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from data.dataset_utils import get_pneumonia_data
from models.model_utils import build_model, save_model
import matplotlib.pyplot as plt

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

# Data
train_loader, val_loader, _ = get_pneumonia_data(BATCH_SIZE)

# Model
model = build_model().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses = []
val_metrics = []

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = criterion(logits, y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_loss = total_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # --- Validation ---
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.float().to(device)
            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)  # Convert logits to probabilities
            all_labels.append(y.cpu())
            all_preds.append(probs.cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    preds_binary = (all_preds >= 0.5).astype(int)

    val_auc = roc_auc_score(all_labels, all_preds)
    val_acc = accuracy_score(all_labels, preds_binary)
    val_prec = precision_score(all_labels, preds_binary)
    val_rec = recall_score(all_labels, preds_binary)
    val_f1 = f1_score(all_labels, preds_binary)

    val_metrics.append({
        "AUC": val_auc, "Accuracy": val_acc,
        "Precision": val_prec, "Recall": val_rec, "F1": val_f1
    })

    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"Loss: {epoch_loss:.4f}, "
          f"Val AUC: {val_auc:.4f}, "
          f"Acc: {val_acc:.4f}, "
          f"Prec: {val_prec:.4f}, "
          f"Rec: {val_rec:.4f}, "
          f"F1: {val_f1:.4f}")

    scheduler.step()

# Save model
save_model(model, "models/cnn_pneumonia.pth")

# Plot training loss
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("reports/training_loss.png")
plt.close()
