import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset_utils import get_pneumonia_data
from models.model_utils import build_model, load_model

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

# Data
_, _, test_loader = get_pneumonia_data(BATCH_SIZE)

# Load model
model = build_model().to(device)
model = load_model(model, "models/cnn_pneumonia.pth", device)

# Evaluate
all_preds, all_labels = [], []
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device).float()
        logits = model(x).squeeze()
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Convert predictions to binary labels
pred_labels = [1 if p > 0.5 else 0 for p in all_preds]

# Compute metrics
auc = roc_auc_score(all_labels, all_preds)
acc = accuracy_score(all_labels, pred_labels)
prec = precision_score(all_labels, pred_labels)
rec = recall_score(all_labels, pred_labels)
f1 = f1_score(all_labels, pred_labels)

print(f"Test AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix plot
cm = confusion_matrix(all_labels, pred_labels)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Pneumonia"], yticklabels=["Normal","Pneumonia"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("reports/confusion_matrix.png")
plt.close()

# Optional: ROC curve
from sklearn.metrics import roc_curve, auc as auc_metric
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc_metric(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("reports/roc_curve.png")
plt.close()
