import os
from torchvision import transforms
from torchvision.utils import save_image
from data.dataset_utils import get_pneumonia_data

# Create folders to store images
SAVE_FOLDER = "task2_report_generation/images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Upscale transform for better visibility
upscale = transforms.Resize((224, 224))  # only for saving

# Load dataset (use positional argument for batch size)
BATCH_SIZE = 64
train_loader, val_loader, test_loader = get_pneumonia_data(BATCH_SIZE)

# Function to save a loader's images
def save_loader_images(loader, split_name):
    split_folder = os.path.join(SAVE_FOLDER, split_name)
    os.makedirs(split_folder, exist_ok=True)
    for i, (x, y) in enumerate(loader):
        for j in range(x.size(0)):
            img = x[j]  # 28x28 tensor
            img = upscale(img)
            save_path = os.path.join(split_folder, f"img_{i*BATCH_SIZE+j}_label{int(y[j].item())}.png")
            save_image(img, save_path)

# Save images for train, val, test
save_loader_images(train_loader, "train")
save_loader_images(val_loader, "val")
save_loader_images(test_loader, "test")

print("Images saved successfully!")
