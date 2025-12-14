import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report  # <-- NEW
import argparse
from vit_model import ViT

# -------------------------------------------------
# Make output directory for images/reports
# -------------------------------------------------
os.makedirs("outputs", exist_ok=True)

# -------------------------------------------------
# Dataset class
# -------------------------------------------------
class SignLanguageMNIST(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------------------------------------
# Transform
# -------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load test set
test_dataset = SignLanguageMNIST("./data/sign_mnist_test.csv", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------------------------------------------------
# Args: select model type
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['cnn','vit'], default='cnn')
args = parser.parse_args()

# -------------------------------------------------
# CNN Model (must match train.py)
# -------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.view(-1, 64*5*5)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

# -------------------------------------------------
# Helper: class id -> letter names
# -------------------------------------------------
def make_class_names(num_classes: int):
    """
    Tries to produce reasonable letter labels.
    - 26 -> A-Z
    - 25 -> A-Y
    - 24 -> A-I, K-Y (common Sign Language MNIST convention: J omitted)
    Otherwise -> Class 0..N-1
    """
    alphabet = [chr(ord('A') + i) for i in range(26)]
    if num_classes == 26:
        return alphabet
    if num_classes == 25:
        return alphabet[:25]  # A..Y
    if num_classes == 24:
        # Common Kaggle Sign Language MNIST: J omitted (dynamic gesture)
        return alphabet[:9] + alphabet[10:25]  # A..I + K..Y
    return [f"Class {i}" for i in range(num_classes)]

# -------------------------------------------------
# Load Model
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device) if args.model == 'cnn' else ViT(img_size=28, patch_size=4, in_chans=1, num_classes=25).to(device)
model_path = f"sign_model_{args.model}.pt"
if not os.path.exists(model_path):
    model_path = "sign_model.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"\nModel Loaded: {model_path}\n")

if hasattr(model, "fc2"):
    num_classes = model.fc2.out_features
else:
    num_classes = model.head.out_features
class_names = make_class_names(num_classes)

# -------------------------------------------------
# Evaluate Model
# -------------------------------------------------
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%\n")

# -------------------------------------------------
# Per-class Precision/Recall/F1 (PRINT + SAVE CSV)
# -------------------------------------------------
# Force the report to include every class 0..num_classes-1 even if some don't appear
labels_order = list(range(num_classes))

report_dict = classification_report(
    all_labels,
    all_preds,
    labels=labels_order,
    target_names=class_names,
    output_dict=True,
    digits=4,
    zero_division=0
)

# Pretty print
report_text = classification_report(
    all_labels,
    all_preds,
    labels=labels_order,
    target_names=class_names,
    digits=4,
    zero_division=0
)
print("Per-class metrics (Precision / Recall / F1):")
print(report_text)

# Save to CSV
df_report = pd.DataFrame(report_dict).T
df_report.to_csv("outputs/per_class_metrics.csv", index=True)
print("Saved: outputs/per_class_metrics.csv\n")

# -------------------------------------------------
# Confusion Matrix (SAVE + SHOW) with letter labels
# -------------------------------------------------
cm = confusion_matrix(all_labels, all_preds, labels=labels_order)

fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

disp.plot(
    cmap="Blues",
    xticks_rotation="vertical",
    ax=ax,
    values_format='d'
)

for txt in disp.text_.ravel():
    txt.set_fontsize(6)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()

# -------------------------------------------------
# Show 12 Predictions (SAVE + SHOW) with letter labels
# -------------------------------------------------
import random
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for ax in axes.flatten():
    idx = random.randint(0, len(test_dataset) - 1)
    img, true_label = test_dataset[idx]

    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        pred = torch.argmax(output).item()

    img_vis = img * 0.5 + 0.5
    img_vis = img_vis.squeeze().numpy()

    true_name = class_names[true_label] if 0 <= true_label < len(class_names) else str(true_label)
    pred_name = class_names[pred] if 0 <= pred < len(class_names) else str(pred)

    ax.imshow(img_vis, cmap="gray")
    ax.set_title(f"True: {true_name} | Pred: {pred_name}")
    ax.axis("off")

plt.tight_layout()
fig.savefig("outputs/sample_predictions.png", dpi=300, bbox_inches="tight")
plt.show()
