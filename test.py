import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------------------------
# Make output directory for images
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
        label = self.labels[idx]
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
# Load Model
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("sign_model.pt", map_location=device))
model.eval()

print("\nModel Loaded: sign_model.pt\n")

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

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# -------------------------------------------------
# Confusion Matrix (SAVE + SHOW)
# -------------------------------------------------
# cm = confusion_matrix(all_labels, all_preds)
# disp = ConfusionMatrixDisplay(cm)
# disp.plot(cmap="Blues", xticks_rotation="vertical")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig("outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")
# plt.show()

# -------------------------------------------------
# Confusion Matrix (SAVE + SHOW)
# -------------------------------------------------
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(10, 10))   # <-- bigger figure
disp = ConfusionMatrixDisplay(cm)

disp.plot(
    cmap="Blues",
    xticks_rotation="vertical",
    ax=ax,
    values_format='d'                      # <-- show integers
)

# make the text smaller so it fits in each cell
for txt in disp.text_.ravel():
    txt.set_fontsize(6)                   # try 5–8 and see what you like

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.show()





# -------------------------------------------------
# Show 12 Predictions (SAVE + SHOW)
# -------------------------------------------------
import random
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for ax in axes.flatten():
    idx = random.randint(0, len(test_dataset) - 1)
    img, true_label = test_dataset[idx]

    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        pred = torch.argmax(output).item()

    # Undo normalization for proper display: from [-1,1] back to [0,1]
    img_vis = img * 0.5 + 0.5
    img_vis = img_vis.squeeze().numpy()

    ax.imshow(img_vis, cmap="gray")
    ax.set_title(f"True: {true_label} | Pred: {pred}")
    ax.axis("off")

plt.tight_layout()
fig.savefig("outputs/sample_predictions.png", dpi=300, bbox_inches="tight")
plt.show()
