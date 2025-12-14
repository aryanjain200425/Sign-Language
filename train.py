import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from vit_model import ViT


# -------------------------------------------------
# Dataset Class
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
# Transformations
# -------------------------------------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------------------------------------
# Load Data
# -------------------------------------------------
train_dataset = SignLanguageMNIST("./data/sign_mnist_train.csv", transform=train_transform)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Validation transform (no augmentation)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_dataset = SignLanguageMNIST("./data/sign_mnist_test.csv", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# -------------------------------------------------
# CNN Model
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# -------------------------------------------------
# Args
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['cnn','vit'], default='cnn', help='Model type to train')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

# -------------------------------------------------
# Train Model
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device) if args.model == 'cnn' else ViT(img_size=28, patch_size=4, in_chans=1, num_classes=25).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

epochs = args.epochs
print(f"Training started with {args.model.upper()}...\n")

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0
best_epoch = 0

for epoch in range(epochs):
    # Training phase
    correct = 0
    total = 0
    epoch_loss = 0

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), f"sign_model_best_{args.model}.pt")
    
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

print(f"\nBest validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

# -------------------------------------------------
# Save Final Model
# -------------------------------------------------
torch.save(model.state_dict(), f"sign_model_{args.model}.pt")
print(f"Final model saved to sign_model_{args.model}.pt")
print(f"Best model saved to sign_model_best_{args.model}.pt")

# -------------------------------------------------
# Plot Training Curves
# -------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss
ax1.plot(range(1, epochs+1), train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
ax1.plot(range(1, epochs+1), val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
ax1.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot Accuracy
ax2.plot(range(1, epochs+1), train_accuracies, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
ax2.plot(range(1, epochs+1), val_accuracies, 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
ax2.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
ax2.axhline(y=best_val_acc, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Best Val Acc ({best_val_acc:.2f}%)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training and Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'outputs/training_curves_{args.model}.png', dpi=300, bbox_inches='tight')
print(f"\nTraining curves saved to outputs/training_curves_{args.model}.png")

# -------------------------------------------------
# Save Metrics to CSV
# -------------------------------------------------
metrics_df = pd.DataFrame({
    'Epoch': range(1, epochs+1),
    'Train_Loss': train_losses,
    'Train_Accuracy': train_accuracies,
    'Val_Loss': val_losses,
    'Val_Accuracy': val_accuracies
})
metrics_df.to_csv(f'outputs/training_metrics_{args.model}.csv', index=False)
print(f"Training metrics saved to outputs/training_metrics_{args.model}.csv")

plt.show()
