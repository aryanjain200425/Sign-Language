import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms


# --------------------------------------
# CNN Model (must match train/test code)
# --------------------------------------
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
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# --------------------------------------
# Load Model
# --------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("sign_model.pt", map_location=device))
model.eval()

print("Model Loaded: sign_model.pt")


# --------------------------------------
# Preprocessing Transform
# --------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# --------------------------------------
# Class Labels (0–24 except J & Z)
# --------------------------------------
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
labels.remove("J")  # Not included in dataset
labels.remove("Z")  # Not included in dataset


# --------------------------------------
# Start Webcam
# --------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Press 'q' to quit.")


# --------------------------------------
# Live Detection Loop
# --------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip for natural viewing
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for easier hand segmentation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define ROI (Region of Interest)
    x, y, w, h = 100, 100, 300, 300
    roi = gray[y:y+h, x:x+w]

    # Preprocess ROI
    try:
        img = transform(roi).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output).item()
            predicted_label = labels[pred]
    except:
        predicted_label = "?"

    # Draw ROI box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display prediction
    cv2.putText(frame, f"Prediction: {predicted_label}",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Sign Language Live Detection", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()