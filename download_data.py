"""
Script to download Sign Language MNIST dataset
"""
import urllib.request
import os

# Create data directory
os.makedirs("data", exist_ok=True)

# URLs for the Sign Language MNIST dataset
train_url = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/X.npy"
test_url = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/raw/master/Y.npy"

print("Note: The Sign Language MNIST dataset needs to be downloaded from Kaggle.")
print("Please download the dataset manually from:")
print("https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
print("\nDownload 'sign_mnist_train.csv' and 'sign_mnist_test.csv'")
print("and place them in the './data/' directory")
