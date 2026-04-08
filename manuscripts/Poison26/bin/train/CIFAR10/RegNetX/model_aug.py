import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import os, sys, json

# Import the RegNetX model
from RegNetX import RegNetX_400MF


# ----------------------------
# Dataset loading
# ----------------------------
def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to [0,1]
    ])

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

# ----------------------------
# Training loop
# ----------------------------
def train(model, train_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_correct / total
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # compute AUROC and AUPRC
    y_true_onehot = np.eye(10)[y_true]
    auroc = roc_auc_score(y_true_onehot, y_pred, multi_class="ovr")
    auprc = average_precision_score(y_true_onehot, y_pred)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test auROC: {auroc:.4f}")
    print(f"Test auPRC: {auprc:.4f}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="MNIST training code (PyTorch) with Augmentation")
    parser.add_argument("--output", type=str, default="mnist_regnet_aug.pt", help="Model output name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load data
    train_loader, test_loader = load_data(batch_size=args.batch_size)
    dummy_batch =  train_loader.dataset[0][0].unsqueeze(0)  # Get a single sample and add batch dimension
    # Initialize RegNetX dynamically for 1-channel, 28x28 inputs and 10 classes
    model = RegNetX_400MF(one_batch=dummy_batch, num_classes=10)


    # Train
    train(model, train_loader, device, epochs=args.epochs)

    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

    # Evaluate
    print("Model statistics on test dataset")
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()