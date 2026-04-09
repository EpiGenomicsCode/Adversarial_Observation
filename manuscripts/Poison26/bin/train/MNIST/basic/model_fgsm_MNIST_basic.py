import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score, average_precision_score
import os, sys, json


# ----------------------------
# Model definition
# ----------------------------
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # raw logits
        return x

# ----------------------------
# Dataset loading
# ----------------------------
def load_data(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to [0,1]
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ----------------------------
# FGSM attack (untargeted) on [0,1] inputs
# ----------------------------
@torch.enable_grad()
def fgsm_attack(model, x, y, eps=0.05, clamp_min=0.0, clamp_max=1.0):
    """
    Generates adversarial examples for x using FGSM (l_infty).
    Assumes inputs are in [clamp_min, clamp_max].
    eps is in the same scale as x (e.g., MNIST ToTensor -> [0,1]).
    """
    model_device = next(model.parameters()).device
    x = x.detach().to(model_device)
    y = y.detach().to(model_device)

    x.requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
    x_adv = x + eps * torch.sign(grad.detach())

    # project back to [clamp_min, clamp_max]
    x_adv = x_adv.clamp(clamp_min, clamp_max)
    return x_adv.detach()

# ----------------------------
# Training loop (with optional FGSM adversarial training)
# ----------------------------
def train(model, train_loader, device, epochs=10, lr=0.001, adv=True, eps=0.05, adv_lambda=1.0):
    """
    adv=True: use adversarial examples in training.
    adv_lambda in [0,1]: loss = (1-λ)*CE(clean) + λ*CE(adv). Set λ=1.0 for pure adversarial training.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # optionally craft adversarial batch with BN/Dropout frozen for stability
            if adv:
                model.eval()
                images_adv = fgsm_attack(model, images, labels, eps=eps, clamp_min=0.0, clamp_max=1.0)
                model.train()
            else:
                images_adv = None

            optimizer.zero_grad(set_to_none=True)

            if adv and adv_lambda >= 1.0 - 1e-8:
                outputs = model(images_adv)
                loss = criterion(outputs, labels)
            elif adv and 0.0 < adv_lambda < 1.0:
                out_clean = model(images)
                out_adv   = model(images_adv)
                loss = (1.0 - adv_lambda) * criterion(out_clean, labels) + adv_lambda * criterion(out_adv, labels)
                outputs = out_adv  # for accuracy, count the adv preds
            else:
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
# Evaluation (clean or robust under FGSM)
# ----------------------------
def evaluate_model(model, test_loader, device, robust=False, eps=0.05):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()

    y_true = []
    y_pred = []

    test_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        if robust:
            # generate test-time adversarial examples with BN/Dropout frozen
            images_in = fgsm_attack(model, images, labels, eps=eps, clamp_min=0.0, clamp_max=1.0)
        else:
            images_in = images

        with torch.no_grad():
            outputs = model(images_in)
            loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_onehot = np.eye(10)[y_true]
    auroc = roc_auc_score(y_true_onehot, y_pred, multi_class="ovr")
    auprc = average_precision_score(y_true_onehot, y_pred)

    tag = "Robust (FGSM)" if robust else "Clean"
    print(f"{tag} Test Loss: {avg_loss:.4f}")
    print(f"{tag} Test Accuracy: {accuracy:.4f}")
    print(f"{tag} Test auROC: {auroc:.4f}")
    print(f"{tag} Test auPRC: {auprc:.4f}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="MNIST training code (PyTorch) with FGSM adversarial training")
    parser.add_argument("--output", type=str, default="mnist_model4.pt", help="Model output name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Adversarial training flags
    parser.add_argument("--adv-train", action="store_true", help="Enable FGSM adversarial training")
    parser.add_argument("--eps", type=float, default=0.05, help="FGSM epsilon (in [0,1] pixel scale)")
    parser.add_argument("--adv-lambda", type=float, default=1.0, help="λ in [0,1]: mix clean/adv loss (1.0 = pure adv)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = MNISTModel()

    # Load data
    train_loader, test_loader = load_data(batch_size=args.batch_size)

    # Train
    train(model, train_loader, device,
          epochs=args.epochs, lr=args.lr,
          adv=args.adv_train, eps=args.eps, adv_lambda=args.adv_lambda)

    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

    # Evaluate (clean)
    print("Evaluate test dataset")
    evaluate_model(model, test_loader, device, robust=False)
    evaluate_model(model, test_loader, device, robust=True, eps=args.eps)

if __name__ == "__main__":
    main()

