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

# Import the MobileNet model
from MobileNet import MobileNet


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
# PGD attack (untargeted) on [0,1] inputs
# ----------------------------
@torch.enable_grad()
def pgd_attack(model, x, y, eps=0.05, alpha=0.01, steps=40, random_start=True, clamp_min=0.0, clamp_max=1.0):
    """
    Generates adversarial examples for x using PGD (l_infty).
    Assumes inputs are in [clamp_min, clamp_max].
    eps/alpha are in the same scale as x (e.g., MNIST ToTensor -> [0,1]).
    """
    model_device = next(model.parameters()).device
    x = x.detach().to(model_device)
    y = y.detach().to(model_device)

    # start from a random point in the epsilon-ball if desired
    if random_start:
        x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = x_adv.clamp(clamp_min, clamp_max)
    else:
        x_adv = x.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())

        # project back to the epsilon l_inf ball around x, then clip to image bounds
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = x_adv.clamp(clamp_min, clamp_max)

    return x_adv.detach()


# ----------------------------
# Training loop (with optional PGD adversarial training)
# ----------------------------
def train(model, train_loader, device, epochs=10, lr=0.001, adv=True, eps=0.05, alpha=0.01, pgd_steps=40, random_start=True, adv_lambda=1.0):
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
                images_adv = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=pgd_steps, random_start=random_start, clamp_min=0.0, clamp_max=1.0)
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
# Evaluation (clean or robust under PGD)
# ----------------------------
def evaluate_model(model, test_loader, device, robust=False, eps=0.05, alpha=0.01, pgd_steps=40):
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
            images_in = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=pgd_steps, random_start=True, clamp_min=0.0, clamp_max=1.0)
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

    tag = "Robust (PGD)" if robust else "Clean"
    print(f"{tag} Test Loss: {avg_loss:.4f}")
    print(f"{tag} Test Accuracy: {accuracy:.4f}")
    print(f"{tag} Test auROC: {auroc:.4f}")
    print(f"{tag} Test auPRC: {auprc:.4f}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="MNIST training code (PyTorch) with PGD adversarial training")
    parser.add_argument("--output", type=str, default="mnist_model_pgd.pt", help="Model output name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Adversarial training flags
    parser.add_argument("--adv-train", action="store_true", help="Enable PGD adversarial training")
    parser.add_argument("--eps", type=float, default=0.05, help="PGD epsilon (in [0,1] pixel scale)")
    parser.add_argument("--alpha", type=float, default=0.01, help="PGD step size (in [0,1] scale)")
    parser.add_argument("--pgd-steps", type=int, default=40, help="Number of PGD steps")
    parser.add_argument("--no-random-start", action="store_true", help="Disable random PGD start")
    parser.add_argument("--adv-lambda", type=float, default=1.0, help="λ in [0,1]: mix clean/adv loss (1.0 = pure adv)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = load_data(batch_size=args.batch_size)
    dummy_batch =  train_loader.dataset[0][0].unsqueeze(0)  # Get a single sample and add batch dimension
    
    # Initialize MobileNet dynamically for 1-channel, 28x28 inputs
    model = MobileNet(one_batch=dummy_batch, num_classes=10)

    # Train
    train(model, train_loader, device,
          epochs=args.epochs, lr=args.lr,
          adv=args.adv_train, eps=args.eps, alpha=args.alpha,
          pgd_steps=args.pgd_steps, random_start=not args.no_random_start,
          adv_lambda=args.adv_lambda)

    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

    # Evaluate (clean)
    print("Evaluate test dataset")
    evaluate_model(model, test_loader, device, robust=False)
    evaluate_model(model, test_loader, device, robust=True, eps=args.eps, alpha=args.alpha, pgd_steps=args.pgd_steps)

if __name__ == "__main__":
    main()