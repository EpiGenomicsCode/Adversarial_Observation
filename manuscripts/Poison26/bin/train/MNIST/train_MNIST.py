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


# ----------------------------
# Model definitions
# ----------------------------
class BasicMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AdvMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)           # 28 -> 26
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)          # 26 -> 24
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)  # 24 -> 12
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)          # 12 -> 10
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)          # 10 -> 8
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)  # 8 -> 4
        self.bn6 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.4)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=4)         # 4 -> 1
        self.bn7 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.4)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = torch.flatten(x, 1)
        x = self.dropout3(x)
        return self.fc(x)


# ----------------------------
# Dataset loading
# ----------------------------
def load_data(batch_size, augment=False):
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])

    test_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


# ----------------------------
# Adversarial attack helpers
# ----------------------------
@torch.enable_grad()
def fgsm_attack(model, x, y, eps=0.05):
    x = x.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    return (x + eps * grad.sign()).clamp(0.0, 1.0).detach()


@torch.enable_grad()
def pgd_attack(model, x, y, eps=0.05, alpha=0.01, steps=40, random_start=True):
    if random_start:
        x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)
    else:
        x_adv = x.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        grad = torch.autograd.grad(F.cross_entropy(model(x_adv), y), x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0.0, 1.0)

    return x_adv.detach()


# ----------------------------
# Training loop
# ----------------------------
def train(model, train_loader, device, training, epochs, lr, eps, alpha, pgd_steps, random_start, adv_lambda):
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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training == "fgsm":
                model.eval()
                images_adv = fgsm_attack(model, images, labels, eps=eps)
                model.train()
            elif training == "pgd":
                model.eval()
                images_adv = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=pgd_steps, random_start=random_start)
                model.train()
            else:
                images_adv = None

            optimizer.zero_grad(set_to_none=True)

            if images_adv is not None and adv_lambda >= 1.0 - 1e-8:
                outputs = model(images_adv)
                loss = criterion(outputs, labels)
            elif images_adv is not None:
                out_clean = model(images)
                out_adv = model(images_adv)
                loss = (1.0 - adv_lambda) * criterion(out_clean, labels) + adv_lambda * criterion(out_adv, labels)
                outputs = out_adv
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s - "
              f"Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {running_correct/total:.4f}")


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, test_loader, device, attack=None, eps=0.05, alpha=0.01, pgd_steps=40):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    y_true, y_pred = [], []
    test_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        if attack == "fgsm":
            images_in = fgsm_attack(model, images, labels, eps=eps)
        elif attack == "pgd":
            images_in = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=pgd_steps)
        else:
            images_in = images

        with torch.no_grad():
            outputs = model(images_in)
            loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_onehot = np.eye(10)[y_true]
    auroc = roc_auc_score(y_true_onehot, y_pred, multi_class="ovr")
    auprc = average_precision_score(y_true_onehot, y_pred)

    tag = {"fgsm": "Robust (FGSM)", "pgd": "Robust (PGD)"}.get(attack, "Clean")
    print(f"{tag} Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"{tag} Test Accuracy: {correct/total:.4f}")
    print(f"{tag} Test auROC: {auroc:.4f}")
    print(f"{tag} Test auPRC: {auprc:.4f}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="MNIST training script (basic or adv architecture, standard/aug/fgsm/pgd training)")
    parser.add_argument("--arch", choices=["basic", "adv"], default="basic", help="Model architecture")
    parser.add_argument("--training", choices=["standard", "fgsm", "pgd"], default="standard", help="Adversarial training regime")
    parser.add_argument("--aug", action="store_true", help="Apply data augmentation during training")
    parser.add_argument("--output", type=str, default="mnist_model.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=0.05, help="FGSM/PGD epsilon in [0,1] scale")
    parser.add_argument("--adv-lambda", type=float, default=1.0, help="Mix ratio: 1.0 = pure adversarial loss")
    parser.add_argument("--alpha", type=float, default=0.01, help="PGD step size")
    parser.add_argument("--pgd-steps", type=int, default=40, help="Number of PGD steps")
    parser.add_argument("--no-random-start", action="store_true", help="Disable random PGD initialization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"arch={args.arch}  training={args.training}  aug={args.aug}  device={device}")

    model = BasicMNISTModel() if args.arch == "basic" else AdvMNISTModel()
    train_loader, test_loader = load_data(args.batch_size, augment=args.aug)

    train(model, train_loader, device,
          training=args.training,
          epochs=args.epochs,
          lr=args.lr,
          eps=args.eps,
          alpha=args.alpha,
          pgd_steps=args.pgd_steps,
          random_start=not args.no_random_start,
          adv_lambda=args.adv_lambda)

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

    print("\nEvaluating on test set...")
    evaluate_model(model, test_loader, device)
    if args.training == "fgsm":
        evaluate_model(model, test_loader, device, attack="fgsm", eps=args.eps)
    elif args.training == "pgd":
        evaluate_model(model, test_loader, device, attack="pgd", eps=args.eps, alpha=args.alpha, pgd_steps=args.pgd_steps)


if __name__ == "__main__":
    main()
