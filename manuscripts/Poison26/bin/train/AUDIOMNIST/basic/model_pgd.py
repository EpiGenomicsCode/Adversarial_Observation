import argparse
import os
import glob
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

import librosa
from sklearn.metrics import roc_auc_score, average_precision_score

import hashlib
import csv
import random

# ----------------------------
# Constants
# ----------------------------
SAMPLING_RATE = 16000
NUM_CLASSES = 10
MAX_AUDIO_LENGTH = 16000

# ----------------------------
# Audio Preprocessing
# ----------------------------
def normalize_audio(x):
    max_val = np.max(np.abs(x))
    return x / max_val if max_val > 0 else x

def pad_audio(audio, max_len=MAX_AUDIO_LENGTH):
    return audio[:max_len] if len(audio) > max_len else np.pad(audio, (0, max_len - len(audio)), 'constant')

# ----------------------------
# Dataset & Wrapper
# ----------------------------
class AudioMNISTBaseDataset(Dataset):
    """Loads all audio into memory ONCE. Returns raw numpy arrays."""
    def __init__(self, data_path):
        self.data = []
        self.labels = []

        wav_files = glob.glob(os.path.join(data_path, '*', '*.wav'))
        wav_files = sorted(wav_files, key=lambda x: hashlib.md5(x.encode()).hexdigest())
        self.wav_files = wav_files.copy()

        for audio_path in tqdm(wav_files, desc="Loading audio files"):
            audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE)
            audio = normalize_audio(audio)
            audio = pad_audio(audio)
            label = int(os.path.basename(audio_path)[0])
            self.data.append(audio)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class AudioSubsetWrapper(Dataset):
    """Wraps a subset to apply dynamic augmentation and convert to Tensors."""
    def __init__(self, subset, augment=False):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def apply_augmentation(self, x):
        if random.random() < 0.5:
            x = np.clip(x + np.random.randn(len(x)) * 0.005, -1.0, 1.0) # noise
        if random.random() < 0.5:
            x = np.roll(x, np.random.randint(-200, 200)) # time shift
        if random.random() < 0.5:
            x = np.clip(x * np.random.uniform(0.8, 1.2), -1.0, 1.0) # random gain
        return x

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.augment:
            x = self.apply_augmentation(x)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, length)
        return x, y

def load_data(data_path, batch_size, augment_train=False, split_tsv="split_indices_standard.tsv"):
    base_dataset = AudioMNISTBaseDataset(data_path)
    
    train_size = int(0.8 * len(base_dataset))
    train_indices = list(range(0, train_size))
    test_indices  = list(range(train_size, len(base_dataset)))

    # Isolate augmentation using the wrapper
    train_dataset = AudioSubsetWrapper(Subset(base_dataset, train_indices), augment=augment_train)
    test_dataset  = AudioSubsetWrapper(Subset(base_dataset, test_indices), augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with open(split_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "split", "label", "file_path"])
        for idx in train_indices:
            writer.writerow([idx, "train", base_dataset.labels[idx], base_dataset.wav_files[idx]])
        for idx in test_indices:
            writer.writerow([idx, "test", base_dataset.labels[idx], base_dataset.wav_files[idx]])
    print(f"Saved split information to {split_tsv}")

    return train_loader, test_loader

# ----------------------------
# Model Definition
# ----------------------------
class AudioMNISTModel(nn.Module):
    def __init__(self, input_length=MAX_AUDIO_LENGTH):
        super(AudioMNISTModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)

        conv_out_length = self._get_conv_output_length(input_length)
        self.fc1 = nn.Linear(64 * conv_out_length, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def _get_conv_output_length(self, input_length):
        length = (input_length - 5 + 1) // 2
        length = (length - 5 + 1) // 2
        return length

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ----------------------------
# PGD attack (untargeted) on [-1, 1] inputs
# ----------------------------
@torch.enable_grad()
def pgd_attack(model, x, y, eps=0.05, alpha=0.01, steps=10, random_start=True, clamp_min=-1.0, clamp_max=1.0):
    model_device = next(model.parameters()).device
    x = x.detach().to(model_device)
    y = y.detach().to(model_device)

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

        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = x_adv.clamp(clamp_min, clamp_max)

    return x_adv.detach()

# ----------------------------
# Training loop
# ----------------------------
def train(model, train_loader, device, epochs=10, lr=0.001, adv=True, eps=0.05, alpha=0.01, pgd_steps=10, random_start=True, adv_lambda=1.0):
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

            if adv:
                model.eval()
                images_adv = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=pgd_steps, random_start=random_start, clamp_min=-1.0, clamp_max=1.0)
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

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_correct / total
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} finished in {elapsed:.2f}s - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(model, test_loader, device, robust=False, eps=0.05, alpha=0.01, pgd_steps=10):
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
            images_in = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=pgd_steps, random_start=True, clamp_min=-1.0, clamp_max=1.0)
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
    y_true_onehot = np.eye(NUM_CLASSES)[y_true]
    
    auroc = roc_auc_score(y_true_onehot, y_pred, multi_class="ovr")
    auprc = average_precision_score(y_true_onehot, y_pred)

    tag = "Robust (PGD)" if robust else "Clean"
    print(f"{tag} Test Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | auROC: {auroc:.4f} | auPRC: {auprc:.4f}")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="AudioMNIST PGD Adversarial Training")
    parser.add_argument("--data", type=str, default="./data/AudioMNIST")
    parser.add_argument("--output", type=str, default="audiomnist_pgd.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    parser.add_argument("--adv-train", action="store_true", help="Enable PGD adversarial training")
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--pgd-steps", type=int, default=10) # Note: 10 steps is standard for audio to save time
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument("--adv-lambda", type=float, default=1.0)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioMNISTModel()
    train_loader, test_loader = load_data(args.data, args.batch_size, augment_train=False, split_tsv="split_indices_pgd.tsv")

    train(model, train_loader, device, epochs=args.epochs, lr=args.lr, 
          adv=args.adv_train, eps=args.eps, alpha=args.alpha, 
          pgd_steps=args.pgd_steps, random_start=not args.no_random_start, adv_lambda=args.adv_lambda)

    torch.save(model.state_dict(), args.output)
    
    print("\nEvaluate test dataset")
    evaluate_model(model, test_loader, device, robust=False)
    evaluate_model(model, test_loader, device, robust=True, eps=args.eps, alpha=args.alpha, pgd_steps=args.pgd_steps)

if __name__ == "__main__":
    main()