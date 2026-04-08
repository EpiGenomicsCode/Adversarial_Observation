import argparse
import os
import glob
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import librosa
from sklearn.metrics import roc_auc_score, average_precision_score

import hashlib
import csv

# ----------------------------
# Constants
# ----------------------------
SAMPLING_RATE = 16000
NUM_CLASSES = 10
MAX_AUDIO_LENGTH = 16000
BATCH_SIZE = 32

# ----------------------------
# Audio Preprocessing
# ----------------------------
def normalize_audio(x):
    return x / np.max(np.abs(x))

def pad_audio(audio, max_len=MAX_AUDIO_LENGTH):
    return audio[:max_len] if len(audio) > max_len else np.pad(audio, (0, max_len - len(audio)), 'constant')

# ----------------------------
# Dataset
# ----------------------------
class AudioMNISTDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        self.labels = []

        wav_files = glob.glob(os.path.join(data_path, '*', '*.wav'))
        # Deterministic shuffle using md5 hash of path
        wav_files = sorted(wav_files, key=lambda x: hashlib.md5(x.encode()).hexdigest())
        self.wav_files = wav_files.copy()  # store for TSV

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
        audio = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = self.labels[idx]
        return audio, label

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
# Load Data
# ----------------------------
def load_data(data_path, batch_size, split_tsv="split_indices_model1.tsv"):
    dataset = AudioMNISTDataset(data_path)
    # Fixed 80/20 split (after deterministic shuffle)
    train_size = int(0.8 * len(dataset))
    train_indices = list(range(0, train_size))
    test_indices  = list(range(train_size, len(dataset)))

    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    test_dataset  = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Write split info to TSV ---
    with open(split_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "split", "label", "file_path"])
        for idx in train_indices:
            writer.writerow([idx, "train", dataset.labels[idx], dataset.wav_files[idx]])
        for idx in test_indices:
            writer.writerow([idx, "test", dataset.labels[idx], dataset.wav_files[idx]])
    print(f"Saved split information to {split_tsv}")
    
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
    model = AudioMNISTModel()

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

